"""
cli/visit_template: the pooled per-visit star template and wing
model (lsst_starsub.visit.template) from all, or a subset, of the
visit's detectors

Writes {outdir}/template-{visit}-{band}.fits and a png.  The
per-visit Gaia extract is made in --gaia-dir on first use
(gaia-dr3-visit-{visit}.fits, from the refcat shards)
"""
import os

from ..visit.exposure import visit_detectors as exposure_visit_detectors
import sys
import time

import numpy as np

_WORKER = {}


def get_args():
    """
    Parse the command line.

    Returns
    -------
    args: argparse.Namespace
    """
    import argparse
    from . import add_butler_arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--visit', type=int, required=True)
    parser.add_argument(
        '--detectors', type=int, nargs='+',
        help='these detectors only (default: every detector with a '
             'wcs and calibration in the visit summary)',
    )
    parser.add_argument(
        '--ndet', type=int,
        help='an evenly spread subset of this many detectors',
    )
    parser.add_argument('--gaia-dir', required=True)
    parser.add_argument('--outdir', required=True)
    add_butler_arguments(parser)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument(
        '--save-extracts', action='store_true',
        help='also write the per-detector extracts (stamps and wing '
             'tables) to {outdir}/extracts-{visit}/',
    )
    parser.add_argument(
        '--extract-only', action='store_true',
        help='write the extracts and stop, no pooling (implies '
             '--save-extracts; for one job per detector on slurm)',
    )
    parser.add_argument(
        '--from-extracts', action='store_true',
        help='pool the saved extracts in {outdir}/extracts-{visit}/ '
             'instead of extracting (no butler access)',
    )
    parser.add_argument(
        '--from-template',
        help='refit this pooled template file (its stored per-star wing '
             'profiles, stack profile and stack) with the current wing '
             'fit and write '
             'the result to {outdir}; no extracts or butler access',
    )
    return parser.parse_args()


def extract_one(butler, visit, detector, gaia_path):
    """
    Load one detector and extract its template inputs.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    visit, detector: int
    gaia_path: str
        The visit's Gaia file

    Returns
    -------
    extract: dict
        From lsst_starsub.visit.template.extract_detector
    """
    from ..visit.template import extract_detector
    from ..visit.exposure import load_gaia_for_exposure, load_visit_exposure

    t0 = time.time()
    vexp = load_visit_exposure(butler, visit, detector)
    gaia = load_gaia_for_exposure(vexp, gaia_file=gaia_path)
    out = extract_detector(vexp, gaia)
    print(f'    detector {detector} done in {time.time() - t0:.0f} s')
    return out


def _worker(args):
    """
    Run extract_one in a pool process, with a butler made once.

    Parameters
    ----------
    args: tuple
        repo, collection, visit, detector, gaia_path

    Returns
    -------
    extract: dict
        None when the detector failed
    """
    from ..visit.exposure import make_visit_butler

    repo, collection, visit, detector, gaia_path = args
    if 'butler' not in _WORKER:
        _WORKER['butler'] = make_visit_butler(repo, collection)
    try:
        return extract_one(_WORKER['butler'], visit, detector, gaia_path)
    except Exception as err:
        print(f'    detector {detector} FAILED: {err!r}')
        return None


def read_extract(fname):
    """
    Read a saved per-detector extract.

    Parameters
    ----------
    fname: str

    Returns
    -------
    extract: dict
        As extract_detector returns it
    """
    import rustfits
    from ..visit.template import wing_edges

    with rustfits.FITS(fname) as fits:
        hdr = fits['stamps'].header
        out = dict(
            visit=int(hdr['visit']), detector=int(hdr['detector']),
            band=str(hdr['band']).strip(), fwhm=float(hdr['fwhm']),
            sky_sigma=float(hdr['sky_sigma']), calib=float(hdr['calib']),
            ambient_flat=float(hdr['ambient_flat']),
            ambient_warp=float(hdr['ambient_warp']),
            stamps=fits['stamps'].read(), stamp_G=fits['stamp_G'].read(),
            wing=fits['wing'].read(), edges=wing_edges(),
        )
        if 'stamp_amp' in fits:
            out['stamp_amp'] = fits['stamp_amp'].read()
    return out


def write_extract(fname, ext):
    """
    Write a per-detector extract.

    Parameters
    ----------
    fname: str
    ext: dict
        From extract_detector
    """
    import rustfits

    with rustfits.FITS(fname, 'w+') as fits:
        hdr = {k: ext[k] for k in ('visit', 'detector', 'band', 'fwhm',
                                   'sky_sigma', 'calib', 'ambient_flat',
                                   'ambient_warp')}
        hdr = {k: (v if not isinstance(v, float) or np.isfinite(v)
                   else -1.0) for k, v in hdr.items()}
        fits.write_image(ext['stamps'], extname='stamps', header=hdr)
        fits.write_image(ext['stamp_G'], extname='stamp_G')
        fits.write_image(ext['stamp_amp'], extname='stamp_amp')
        fits.write_table(ext['wing'], extname='wing')


def main():
    """
    Extract the detectors of a visit and pool their template.
    """
    from ..visit.gaia import ensure_visit_gaia_file
    from ..visit.exposure import make_visit_butler

    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    visit = args.visit
    edir = os.path.join(args.outdir, f'extracts-{visit}')

    if args.from_template:
        from ..visit.template import read_template_file, refit_template
        pooled = refit_template(read_template_file(args.from_template))
        write_pooled(args, pooled)
        return

    if args.from_extracts:
        import glob
        files = sorted(glob.glob(os.path.join(edir, 'extract-*.fits')))
        extracts = [read_extract(f) for f in files]
        print(f'visit {visit}: {len(extracts)} saved extracts')
        pool_and_write(args, extracts)
        return

    butler = make_visit_butler(args.repo, args.collection)

    gaia_path = ensure_visit_gaia_file(butler, visit, args.gaia_dir)
    dets = args.detectors or exposure_visit_detectors(butler, visit)
    if args.ndet is not None and args.ndet < len(dets):
        idx = np.unique(np.round(
            np.linspace(0, len(dets) - 1, args.ndet)
        ).astype(int))
        dets = [dets[i] for i in idx]
    print(f'visit {visit}: {len(dets)} detectors, {args.nproc} processes')

    jobs = [(args.repo, args.collection, visit, d, gaia_path) for d in dets]
    t0 = time.time()
    if args.nproc > 1:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(args.nproc) as pool:
            extracts = list(pool.imap_unordered(_worker, jobs))
    else:
        extracts = [_worker(j) for j in jobs]
    extracts = [e for e in extracts if e is not None]
    extracts.sort(key=lambda e: e['detector'])
    print(f'{len(extracts)} detectors extracted in {time.time() - t0:.0f} s')
    if len(extracts) == 0:
        raise RuntimeError('no detector extracted')

    if args.save_extracts or args.extract_only:
        os.makedirs(edir, exist_ok=True)
        for e in extracts:
            name = f'extract-{visit}-{e["detector"]:03d}.fits'
            fname = os.path.join(edir, name)
            write_extract(fname + '.tmp', e)
            os.replace(fname + '.tmp', fname)
            print('wrote', fname)
    if args.extract_only:
        return

    pool_and_write(args, extracts)


def pool_and_write(args, extracts):
    """
    Pool the extracts and write the template file and its png.

    Parameters
    ----------
    args: argparse.Namespace
        From get_args
    extracts: list of dict
        From extract_detector
    """
    from ..visit.template import pool_visit

    write_pooled(args, pool_visit(extracts))


def write_pooled(args, pooled):
    """
    Write a pooled template's file and png to the output directory.

    Parameters
    ----------
    args: argparse.Namespace
        From get_args
    pooled: dict
        From pool_visit or refit_template
    """
    from ..visit.template import plot_template, write_template_file

    band = pooled['params']['band']
    stem = os.path.join(args.outdir, f'template-{args.visit}-{band}')
    write_template_file(stem + '.fits', pooled)
    try:
        plot_template(stem + '.png', pooled)
    except Exception as err:
        print(f'    WARNING: plot failed: {err!r}')


if __name__ == '__main__':
    main()
