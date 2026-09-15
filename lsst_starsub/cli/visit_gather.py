"""
cli/visit_gather: consolidate the per-detector joint fits of a visit.

Reads the profiles files lsst-starsub-visit --profiles-only wrote with
the joint model for every detector of the visit, gives each star one
amplitude (lsst_starsub.visit.gather), prints the cross-detector check,
and writes {outdir}/amplitudes-{visit}-{band}.fits for the second pass
(lsst-starsub-visit --amplitudes).
"""
import os


def get_args():
    """
    Parse the command line.

    Returns
    -------
    args: argparse.Namespace
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--visit', type=int, required=True)
    parser.add_argument('--indir', required=True,
                        help='the pass-1 profiles files')
    parser.add_argument('--outdir', default=None,
                        help='default --indir')
    parser.add_argument('--detectors', default=None,
                        help='a file with one detector per line: use '
                             'only these (e.g. lsst-starsub-visit-detectors '
                             'with the field-radius cut)')
    parser.add_argument('--max-err', type=float, default=0.05,
                        help='amplitude error below which a star counts '
                             'as well constrained for the visit scale')
    return parser.parse_args()


def main():
    """
    Consolidate one visit.
    """
    from ..visit.gather import (
        consolidate, profiles_files, read_detector_stars, report,
        write_amplitudes,
    )

    args = get_args()
    outdir = args.indir if args.outdir is None else args.outdir
    os.makedirs(outdir, exist_ok=True)

    files = profiles_files(args.indir, args.visit)
    if not files:
        raise RuntimeError(f'no profiles files for visit {args.visit} '
                           f'in {args.indir}')
    keep = None
    if args.detectors is not None:
        keep = {int(line) for line in open(args.detectors) if line.strip()}
    tables = []
    detectors = []
    band = None
    for fname in files:
        stars, meta = read_detector_stars(fname)
        det = int(meta['detector'][0])
        if keep is not None and det not in keep:
            continue
        tables.append(stars)
        detectors.append(det)
        band = str(meta['band'][0])
    print(f'{len(detectors)} detectors of visit {args.visit} band {band}'
          + (f' ({len(files) - len(detectors)} left out)' if keep else ''))

    amplitudes, pairs = consolidate(tables, detectors)
    scale = report(amplitudes, pairs, max_err=args.max_err)
    fname = os.path.join(outdir, f'amplitudes-{args.visit}-{band}.fits')
    write_amplitudes(fname, amplitudes, pairs, args.visit, band, scale)


if __name__ == '__main__':
    main()
