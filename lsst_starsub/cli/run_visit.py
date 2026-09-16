"""
cli/run_visit: the joint star-wing and sky characterization on
the visit-detectors that went into a patch's coadd, best IQ
first, or on one given visit/detector

Per detector it writes {outdir}/{tract}-{patch}-{band}-{visit}-
{detector}.fits with the image states (delivered, restored,
sky, star_model, residual), the mask and star mask, the gaia
census with fitted amplitudes, the per-star profile table, and
a summary png of the stacked profiles and a bright-star cutout
in each state
"""
import os

import numpy as np


def get_args():
    """
    Parse the command line.

    Returns
    -------
    args: argparse.Namespace
    """
    import argparse
    from . import add_butler_arguments
    from ..census import GSUB

    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--outdir', required=True)
    add_butler_arguments(parser)
    parser.add_argument(
        '--nbest', type=int, default=1,
        help='process this many coadd inputs, best IQ first',
    )
    parser.add_argument(
        '--visit', type=int, help='process this visit only',
    )
    parser.add_argument(
        '--detector', type=int, help='with --visit: this detector',
    )
    parser.add_argument('--gsub', type=float, default=GSUB)
    parser.add_argument(
        '--gaia-file',
        help='gaia stars from this file instead of the TAP query',
    )
    parser.add_argument(
        '--restore', default='initial',
        choices=['initial', 'fine', 'none'],
        help='which stored background layers to add back',
    )
    parser.add_argument('--nround', type=int, default=2)
    parser.add_argument(
        '--star-model', default='template', choices=['template', 'joint'],
        help='the star model: the stamp templates of the image '
             '(default) or the joint fit of the canonical wing and the '
             'sky mesh, which needs --canonical',
    )
    parser.add_argument(
        '--canonical',
        help='the canonical wing file of the band for --star-model joint',
    )
    parser.add_argument(
        '--edge-factor', type=float, default=None,
        help='joint model: also fit the stars across the detector edge '
             'whose center is within this many mask radii of the image '
             '(pass 1 of the visit scheme); their profiles are measured '
             'too',
    )
    parser.add_argument(
        '--amplitudes',
        help='joint model: pin every star to its amplitude from this '
             'lsst-starsub-visit-gather file and refit the sky alone '
             '(pass 2 of the visit scheme)',
    )
    parser.add_argument(
        '--core-rap', type=float, default=None,
        help='joint model: measure the unsaturated stars\' amplitudes '
             'from the flux within this many px of their centers and '
             'pin them; --canonical must then be the visit\'s own wing '
             '(lsst-starsub-visit-wing)',
    )
    parser.add_argument(
        '--no-plot', action='store_true', help='skip the png',
    )
    parser.add_argument(
        '--profiles-only', action='store_true',
        help='write only the profile tables and the census '
             '(profiles-{stem}.fits, as lsst-starsub-remeasure '
             'writes), not the image planes; implies --no-plot',
    )
    parser.add_argument(
        '--list-inputs', action='store_true',
        help='print the selected inputs (visit detector iq_score, '
             'best first) and exit, for driving parallel workers',
    )
    return parser.parse_args()


def output_name(outdir, tract, patch, band, visit, detector, ext):
    """
    Build the per-detector output path.

    Parameters
    ----------
    outdir: str
    tract, patch: int
    band: str
    visit, detector: int
    ext: str
        The extension, without the dot

    Returns
    -------
    path: str
    """
    return os.path.join(
        outdir,
        f'{tract:05d}-{patch:02d}-{band}-{visit}-{detector:03d}.{ext}',
    )


def joint_outputs(res, meta):
    """
    Build the joint model's outputs: the census with errors, the nodes.

    Parameters
    ----------
    res: dict
        From handle_stars_visit with the joint model
    meta: dict
        The run meta, extended in place with the fit's settings and
        quality

    Returns
    -------
    star_table: structured array
        The census table with A, A_err and free
    extra: dict
        'nodes': the sky mesh nodes (ix, iy, x, y, value, err)
    """
    from numpy.lib import recfunctions as rfn

    jf = res['joint']
    star_table = rfn.append_fields(
        res['star_table'], ['A_err', 'free'],
        [jf['A_err'].astype('f8'), jf['free'].astype('i2')],
        usemask=False,
    )
    xn, yn = jf['nodes']
    iy, ix = np.mgrid[0:yn.size, 0:xn.size]
    nodes = np.zeros(ix.size, dtype=[
        ('ix', 'i2'), ('iy', 'i2'), ('x', 'f4'), ('y', 'f4'),
        ('value', 'f4'), ('err', 'f4'),
    ])
    nodes['ix'] = ix.ravel()
    nodes['iy'] = iy.ravel()
    nodes['x'] = xn[ix.ravel()]
    nodes['y'] = yn[iy.ravel()]
    nodes['value'] = np.asarray(jf['node_values']).ravel()
    nodes['err'] = np.asarray(jf['node_err']).ravel()
    meta.update(
        spacing=float(xn[1] - xn[0]), nxnode=int(xn.size),
        nynode=int(yn.size), ny=int(jf['sky'].shape[0]),
        nx=int(jf['sky'].shape[1]), chi2=float(jf['chi2']),
        nfree=int(jf['free'].sum()),
    )
    return star_table, {'nodes': nodes}


def process_one(butler, visit, detector, args, iq_score=np.nan):
    """
    Characterize one visit-detector and write its outputs.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    visit, detector: int
    args: argparse.Namespace
        From get_args
    iq_score: float, optional
        The shapelets IQ score, for the record
    """
    from ..census import field_segmentation
    from ..visit.profiles import ambient_levels, measure_profiles
    from ..visit.exposure import (
        build_wide_star_mask,
        handle_stars_visit,
        iq_tier,
        load_gaia_for_exposure,
        load_visit_exposure,
    )
    from ..coadd.io import write_visit_file, plot_summary

    print(
        f'visit {visit} detector {detector} '
        f'iq {iq_score:.2e} ({iq_tier(iq_score)})'
    )
    vexp = load_visit_exposure(butler, visit, detector)
    band = vexp.band
    gaia = load_gaia_for_exposure(
        vexp, gaia_file=args.gaia_file, gmax=max(args.gsub, 19.0),
    )

    joint = {}
    if args.star_model == 'joint':
        from ..wing import read_wing_model
        if args.canonical is None:
            raise ValueError('--star-model joint needs --canonical')
        joint = dict(
            star_model='joint',
            canonical=read_wing_model(args.canonical),
            edge_factor=args.edge_factor,
            core_rap=args.core_rap,
        )
        if args.amplitudes is not None:
            from ..visit.gather import read_amplitudes
            joint['amplitudes'] = read_amplitudes(args.amplitudes)

    res = handle_stars_visit(
        vexp, gaia, gsub=args.gsub, restore=args.restore,
        nround=args.nround, **joint,
    )

    residual = vexp.image.array
    # the restored image, sky-flattened by our model: the
    # wings-on-flat-sky state the template was fit to
    flat = res['delivered'] + res['restored'] - res['sky']
    # the skyCorr state: the delivered image with the visit-level sky
    # correction applied, the input a warp would be made from if it
    # were.  The DP2 deep coadd's warps do not apply it (makeDirectWarp
    # doApplyNewBackground=False, verified 2026-09-07), so the coadd
    # carries the delivered state; the pretty warps apply it
    skycorr = res['delivered'] - vexp.backgrounds['skycorr']
    # the pipeline's sky with our star model removed, at both levels:
    # what the pipeline's product would look like with only the stars
    # fixed, against residual with the sky refit too.  (Files written
    # before 2026-09-15 call the skycorr states 'warp'.)
    states = dict(
        delivered=res['delivered'],
        skycorr=skycorr,
        flat=flat,
        residual=residual,
        delivered_starsub=res['delivered'] - res['star_model'],
        skycorr_starsub=skycorr - res['star_model'],
    )
    seg = field_segmentation(residual, vexp.good, vexp.sky_sigma)
    # each state referenced to its own ambient level, measured
    # outside the wide star exclusion
    wide = build_wide_star_mask(res['stars'], seg.shape)
    ambient = ambient_levels(states, vexp, seg, wide)
    print('    ambient levels (nJy): ' + ', '.join(
        f'{k} {v:.2f}' for k, v in ambient.items()
    ))
    # with the edge stars in the fit, their wings on this detector
    # are measured too
    on_image_only = args.edge_factor is None
    edges, ptable = measure_profiles(
        states, vexp, res['stars'], seg, ambient=ambient,
        on_image_only=on_image_only,
    )
    dedges, dtable = measure_profiles(
        states, vexp, res['stars'], seg, ambient=ambient,
        mode='dmask', on_image_only=on_image_only,
    )

    meta = dict(
        tract=args.tract, patch=args.patch, band=band,
        visit=visit, detector=detector, iq_score=iq_score,
        restore=args.restore, nround=args.nround,
        gsub=args.gsub, collection=args.collection,
        fwhm=res['fwhm'] if res['fwhm'] is not None else -1.0,
        star_model=args.star_model,
        edge_factor=-1.0 if args.edge_factor is None else args.edge_factor,
        amplitudes='' if args.amplitudes is None else args.amplitudes,
        core_rap=-1.0 if args.core_rap is None else args.core_rap,
        canonical='' if args.canonical is None else args.canonical,
    )
    star_table = res['star_table']
    extra = None
    if joint:
        star_table, extra = joint_outputs(res, meta)
    if args.profiles_only:
        from ..coadd.io import write_profiles_file
        from ..visit.profiles import state_maps
        stem = os.path.basename(output_name(
            args.outdir, args.tract, args.patch, band, visit,
            detector, 'fits',
        ))[:-5]
        # the image states binned to box medians with the sources
        # masked (the star masks and the segmentation: a bright
        # extended source in a box median is a 100-1000 nJy outlier),
        # and the three total sky models (the pipeline's per-detector
        # and focal-plane skies, ours) unmasked: the sky at the nJy
        # level across the focal plane from the small files
        # (scripts/focal_plane_mosaic.py)
        usable = vexp.good & ~res['starmask'] & (seg == 0)
        dm_initial = (vexp.backgrounds['initial_coarse']
                      + vexp.backgrounds['initial_fine'])
        skies = dict(
            sky_delivered=dm_initial,
            sky_skycorr=dm_initial + vexp.backgrounds['skycorr'],
            sky_starsub=dm_initial - res['restored'] + res['sky'],
        )
        maps = state_maps(states, usable)
        maps.update(state_maps(skies, np.ones(usable.shape, dtype=bool)))
        write_profiles_file(
            os.path.join(args.outdir, f'profiles-{stem}.fits'),
            dedges, dtable, meta, star_table=star_table,
            rtable=(edges, ptable), extra=extra, maps=maps,
        )
        return

    fname = output_name(
        args.outdir, args.tract, args.patch, band, visit, detector,
        'fits',
    )
    write_visit_file(
        fname, vexp, res, states, edges, ptable,
        dmask=(dedges, dtable), meta=meta,
    )
    if not args.no_plot:
        png = output_name(
            args.outdir, args.tract, args.patch, band, visit,
            detector, 'png',
        )
        try:
            plot_summary(png, vexp, res, states, edges, ptable)
        except Exception as err:
            # the data are written; a diagnostic plot failure
            # must not fail the detector
            print(f'    WARNING: summary plot failed: {err!r}')


def main():
    """
    Characterize the coadd inputs of a patch, or one visit-detector.
    """
    from ..visit.exposure import (
        load_iq_scores,
        make_visit_butler,
        select_coadd_inputs,
    )

    import sys
    # the log is usually a file: flush per line so progress
    # shows while the butler and the fits are working
    sys.stdout.reconfigure(line_buffering=True)

    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    butler = make_visit_butler(args.repo, args.collection)

    if args.visit is not None:
        if args.detector is None:
            raise ValueError('--visit needs --detector')
        todo = [(args.visit, args.detector, np.nan)]
    else:
        iq = load_iq_scores(butler)
        inputs = select_coadd_inputs(
            butler, args.tract, args.patch, args.band, iq=iq,
        )
        todo = [
            (int(r['visit']), int(r['detector']), float(r['iq_score']))
            for r in inputs[:args.nbest]
        ]

    if args.list_inputs:
        for visit, detector, score in todo:
            print(f'{visit} {detector} {score:.3e}')
        return

    for visit, detector, score in todo:
        process_one(butler, visit, detector, args, iq_score=score)


if __name__ == '__main__':
    main()
