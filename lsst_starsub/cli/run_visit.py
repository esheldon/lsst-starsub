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
    import argparse
    from ..visit import VISIT_COLLECTION, VISIT_REPO
    from lsst_mdet.starsub import GSUB

    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--repo', default=VISIT_REPO)
    parser.add_argument('--collection', default=VISIT_COLLECTION)
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
    return os.path.join(
        outdir,
        f'{tract:05d}-{patch:02d}-{band}-{visit}-{detector:03d}.{ext}',
    )


def process_one(butler, visit, detector, args, iq_score=np.nan):
    from lsst_mdet.starsub import field_segmentation
    from ..profiles import ambient_levels, measure_profiles
    from ..visit import (
        build_wide_star_mask, handle_stars_visit, iq_tier,
        load_gaia_for_exposure, load_visit_exposure,
    )
    from ..io import write_visit_file, plot_summary

    print(
        f'visit {visit} detector {detector} '
        f'iq {iq_score:.2e} ({iq_tier(iq_score)})'
    )
    vexp = load_visit_exposure(butler, visit, detector)
    band = vexp.band
    gaia = load_gaia_for_exposure(
        vexp, gaia_file=args.gaia_file, gmax=max(args.gsub, 19.0),
    )

    res = handle_stars_visit(
        vexp, gaia, gsub=args.gsub, restore=args.restore,
        nround=args.nround,
    )

    residual = vexp.image.array
    # the restored image, sky-flattened by our model: the
    # wings-on-flat-sky state the template was fit to
    flat = res['delivered'] + res['restored'] - res['sky']
    # the warp state: what the coadd inputs carried (the
    # delivered image with the visit-level sky correction)
    warp = res['delivered'] - vexp.backgrounds['skycorr']
    states = dict(
        delivered=res['delivered'],
        warp=warp,
        flat=flat,
        residual=residual,
    )
    seg = field_segmentation(residual, vexp.good, vexp.sky_sigma)
    # each state referenced to its own ambient level, measured
    # outside the wide star exclusion
    wide = build_wide_star_mask(res['stars'], seg.shape)
    ambient = ambient_levels(states, vexp, seg, wide)
    print('    ambient levels (nJy): ' + ', '.join(
        f'{k} {v:.2f}' for k, v in ambient.items()
    ))
    edges, ptable = measure_profiles(
        states, vexp, res['stars'], seg, ambient=ambient,
    )
    dedges, dtable = measure_profiles(
        states, vexp, res['stars'], seg, ambient=ambient,
        mode='dmask',
    )

    meta = dict(
        tract=args.tract, patch=args.patch, band=band,
        visit=visit, detector=detector, iq_score=iq_score,
        restore=args.restore, nround=args.nround,
        gsub=args.gsub, collection=args.collection,
        fwhm=res['fwhm'] if res['fwhm'] is not None else -1.0,
    )
    if args.profiles_only:
        from ..io import write_profiles_file
        stem = os.path.basename(output_name(
            args.outdir, args.tract, args.patch, band, visit,
            detector, 'fits',
        ))[:-5]
        write_profiles_file(
            os.path.join(args.outdir, f'profiles-{stem}.fits'),
            dedges, dtable, meta, star_table=res['star_table'],
            rtable=(edges, ptable),
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
    from ..visit import (
        load_iq_scores, make_visit_butler, select_coadd_inputs,
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
