"""
cli/remeasure: re-measure the per-star d - r_mask profiles from
the image states stored in lsst-starsub-visit outputs, without
rerunning the characterization, and write them to small
profile files (profiles-{stem}.fits with profiles_dmask and
dmask_edges) that lsst-starsub-stack reads like the originals
"""
import os

import numpy as np


def get_args():
    import argparse
    from ..profiles import LOCAL_REF

    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--outdir', required=True)
    parser.add_argument(
        '--states', nargs='+',
        default=['delivered', 'warp', 'flat', 'residual'],
    )
    parser.add_argument(
        '--no-wide', action='store_true',
        help='exclude other stars only to their mask circle plus '
             'taper, not their wide radius',
    )
    parser.add_argument(
        '--local-ref', nargs=2, type=float, default=list(LOCAL_REF),
        help='d - r_mask range of the local reference annulus',
    )
    return parser.parse_args()


def remeasure_one(fname, args):
    import rustfits
    from lsst_mdet.starsub import field_segmentation
    from ..profiles import ambient_levels, measure_profiles
    from ..visit import VisitExposure, build_wide_star_mask

    print('reading:', fname)
    with rustfits.FITS(fname) as fits:
        states = {}
        for name in args.states:
            try:
                states[name] = fits[name].read()
            except ValueError:
                if name != 'warp':
                    raise
                # earlier outputs stored the ingredients only
                states[name] = (
                    fits['delivered'].read() - fits['skycorr'].read()
                )
        var = fits['var'].read()
        mask = fits['mask'].read()
        stars = fits['gaia_stars'].read()
        meta = fits['meta'].read()

    ny, nx = var.shape
    vexp = VisitExposure(
        image=states['residual'] if 'residual' in states
        else states[args.states[0]],
        variance=var, mask=mask[:, :, np.newaxis], band=str(meta['band'][0]),
        backgrounds={}, noise=np.zeros((ny, nx), dtype='f4'),
    )
    seg = field_segmentation(vexp.image.array, vexp.good, vexp.sky_sigma)
    wide = build_wide_star_mask(stars, seg.shape)
    ambient = ambient_levels(states, vexp, seg, wide)
    edges, table = measure_profiles(
        states, vexp, stars, seg, ambient=ambient, mode='dmask',
        wide=not args.no_wide, local_ref=tuple(args.local_ref),
    )

    stem = os.path.basename(fname)[:-5]
    out = os.path.join(args.outdir, f'profiles-{stem}.fits')
    edges_t = np.zeros(1, dtype=[('edges', 'f8', edges.size)])
    edges_t['edges'][0] = edges
    print('writing:', out)
    with rustfits.FITS(out, 'w+') as fits:
        fits.write_table(table, extname='profiles_dmask')
        fits.write_table(edges_t, extname='dmask_edges')
        fits.write_table(meta, extname='meta')


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    for fname in args.files:
        remeasure_one(fname, args)


if __name__ == '__main__':
    main()
