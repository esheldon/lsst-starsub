"""
cli/cell_forward: the forward-modeled trough on one patch, from
the per-visit templates (lsst_starsub.trough), added to an
existing lsst-starsub-cell-restore output as the state

    forward = none + response coadd structure

the prediction of the restored state.  The profile tables are
remeasured over none, restored and forward with the same census
and masks, and written with the response coadd plane to
{outdir}/forward-{tract}-{patch}-{band}.fits
"""
import os
import sys

import numpy as np


def get_args():
    import argparse
    from ..visit import VISIT_COLLECTION, VISIT_REPO
    from lsst_mdet.starsub import GSUB

    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--cell-file', required=True,
                        help='lsst-starsub-cell-restore output with images')
    parser.add_argument('--templates', required=True,
                        help='directory of template-{visit}-{band}.fits')
    parser.add_argument('--visit-gaia-dir', required=True,
                        help='directory of the per-visit Gaia files')
    parser.add_argument('--gaia-file', required=True,
                        help='the tract Gaia file for the census')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--repo', default=VISIT_REPO)
    parser.add_argument('--collection', default=VISIT_COLLECTION)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--gsub', type=float, default=GSUB)
    return parser.parse_args()


def main():
    import rustfits
    from lsst_mdet.gaia import GMAX, gaia_pixel_positions, read_gaia_file
    from lsst_mdet.patchfiles import SimpleBox
    from lsst_mdet.starsub import (
        build_star_mask, field_segmentation, select_stars,
    )
    from lsst_mdet.wcs import ButlerWcs
    from ..coadd import coadd_data_id, load_cell_coadd, polynomial_cell_coadd
    from ..io import _meta_table
    from ..profiles import ambient_levels, measure_profiles
    from ..trough import ResponseCache
    from ..visit import VisitExposure, build_wide_star_mask, make_visit_butler

    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    butler = make_visit_butler(args.repo, args.collection)
    did = coadd_data_id(args.tract, args.patch, args.band)

    print('reading the cell file')
    with rustfits.FITS(args.cell_file) as fits:
        none = fits['none'].read()
        restored = fits['restored'].read()
        var = fits['var'].read()
        mask = fits['mask'].read()
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    print('loading the cell coadd for the inputs and weights')
    mcoadd = load_cell_coadd(butler, did)

    print('building the response coadd')
    cache = ResponseCache(
        butler, args.templates, args.visit_gaia_dir, args.band,
    )
    rcoadd, rlevel, wsum, count = polynomial_cell_coadd(
        butler, mcoadd, repo=args.repo, collection=args.collection,
        nproc=args.nproc, cache=cache,
    )
    forward = none + np.nan_to_num(rcoadd).astype('f4')
    states = dict(none=none, restored=restored, forward=forward)

    vexp = VisitExposure(
        image=none, variance=var, mask=mask, band=args.band,
        backgrounds={}, noise=np.zeros(none.shape, dtype='f4'),
    )
    bb = mcoadd.inner_bbox
    wcs = ButlerWcs(mcoadd.wcs)
    tbox = SimpleBox(bb.getBeginX(), bb.getEndX(), bb.getBeginY(),
                     bb.getEndY())
    gaia = read_gaia_file(
        args.gaia_file, wcs, tbox, gmax=max(args.gsub, GMAX),
    )
    x, y = gaia_pixel_positions(gaia, wcs, tbox)
    stars = select_stars(gaia, x, y, mask[:, :, 0], gsub=args.gsub)
    build_star_mask(stars, mask[:, :, 0])
    seg = field_segmentation(none, vexp.good, vexp.sky_sigma)
    wide = build_wide_star_mask(stars, seg.shape)
    ambient = ambient_levels(states, vexp, seg, wide)
    print('    ambient levels (nJy): ' + ', '.join(
        f'{k} {v:.2f}' for k, v in ambient.items()
    ))
    dedges, dtable = measure_profiles(
        states, vexp, stars, seg, ambient=ambient, mode='dmask',
    )

    inputs = np.array([
        (k[0], k[1], s['nstar'], s['detected_fraction'],
         s['response_rms'], s['level'])
        for k, s in sorted(cache.stats.items())
    ], dtype=[('visit', 'i8'), ('detector', 'i4'), ('nstar', 'i4'),
              ('detected_fraction', 'f4'), ('response_rms', 'f4'),
              ('level', 'f4')])
    meta = dict(
        tract=args.tract, patch=args.patch, band=args.band,
        collection=args.collection, gsub=args.gsub,
        sky_sigma=vexp.sky_sigma, templates=args.templates,
    )
    fname = os.path.join(
        args.outdir,
        f'forward-{args.tract:05d}-{args.patch:02d}-{args.band}.fits',
    )
    print('writing:', fname)
    edges_t = np.zeros(1, dtype=[('edges', 'f8', dedges.size)])
    edges_t['edges'][0] = dedges
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_image(np.nan_to_num(rcoadd).astype('f4'),
                         extname='rcoadd', compress='gzip_2')
        fits.write_image(count, extname='count', compress='gzip_2')
        fits.write_table(dtable, extname='profiles_dmask')
        fits.write_table(edges_t, extname='dmask_edges')
        fits.write_table(inputs, extname='inputs')
        fits.write_table(_meta_table(meta), extname='meta')
    ok = np.isfinite(rcoadd)
    print(f'    response coadd rms {np.nanstd(rcoadd[ok]):.3f} nJy over '
          f'{ok.mean():.3f} of the patch, {len(inputs)} inputs')


if __name__ == '__main__':
    main()
