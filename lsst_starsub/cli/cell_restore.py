"""
cli/cell_restore: restore the visit-level backgrounds on the
predetection cell coadd of a patch (coadd.polynomial_cell_coadd)
and measure the d - r_mask wing profiles around the patch's Gaia
stars in the coadd states

    object     the delivered deep coadd state (object background
               subtracted; the coadd-level model from
               deep_coadd_background applied to the stitched
               predetection image)
    none       the stitched predetection cell coadd as is
    restored   none plus the background coadd

Writes {outdir}/cell-{tract}-{patch}-{band}.fits with the state
images, the background coadd, the per-cell weight sum and input
count, the mask and variance, the census, and the profile tables
(profiles_dmask / dmask_edges as the other tools write), so that
lsst-starsub-stack runs on it directly
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
    parser.add_argument('--gaia-file', required=True)
    parser.add_argument('--gsub', type=float, default=GSUB)
    parser.add_argument(
        '--no-images', action='store_true',
        help='write only the tables',
    )
    parser.add_argument(
        '--nproc', type=int, default=4,
        help='forked workers for the input loading and the cells',
    )
    parser.add_argument(
        '--good-cells',
        help='lsst-mdet good-cells fits (tract, patch, cell_i, '
             'cell_j); pixels of cells not listed are flagged '
             'NO_DATA before the profiles are measured',
    )
    parser.add_argument(
        '--min-inputs', type=int, default=0,
        help='flag cells with fewer inputs than this NO_DATA',
    )
    return parser.parse_args()


def bad_cell_mask(mcoadd, count, good_cells_file, min_inputs, tract,
                  patch):
    """
    bool patch-frame mask of the pixels in cells to exclude: not
    in the good-cells list (when given) or with fewer than
    min_inputs inputs.  cell_i is the row (y) index and cell_j
    the column (x) index of the 150 px grid, as in
    lsst_mdet.cli.make_good_cells; the list's cell centers are
    checked against the grid
    """
    import rustfits
    import lsst.geom

    bb = mcoadd.inner_bbox
    ny, nx = bb.getHeight(), bb.getWidth()
    size = mcoadd.grid.cell_size.x
    ncy, ncx = ny // size, nx // size
    bad = np.zeros((ny, nx), dtype=bool)
    if min_inputs > 0:
        bad |= count < min_inputs

    if good_cells_file is not None:
        g = rustfits.read(good_cells_file)
        w = (g['tract'] == tract) & (g['patch'] == patch)
        rows = g[w]
        good = np.zeros((ncy, ncx), dtype=bool)
        good[rows['cell_i'], rows['cell_j']] = True
        # the convention check: the listed centers must land in
        # their cells
        wcs = mcoadd.wcs
        for r in rows[:20]:
            p = wcs.skyToPixel(lsst.geom.SpherePoint(
                float(r['ra_center']), float(r['dec_center']),
                lsst.geom.degrees,
            ))
            ci = int((p.getY() - bb.getBeginY()) // size)
            cj = int((p.getX() - bb.getBeginX()) // size)
            if (ci, cj) != (int(r['cell_i']), int(r['cell_j'])):
                raise RuntimeError(
                    f'good-cells convention mismatch: row says '
                    f'({r["cell_i"]}, {r["cell_j"]}), center maps '
                    f'to ({ci}, {cj})'
                )
        cellbad = ~good
        bad |= np.repeat(np.repeat(cellbad, size, axis=0), size, axis=1)[
            :ny, :nx
        ]
        print(
            f'    good cells: {int(good.sum())} of {ncy * ncx} listed '
            f'for {tract} {patch}'
        )
    print(f'    excluded cell fraction {bad.mean():.3f}')
    return bad


def main():
    import sys
    import rustfits
    from scipy import ndimage
    from lsst_mdet.gaia import GMAX, gaia_pixel_positions, read_gaia_file
    from lsst_mdet.patchfiles import SimpleBox
    from lsst_mdet.starsub import (
        build_star_mask, field_segmentation, select_stars,
    )
    from lsst_mdet.wcs import ButlerWcs
    from ..coadd import coadd_data_id, polynomial_cell_coadd
    from ..io import _meta_table
    from ..profiles import ambient_levels, measure_profiles
    from ..visit import (
        VisitExposure, build_wide_star_mask, convert_mask,
        make_visit_butler,
    )

    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    butler = make_visit_butler(args.repo, args.collection)
    did = coadd_data_id(args.tract, args.patch, args.band)

    print('loading the cell coadd')
    mcoadd = butler.get('deep_coadd_cell_predetection', dataId=did)
    stitched = mcoadd.stitch()
    none = np.ascontiguousarray(stitched.image.array, dtype='f4')
    var = np.ascontiguousarray(stitched.variance.array, dtype='f4')
    mask_afw = stitched.mask
    mask = convert_mask(mask_afw.array, mask_afw.getMaskPlaneDict())
    bb = stitched.bbox
    print(f'    stitched {none.shape}, bbox {bb}')

    obj = butler.get('deep_coadd_background', dataId=did).getImage().array
    if obj.shape != none.shape:
        raise RuntimeError('object background shape mismatch')

    print('building the background coadd')
    bcoadd, blevel, wsum, count = polynomial_cell_coadd(
        butler, mcoadd, repo=args.repo, collection=args.collection,
        nproc=args.nproc,
    )

    # the restored state gets the polynomials' star structure
    # back; their detector-scale parts (level and gradients)
    # were removed per input in the cache, since their per-cell
    # weighted mean is a sky patchwork stepping at the cell
    # boundaries that no sky model can follow
    states = dict(
        object=none - obj.astype('f4'),
        none=none,
        restored=none + np.nan_to_num(bcoadd),
    )

    # cells that fail the cut are flagged NO_DATA so neither the
    # census nor any annulus pixel uses them
    if args.good_cells is not None or args.min_inputs > 0:
        from lsst_mdet.defaults import DM_NO_DATA
        bad = bad_cell_mask(
            mcoadd, count, args.good_cells, args.min_inputs,
            args.tract, args.patch,
        )
        mask[:, :, 0][bad] |= DM_NO_DATA

    # the coadd as the deep_coadd-like object the profile code
    # reads; positions in the patch frame
    vexp = VisitExposure(
        image=states['none'], variance=var, mask=mask, band=args.band,
        backgrounds={}, noise=np.zeros(none.shape, dtype='f4'),
    )
    wcs = ButlerWcs(mcoadd.wcs)
    tbox = SimpleBox(bb.getBeginX(), bb.getEndX(), bb.getBeginY(),
                     bb.getEndY())
    gaia = read_gaia_file(
        args.gaia_file, wcs, tbox, gmax=max(args.gsub, GMAX),
    )
    x, y = gaia_pixel_positions(gaia, wcs, tbox)
    stars = select_stars(gaia, x, y, mask[:, :, 0], gsub=args.gsub)
    starmask, _ = build_star_mask(stars, mask[:, :, 0])
    dstar = ndimage.distance_transform_edt(~starmask)

    seg = field_segmentation(states['object'], vexp.good, vexp.sky_sigma)
    wide = build_wide_star_mask(stars, seg.shape)
    ambient = ambient_levels(states, vexp, seg, wide)
    print('    ambient levels (nJy): ' + ', '.join(
        f'{k} {v:.2f}' for k, v in ambient.items()
    ))
    print(f'    coadd sky sigma {vexp.sky_sigma:.2f} nJy')
    dedges, dtable = measure_profiles(
        states, vexp, stars, seg, ambient=ambient, mode='dmask',
    )

    meta = dict(
        tract=args.tract, patch=args.patch, band=args.band,
        collection=args.collection, gsub=args.gsub,
        sky_sigma=vexp.sky_sigma,
    )
    fname = os.path.join(
        args.outdir,
        f'cell-{args.tract:05d}-{args.patch:02d}-{args.band}.fits',
    )
    print('writing:', fname)
    dedges_t = np.zeros(1, dtype=[('edges', 'f8', dedges.size)])
    dedges_t['edges'][0] = dedges
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(dtable, extname='profiles_dmask')
        fits.write_table(dedges_t, extname='dmask_edges')
        fits.write_table(_meta_table(meta), extname='meta')
        fits.write_table(stars, extname='gaia_stars')
        if not args.no_images:
            for name, arr in states.items():
                fits.write_image(
                    np.ascontiguousarray(arr, dtype='f4'),
                    extname=name, compress='gzip_2',
                )
            fits.write_image(bcoadd, extname='bcoadd', compress='gzip_2')
            fits.write_image(blevel, extname='blevel', compress='gzip_2')
            fits.write_image(wsum, extname='wsum', compress='gzip_2')
            fits.write_image(count, extname='count', compress='gzip_2')
            fits.write_image(
                np.ascontiguousarray(obj, dtype='f4'),
                extname='object_bg', compress='gzip_2',
            )
            fits.write_image(mask[:, :, 0], extname='mask',
                             compress='gzip_2')
            fits.write_image(var, extname='var', compress='gzip_2')
            fits.write_image(starmask.astype('u1'), extname='starmask',
                             compress='gzip_2')
            fits.write_image(dstar.astype('f4'), extname='dstar',
                             compress='gzip_2')


if __name__ == '__main__':
    main()
