"""
cli/cell_clean: steps 5 and 6 on one patch: the star-free,
sky-flat coadd

The input state is

    forward = none + R

where R is the response coadd of lsst-starsub-cell-forward: the
coadd with the polynomials' star response added back, i.e. the
stars on a smooth sky with the trough removed.  The existing joint
sky-plus-template characterization (lsst_starsub.visit.
handle_stars_visit, the lsst_mdet machinery) then runs on it as it
would on a visit: wide-box sky pass with the stars excluded,
template from the coadd's own stamps, anchor-ring amplitudes,
refined sky.  --state none runs the same on the raw coadd for
comparison (the current product, trough in), --state restored on
the polynomial restoration.

Writes {outdir}/clean-{state}-{tract}-{patch}-{band}.fits with the
image states (input, flat, residual), the star model and sky, the
census with amplitudes, the d - r_mask profile table (read by
lsst-starsub-stack) and a summary png
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
    parser.add_argument('--cell-file',
                        help='lsst-starsub-cell-restore output with images; '
                             'without it the delivered coadd is stitched '
                             'from the butler (--state none only)')
    parser.add_argument('--forward-file',
                        help='lsst-starsub-cell-forward output (rcoadd); '
                             'needed for --state forward')
    parser.add_argument('--state', default='forward',
                        choices=['forward', 'none', 'restored'])
    parser.add_argument('--gaia-file', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--repo', default=VISIT_REPO)
    parser.add_argument('--collection', default=VISIT_COLLECTION)
    parser.add_argument('--gsub', type=float, default=GSUB)
    parser.add_argument('--nround', type=int, default=2)
    parser.add_argument(
        '--star-model', default='template',
        choices=['template', 'canonical', 'canonical-fit', 'joint'],
        help='the star model: lsst_mdet template from the coadd stamps '
             'with anchor-ring amplitudes (default); the canonical '
             'wing as a pure prediction; or its shape with the '
             'anchor-ring amplitudes',
    )
    parser.add_argument('--canonical', help='canonical wing file')
    parser.add_argument(
        '--bright-grow', type=float, default=None,
        help='exclude this many pixels beyond the bright-star masks '
             'from the 64 px sky passes (lsst_mdet coadd route: 128); '
             'default: 12 px like every other mask',
    )
    parser.add_argument(
        '--sky-from', default='stars', choices=['stars', 'residual'],
        help='what the 64 px sky passes after round 1 see: the image '
             'with the star model added back (default) or the '
             'star-subtracted image (handle_stars_visit sky_from)',
    )
    parser.add_argument('--joint-spacing', type=float, default=None,
                        help='joint model: sky mesh node spacing (px)')
    parser.add_argument('--joint-shape', default='canonical',
                        choices=['canonical', 'coadd'],
                        help='joint model: the shipped canonical wing or '
                             'the shape derived from the coadd itself')
    parser.add_argument('--joint-prior', type=float, default=None,
                        help='joint model: amplitude prior width about the '
                             'prediction (default lsst_starsub.joint.'
                             'PRIOR_SIGMA; 0 disables)')
    parser.add_argument('--joint-shallow-seg', action='store_true',
                        help='joint model: the 1.5 sigma per-pixel '
                             'segmentation instead of the deep one')
    parser.add_argument('--joint-final-pass', action='store_true',
                        help='joint model: add the 64 px pass on the '
                             'star-free image')
    parser.add_argument('--no-images', action='store_true',
                        help='write the tables and png only')
    parser.add_argument(
        '--inject', default=None, metavar='PLAN',
        help='inject synthetic stars (TODO step 7) before the clean: '
             'a plan "glo-ghi:n,..." of per-G-range counts, or '
             '"default" for lsst_starsub.inject.DEFAULT_PLAN; the '
             'cores from the coadd psf, the wings from --canonical',
    )
    parser.add_argument('--inject-seed', type=int, default=None,
                        help='rng seed; default: 1000 + patch')
    parser.add_argument('--inject-wing-scale', type=float, default=1.0,
                        help='scale the injected wings by this')
    parser.add_argument('--inject-tag', default='inj',
                        help='output-name tag of the injection run')
    return parser.parse_args()


def main():
    import rustfits
    from lsst_mdet.gaia import GMAX, read_gaia_file
    from lsst_mdet.patchfiles import SimpleBox
    from lsst_mdet.wcs import ButlerWcs
    from ..coadd import coadd_data_id, load_cell_coadd
    from ..visit import VisitExposure, make_visit_butler

    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    butler = make_visit_butler(args.repo, args.collection)
    did = coadd_data_id(args.tract, args.patch, args.band)

    print('loading the cell coadd for the wcs')
    mcoadd = load_cell_coadd(butler, did)
    bb = mcoadd.inner_bbox
    wcs = ButlerWcs(mcoadd.wcs)
    tbox = SimpleBox(bb.getBeginX(), bb.getEndX(), bb.getBeginY(),
                     bb.getEndY())
    if args.inject:
        from ..inject import psf_cube
        cube, grid_origin, cell_size = psf_cube(mcoadd)

    restored = None
    dstar = None
    if args.cell_file:
        print('reading the cell file')
        with rustfits.FITS(args.cell_file) as fits:
            none = fits['none'].read()
            var = fits['var'].read()
            mask = fits['mask'].read()
            restored = fits['restored'].read() if args.state == 'restored' \
                else None
            dstar = fits['dstar'].read() if args.inject else None
    else:
        # the delivered coadd straight from the butler, as
        # lsst-starsub-cell-restore stitches it
        from ..visit import convert_mask
        if args.state == 'restored':
            raise ValueError('--state restored needs --cell-file')
        stitched = mcoadd.stitch()
        none = np.ascontiguousarray(stitched.image.array, dtype='f4')
        var = np.ascontiguousarray(stitched.variance.array, dtype='f4')
        mask_afw = stitched.mask
        mask = convert_mask(mask_afw.array, mask_afw.getMaskPlaneDict())
        del stitched
        print(f'    stitched {none.shape}')
    del mcoadd
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    if args.inject and dstar is None:
        from scipy import ndimage
        from lsst_mdet.gaia import gaia_pixel_positions, read_gaia_file as _rg
        from lsst_mdet.starsub import build_star_mask, select_stars
        g0 = _rg(args.gaia_file, wcs, tbox, gmax=max(args.gsub, GMAX))
        x0, y0 = gaia_pixel_positions(g0, wcs, tbox)
        st0 = select_stars(g0, x0, y0, mask[:, :, 0], gsub=args.gsub)
        sm0, _ = build_star_mask(st0, mask[:, :, 0], verbose=False)
        dstar = ndimage.distance_transform_edt(~sm0)
    if args.state == 'forward':
        if args.forward_file is None:
            raise ValueError('--state forward needs --forward-file')
        with rustfits.FITS(args.forward_file) as fits:
            rcoadd = fits['rcoadd'].read()
        image = none + np.nan_to_num(rcoadd).astype('f4')
    elif args.state == 'restored':
        image = restored
    else:
        image = none.copy()
    del none, restored

    vexp = VisitExposure(
        image=image, variance=var, mask=mask, band=args.band,
        backgrounds={}, wcs=wcs, visit=args.tract, detector=args.patch,
        noise=np.zeros(image.shape, dtype='f4'),
    )
    # the profile code works in the patch frame (origin 0); the
    # gaia extract wants the tract-frame box for the sky circle
    vexp.bbox = tbox
    gaia = read_gaia_file(
        args.gaia_file, wcs, tbox, gmax=max(args.gsub, GMAX),
    )
    print(f'    state {args.state}: sky sigma {vexp.sky_sigma:.2f} nJy')

    canonical = None
    if (args.star_model not in ('template', 'joint') or args.inject
            or (args.star_model == 'joint' and args.joint_shape == 'canonical')):
        from ..template import read_canonical_wing
        canonical = read_canonical_wing(args.canonical)

    inj = None
    truth = None
    if args.inject:
        from ..inject import (
            DEFAULT_PLAN, census_rows, draw_positions, injected_table,
            parse_plan, render_injected,
        )
        plan = DEFAULT_PLAN if args.inject == 'default' \
            else parse_plan(args.inject)
        seed = 1000 + args.patch if args.inject_seed is None \
            else args.inject_seed
        rng = np.random.default_rng(seed)
        ix, iy, iG = draw_positions(rng, plan, dstar, image.shape)
        bstart = (tbox.x.start, tbox.y.start)
        truth, cores = render_injected(
            image.shape, ix, iy, iG, canonical, cube, grid_origin,
            cell_size, bstart, wing_scale=args.inject_wing_scale,
        )
        rows = census_rows(ix, iy, wcs, bstart, iG, gaia.dtype)
        gaia = np.concatenate([gaia, rows])
        inj = injected_table(ix, iy, iG, rows['ra'], rows['dec'], cores,
                             args.inject_wing_scale)
        vexp.image.array[:, :] += truth
        print(f'    injected {inj.size} stars (seed {seed}, wing scale '
              f'{args.inject_wing_scale}): G '
              + ' '.join(f'{g:.1f}' for g in np.sort(iG)))
        del dstar

    from ..clean import clean_stem, clean_tag, run_clean, write_clean_file

    out = run_clean(
        vexp, gaia, tbox, args.state, args.gsub, args.nround,
        args.bright_grow, args.star_model, canonical,
        sky_from=args.sky_from, truth=truth, inj=inj,
        joint_spacing=args.joint_spacing, joint_final=args.joint_final_pass,
        joint_deep=not args.joint_shallow_seg,
        joint_shape=args.joint_shape,
        joint_prior=None if args.joint_prior is None
        else (None if args.joint_prior <= 0 else args.joint_prior),
    )
    res = out['res']
    meta = dict(
        tract=args.tract, patch=args.patch, band=args.band,
        state=args.state, collection=args.collection, gsub=args.gsub,
        nround=args.nround, sky_sigma=out['sky_sigma'],
        fwhm=res['fwhm'] if res['fwhm'] is not None else -1.0,
        bright_grow=-1.0 if args.bright_grow is None else args.bright_grow,
        star_model=args.star_model,
        inject=args.inject or '',
        inject_wing_scale=args.inject_wing_scale,
        sky_from=args.sky_from,
    )
    tag = clean_tag(args.state, args.star_model, args.sky_from,
                    args.inject_tag if inj is not None else None,
                    joint_spacing=args.joint_spacing,
                    joint_final=args.joint_final_pass,
                    joint_deep=not args.joint_shallow_seg,
                    joint_shape=args.joint_shape)
    stem = clean_stem(args.outdir, tag, args.tract, args.patch, args.band)
    write_clean_file(stem, out, meta, no_images=args.no_images)


if __name__ == '__main__':
    main()
