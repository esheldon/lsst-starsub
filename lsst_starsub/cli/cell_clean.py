"""
cli/cell_clean: the star-free, sky-flat coadd of one patch

The delivered cell coadd is stitched from the butler (the object
background restored; the per-visit polynomials' trough is in, the
joint fit's sky mesh absorbs it), the Gaia census and star masks
are built as in production (lsst_starsub.census), and the star
model is one of

    joint      the canonical wing shape with the per-star
               amplitudes and a bilinear sky mesh solved
               together (lsst_starsub.joint): the production
               model
    template   the stamp route's sequential scheme: sky passes with
               the stars excluded, the template from the coadd's
               stamps, anchor-ring amplitudes (the comparison
               baseline; --bright-grow, --nround apply)
    canonical  the canonical wing as a pure prediction, no fit

With --inject, synthetic stars with the canonical wing as truth
are added first (lsst_starsub.inject) and the tables carry the
truth decomposition: 'perfect' (the residual under a perfect star
model) and 'model_error' states, and an 'injected' flag.

Writes {outdir}/clean-{tag}-{tract}-{patch}-{band}.fits with the
image states (unless --no-images), the census with amplitudes,
the radial and d - r_mask profile tables (scripts/compare_inject.py,
lsst-starsub-stack) and a summary png
"""
import os
import sys

import numpy as np

# the input state's name in the tables: the delivered coadd
STATE = 'none'


def get_args():
    """
    Parse the command line.
    """
    import argparse
    from . import add_butler_arguments
    from ..census import GSUB

    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--gaia-file', required=True)
    parser.add_argument('--canonical', required=True,
                        help='the wing file (lsst_starsub.wing)')
    parser.add_argument('--outdir', required=True)
    add_butler_arguments(parser)
    parser.add_argument('--gsub', type=float, default=GSUB)
    parser.add_argument('--star-model', default='joint',
                        choices=['joint', 'template', 'canonical'])
    parser.add_argument('--joint-spacing', type=float, default=None,
                        help='joint model: sky mesh node spacing (px)')
    parser.add_argument('--joint-prior', type=float, default=None,
                        help='joint model: amplitude prior width about the '
                             'prediction (default lsst_starsub.joint.'
                             'PRIOR_SIGMA; 0 disables)')
    parser.add_argument('--nround', type=int, default=2,
                        help='template model: sky-and-amplitude rounds')
    parser.add_argument(
        '--bright-grow', type=float, default=None,
        help='template model: exclude this many pixels beyond the '
             'bright-star masks from the 64 px sky passes (stamp '
             'coadd route: 128); default 12 px like every other mask',
    )
    parser.add_argument('--no-images', action='store_true',
                        help='write the tables and png only')
    parser.add_argument(
        '--inject', default=None, metavar='PLAN',
        help='inject synthetic stars before the clean: a plan '
             '"glo-ghi:n,..." of per-G-range counts, or "default" for '
             'lsst_starsub.inject.DEFAULT_PLAN; the cores from the '
             'coadd psf, the wings from --canonical',
    )
    parser.add_argument('--inject-seed', type=int, default=None,
                        help='rng seed; default: 1000 + patch')
    parser.add_argument('--inject-wing-scale', type=float, default=1.0,
                        help='scale the injected wings by this')
    parser.add_argument('--inject-tag', default='inj',
                        help='output-name tag of the injection run')
    return parser.parse_args()


def main():
    """
    Clean one patch coadd.
    """
    from ..gaia import GMAX, gaia_pixel_positions, read_gaia_file
    from ..geom import ButlerWcs, SimpleBox
    from ..coadd.clean import (
        clean_stem,
        clean_tag,
        run_clean,
        write_clean_file,
    )
    from ..coadd.cellcoadd import coadd_data_id, load_cell_coadd
    from ..visit.exposure import VisitExposure, convert_mask, make_visit_butler
    from ..wing import read_wing_model

    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    butler = make_visit_butler(args.repo, args.collection)
    did = coadd_data_id(args.tract, args.patch, args.band)

    print('loading the cell coadd')
    mcoadd = load_cell_coadd(butler, did)
    bb = mcoadd.inner_bbox
    wcs = ButlerWcs(mcoadd.wcs)
    tbox = SimpleBox(bb.getBeginX(), bb.getEndX(), bb.getBeginY(),
                     bb.getEndY())
    if args.inject:
        from ..inject import psf_cube
        cube, grid_origin, cell_size = psf_cube(mcoadd)
    stitched = mcoadd.stitch()
    image = np.ascontiguousarray(stitched.image.array, dtype='f4')
    var = np.ascontiguousarray(stitched.variance.array, dtype='f4')
    mask_afw = stitched.mask
    mask = convert_mask(mask_afw.array, mask_afw.getMaskPlaneDict())
    del stitched, mcoadd
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    print(f'    stitched {image.shape}')

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
    print(f'    sky sigma {vexp.sky_sigma:.2f} nJy')
    canonical = read_wing_model(args.canonical)

    inj = None
    truth = None
    if args.inject:
        from scipy import ndimage
        from ..census import build_star_mask, select_stars
        from ..inject import (
            DEFAULT_PLAN,
            census_rows,
            draw_positions,
            injected_table,
            parse_plan,
            render_injected,
        )
        # the existing star masks, to keep the injected centers clear
        x0, y0 = gaia_pixel_positions(gaia, wcs, tbox)
        st0 = select_stars(gaia, x0, y0, mask[:, :, 0], gsub=args.gsub)
        sm0, _ = build_star_mask(st0, mask[:, :, 0], verbose=False)
        dstar = ndimage.distance_transform_edt(~sm0)
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
        del dstar, sm0

    prior = None if args.joint_prior is None else (
        None if args.joint_prior <= 0 else args.joint_prior)
    out = run_clean(
        vexp, gaia, tbox, STATE, args.gsub, args.nround, args.bright_grow,
        args.star_model, canonical, truth=truth, inj=inj,
        joint_spacing=args.joint_spacing, joint_prior=prior,
    )
    res = out['res']
    meta = dict(
        tract=args.tract, patch=args.patch, band=args.band,
        state=STATE, collection=args.collection, gsub=args.gsub,
        nround=args.nround, sky_sigma=out['sky_sigma'],
        fwhm=res['fwhm'] if res['fwhm'] is not None else -1.0,
        bright_grow=-1.0 if args.bright_grow is None else args.bright_grow,
        star_model=args.star_model,
        inject=args.inject or '',
        inject_wing_scale=args.inject_wing_scale,
        wing=os.path.basename(args.canonical),
    )
    tag = clean_tag(STATE, args.star_model,
                    args.inject_tag if inj is not None else None,
                    joint_spacing=args.joint_spacing)
    stem = clean_stem(args.outdir, tag, args.tract, args.patch, args.band)
    write_clean_file(stem, out, meta, no_images=args.no_images)


if __name__ == '__main__':
    main()
