"""
A raft's images on the focal plane: the delivered data and the clean.

For each detector of a raft, the delivered visit image (the
pipeline's per-detector sky removed) and lsst-starsub's clean image
(the initial background restored, the product's sky and star model
removed: the residual state), binned by medians and placed on the
focal plane by the camera geometry as focal_plane_mosaic does, on an
asinh stretch.  Needs the pass 2 products of the detectors (the
nodes, sky boxes and amplitudes) and butler access for the images.

usage: python raft_images.py PRODUCTDIR VISIT RAFT OUT.png [--bin 8]
           [--knee 10] [--vmax 500] [--detectors FILE]
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focal_plane_mosaic import PIXEL_MM, pixel_to_focal_plane  # noqa: E402
from raft_sky_checks import raft_name  # noqa: E402


def binned(img, b):
    """b x b medians of an image, cropped to whole bins"""
    ny, nx = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    return np.nanmedian(img[:ny, :nx].reshape(ny // b, b, nx // b, b),
                        axis=(1, 3))


def detector_images(butler, product_file, visit, det, b):
    """the delivered and clean images of one detector, binned"""
    from lsst_starsub.maskbits import DM_INTRP, DM_SAT
    from lsst_starsub.visit.exposure import (
        load_visit_exposure, restore_background,
    )
    from lsst_starsub.visit.product import (
        read_product, render_sky, render_stars, wing_file,
    )
    from lsst_starsub.wing import read_wing_model

    vexp = load_visit_exposure(butler, visit, det)
    delivered = vexp.image.array.astype('f8')
    # the unusable pixels, black in the figure: the pipeline's bad,
    # saturated and interpolated ones (the bleed trails among them)
    mask0 = vexp.mask.array[:, :, 0]
    bad = ~vexp.good | ((mask0 & (DM_SAT | DM_INTRP)) != 0)
    delivered[bad] = np.nan
    product = read_product(product_file)
    wing = read_wing_model(wing_file(product))
    restore_background(vexp, which='initial')
    clean = vexp.image.array.astype('f8') - render_sky(product) \
        - render_stars(product, wing)
    clean[bad] = np.nan
    return binned(delivered, b), binned(clean, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('indir')
    p.add_argument('visit', type=int)
    p.add_argument('raft')
    p.add_argument('outfile')
    p.add_argument('--bin', type=int, default=8)
    p.add_argument('--knee', type=float, default=10.0,
                   help='nJy, the asinh knee; the 8 px medians carry ~4 '
                        'nJy of noise')
    p.add_argument('--vmax', type=float, default=500.0,
                   help='the top of the stretch in knees')
    p.add_argument('--detectors', default=None,
                   help='a file of detectors, in place of the raft')
    p.add_argument('--repo', default='dp2_prep_future')
    p.add_argument('--collection', default='LSSTCam/runs/DRP/DP2')
    args = p.parse_args()
    from lsst.daf.butler import Butler
    from lsst_starsub.visit.exposure import make_visit_butler

    butler = make_visit_butler(args.repo, args.collection)
    camera = Butler(args.repo, collections=args.collection).get(
        'camera', instrument='LSSTCam')
    files = sorted(glob.glob(
        os.path.join(args.indir, f'profiles-*-{args.visit}-*.fits')))
    keep = None
    if args.detectors:
        keep = {int(t) for t in open(args.detectors).read().split()}
    per = {}
    for f in files:
        det = int(f[:-5].rsplit('-', 1)[1])
        if keep is not None and det not in keep:
            continue
        if keep is None and raft_name(det) != args.raft:
            continue
        deliv, clean = detector_images(butler, f, args.visit, det, args.bin)
        to_fp = pixel_to_focal_plane(camera[det])
        my, mx = deliv.shape
        jj, ii = np.mgrid[0:my, 0:mx]
        xfp, yfp = to_fp((ii + 0.5) * args.bin, (jj + 0.5) * args.bin)
        per[det] = (xfp, yfp, deliv, clean)
        print(f'    detector {det}: {my} x {mx} bins')
    if not per:
        raise RuntimeError('no products for the raft')
    cell = args.bin * PIXEL_MM
    allx = np.concatenate([v[0].ravel() for v in per.values()])
    ally = np.concatenate([v[1].ravel() for v in per.values()])
    x0, y0 = allx.min() - cell, ally.min() - cell
    nx = int((allx.max() - x0) / cell) + 3
    ny = int((ally.max() - y0) / cell) + 3
    grids = [np.full((ny, nx), np.nan) for _ in range(2)]
    for xfp, yfp, deliv, clean in per.values():
        flipx = xfp[0, -1] < xfp[0, 0]
        flipy = yfp[-1, 0] < yfp[0, 0]
        ix0 = int(round((xfp.min() - x0) / cell))
        iy0 = int(round((yfp.min() - y0) / cell))
        for grid, block in zip(grids, (deliv, clean)):
            if flipx:
                block = block[:, ::-1]
            if flipy:
                block = block[::-1, :]
            grid[iy0:iy0 + block.shape[0], ix0:ix0 + block.shape[1]] = block

    extent = (x0, x0 + nx * cell, y0, y0 + ny * cell)
    knee = args.knee
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad('black')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4))
    for ax, grid, lab in zip(axes, grids, (
            'delivered: the pipeline per-detector polynomial sky removed',
            'lsst-starsub: the sky mesh and star model removed')):
        ax.imshow(np.arcsinh(grid / knee), origin='lower', cmap=cmap,
                  vmin=np.arcsinh(-3.0), vmax=np.arcsinh(args.vmax),
                  extent=extent, interpolation='nearest')
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel('focal plane x (mm)')
    axes[0].set_ylabel('focal plane y (mm)')
    fig.suptitle(f'visit {args.visit}, raft {args.raft}: {args.bin} x '
                 f'{args.bin} px medians, asinh stretch with a '
                 f'{knee:.0f} nJy knee, -3 to {args.vmax:.0f} knees; '
                 f'masked pixels black')
    fig.tight_layout()
    fig.savefig(args.outfile, dpi=110)
    print('wrote', args.outfile)


if __name__ == '__main__':
    main()
