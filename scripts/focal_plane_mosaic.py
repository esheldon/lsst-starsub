"""
The sky across the focal plane, the pipeline's against lsst-starsub's,
from the box-median maps the profiles files carry.

Places every detector's maps (lsst-starsub-visit --profiles-only,
visit.profiles.state_maps: BOX px box medians with the sources masked)
on the focal plane by the camera geometry, and draws three sky
treatments across by two rows down: the pipeline's per-detector
polynomial sky (the delivered image), its focal-plane sky (skyCorr
applied) and lsst-starsub's, with the stars in and with the star
model removed; each panel minus its median, averaged over --smooth
boxes on a side for display (the raw 32 px boxes carry 0.85 nJy of
noise each, which hides the troughs when a screen pixel shows one
box).  Prints the flatness table (box medians by distance to the
nearest bright star) and, with --cut, draws the cut through the
brightest star.

usage: python focal_plane_mosaic.py INDIR VISIT OUT.png
           [--cut CUT.png] [--vmax 4] [--detectors FILE]
"""
import argparse
import glob
import os

import numpy as np
import rustfits

GBRIGHT = 10.0
FAR_PX = 2400.0
PIXEL_MM = 0.01
ALL_COLUMNS = {
    'delivered': 'pipeline: per-detector polynomial sky',
    'skycorr': 'pipeline: focal-plane sky (skyCorr)',
    'starsub': 'lsst-starsub: stars + mesh sky',
}
# the skyCorr state is off by default: the DP2 deep coadd does not
# apply it, and its focal-plane model (4096 and 8192 px bins) cannot
# follow the per-detector offsets the data carry, which dominates any
# panel it is in
DEFAULT_COLUMNS = 'delivered,starsub'
# LSSTCam science rafts in detector order (detector // 9)
RAFTS = ['R01', 'R02', 'R03', 'R10', 'R11', 'R12', 'R13', 'R14', 'R20',
         'R21', 'R22', 'R23', 'R24', 'R30', 'R31', 'R32', 'R33', 'R34',
         'R41', 'R42', 'R43']
# the map of each column for the two rows
STATE_OF = {
    ('img', 'delivered'): 'delivered',
    ('img', 'skycorr'): 'skycorr',
    ('img', 'starsub'): 'flat',
    ('res', 'delivered'): 'delivered_starsub',
    ('res', 'skycorr'): 'skycorr_starsub',
    ('res', 'starsub'): 'residual',
}
ROWS = [('img', 'stars in'), ('res', 'star model removed')]
DBINS = [0, 150, 300, 600, 1200, 2400, 1e9]


def pixel_to_focal_plane(detector):
    """
    The affine pixel -> focal plane (mm) map of a detector.

    Fit from three points of the camera transform, which is rigid for
    the science detectors; applied with numpy to whole grids.

    Returns
    -------
    fn: callable
        fn(x, y) -> (xfp, yfp), arrays in mm
    """
    import lsst.afw.cameraGeom as cg
    import lsst.geom as geom

    tr = detector.getTransform(cg.PIXELS, cg.FOCAL_PLANE)
    pts = [tr.applyForward(geom.Point2D(x, y))
           for x, y in [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0)]]
    o = np.array([pts[0].x, pts[0].y])
    ax = (np.array([pts[1].x, pts[1].y]) - o) / 1000.0
    ay = (np.array([pts[2].x, pts[2].y]) - o) / 1000.0

    def fn(x, y):
        x, y = np.asarray(x, dtype='f8'), np.asarray(y, dtype='f8')
        return o[0] + ax[0] * x + ay[0] * y, o[1] + ax[1] * x + ay[1] * y
    return fn


def read_maps(fname, names):
    """the maps, the box size, the detector and its bright stars"""
    out = {}
    with rustfits.FITS(fname) as fits:
        meta = fits['meta'].read()
        stars = fits['gaia_stars'].read()
        box = None
        for name in names:
            ext = f'box_{name}'
            hdu = fits[ext]
            out[name] = hdu.read()
            box = int(hdu.header['BOX'])
    det = int(meta['detector'][0])
    bright = stars[(stars['on_image'] == 1) & (stars['G'] < GBRIGHT)]
    return det, box, out, bright


class FocalPlane:
    """
    The box maps of a visit's detectors on one focal-plane grid.

    Attributes
    ----------
    grid: dict
        name -> array (ny, nx), nan where no detector
    x0, y0, cell: float
        The grid origin and cell size in mm
    box: int
        The box size in px
    stars: list of (x, y, G)
        The bright stars in mm
    bounds: dict
        detector -> (xmin, xmax, ymin, ymax) in mm
    """


def load_focal_plane(indir, visit, names, keep=None, repo='dp2_prep_future',
                     collection='LSSTCam/runs/DRP/DP2'):
    """
    Place the box maps of every detector of a visit on the focal plane.

    Parameters
    ----------
    indir: str
        The profiles files (with the box maps)
    visit: int
    names: list of str
        The map names (without the box_ prefix)
    keep: set of int, optional
        Only these detectors
    repo, collection: str
        For the camera geometry

    Returns
    -------
    fp: FocalPlane
    """
    from lsst.daf.butler import Butler

    butler = Butler(repo, collections=collection)
    camera = butler.get('camera', instrument='LSSTCam')
    files = sorted(glob.glob(
        os.path.join(indir, f'profiles-*-{visit}-*.fits')
    ))
    per = {}
    stars = []
    bounds = {}
    box = None
    for f in files:
        det = int(f[:-5].rsplit('-', 1)[1])
        if keep is not None and det not in keep:
            continue
        det, box, maps, bright = read_maps(f, names)
        to_fp = pixel_to_focal_plane(camera[det])
        my, mx = maps[names[0]].shape
        jj, ii = np.mgrid[0:my, 0:mx]
        xfp, yfp = to_fp((ii.ravel() + 0.5) * box, (jj.ravel() + 0.5) * box)
        per[det] = (xfp, yfp, {n: m.ravel() for n, m in maps.items()},
                    (my, mx))
        half = 0.5 * box * PIXEL_MM
        bounds[det] = (xfp.min() - half, xfp.max() + half,
                       yfp.min() - half, yfp.max() + half)
        if bright.size:
            sx, sy = to_fp(bright['x'], bright['y'])
            for x, y, g in zip(sx, sy, bright['G']):
                stars.append((float(x), float(y), float(g)))
    if not per:
        raise RuntimeError('no profiles files with maps')
    cell = box * PIXEL_MM
    allx = np.concatenate([v[0] for v in per.values()])
    ally = np.concatenate([v[1] for v in per.values()])
    x0, y0 = allx.min() - cell, ally.min() - cell
    nx = int((allx.max() - x0) / cell) + 3
    ny = int((ally.max() - y0) / cell) + 3
    # each detector's map goes in as one contiguous block at the
    # rounded cell of its first box (a per-box truncated index leaves
    # unassigned columns where the detector origin falls between cells)
    grid = {n: np.full((ny, nx), np.nan) for n in names}
    for xfp, yfp, maps, (my, mx) in per.values():
        xfp, yfp = xfp.reshape(my, mx), yfp.reshape(my, mx)
        # the science detectors are axis-aligned: the map may be
        # flipped along either axis, never rotated by 90 degrees
        flipx = xfp[0, -1] < xfp[0, 0]
        flipy = yfp[-1, 0] < yfp[0, 0]
        ix0 = int(round((xfp.min() - x0) / cell))
        iy0 = int(round((yfp.min() - y0) / cell))
        for n in names:
            block = maps[n].reshape(my, mx)
            if flipx:
                block = block[:, ::-1]
            if flipy:
                block = block[::-1, :]
            grid[n][iy0:iy0 + my, ix0:ix0 + mx] = block
    fp = FocalPlane()
    fp.grid, fp.x0, fp.y0, fp.cell, fp.box = grid, x0, y0, cell, box
    fp.stars, fp.bounds = stars, bounds
    print(f'{len(per)} detectors, {box} px boxes, grid {ny} x {nx} cells '
          f'of {cell:.2f} mm; {len(stars)} stars brighter than G {GBRIGHT}: '
          + ' '.join(f'{g:.1f}' for g in sorted(t[2] for t in stars)))
    return fp


def star_distance(fp):
    """
    The distance of every grid cell to the nearest bright star, in px.
    """
    ny, nx = next(iter(fp.grid.values())).shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xc, yc = fp.x0 + (xx + 0.5) * fp.cell, fp.y0 + (yy + 0.5) * fp.cell
    dist = np.full((ny, nx), np.inf)
    for x, y, g in fp.stars:
        dist = np.minimum(dist, np.hypot(xc - x, yc - y) / PIXEL_MM)
    return dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('indir')
    p.add_argument('visit', type=int)
    p.add_argument('outfile')
    p.add_argument('--cut', default=None, help='the cut figure')
    p.add_argument('--vmax', type=float, default=4.0, help='nJy')
    p.add_argument('--smooth', type=int, default=4,
                   help='display the maps averaged over this many boxes '
                        'on a side (4: 128 px, the box noise 0.85 -> 0.2 '
                        'nJy); the numbers use the raw boxes')
    p.add_argument('--label-gmax', type=float, default=8.0,
                   help='label the stars brighter than this; every '
                        'bright star gets a circle')
    p.add_argument('--marks', choices=['none', 'circles', 'all'],
                   default='all',
                   help='mark the bright stars: none (the marks can '
                        'hide the structure when zooming), circles only, '
                        'or circles with the G labels')
    p.add_argument('--detectors', default=None,
                   help='a file with one detector per line: only these')
    p.add_argument('--raft', default=None,
                   help='one raft only, e.g. R23 (its 9 detectors); the '
                        'figure zooms to it')
    p.add_argument('--columns', default=DEFAULT_COLUMNS,
                   help='the sky treatments shown, comma separated from '
                        f'{",".join(ALL_COLUMNS)}; default {DEFAULT_COLUMNS}')
    p.add_argument('--repo', default='dp2_prep_future')
    p.add_argument('--collection', default='LSSTCam/runs/DRP/DP2')
    args = p.parse_args()

    keep = None
    if args.detectors is not None:
        keep = {int(line) for line in open(args.detectors) if line.strip()}
    if args.raft is not None:
        i0 = 9 * RAFTS.index(args.raft.upper())
        raft = set(range(i0, i0 + 9))
        keep = raft if keep is None else keep & raft
    columns = [(c, ALL_COLUMNS[c]) for c in args.columns.split(',')]
    names = sorted({STATE_OF[(r, c)] for r, _ in ROWS for c, _ in columns})
    fp = load_focal_plane(args.indir, args.visit, names, keep=keep,
                          repo=args.repo, collection=args.collection)
    grid, x0, y0, cell, box, stars = (
        fp.grid, fp.x0, fp.y0, fp.cell, fp.box, fp.stars,
    )
    ny, nx = next(iter(grid.values())).shape

    # the distance of every cell to the nearest bright star, in px
    dist = star_distance(fp)
    # each map minus its own median (the troughs are a small fraction
    # of the area); the far field is reported for reference
    far = dist > FAR_PX
    pedestal = {}
    for n in names:
        pedestal[n] = np.nanmedian(grid[n])
        grid[n] -= pedestal[n]
    print('pedestals (nJy, the map median): '
          + ', '.join(f'{n} {pedestal[n]:+.2f}' for n in names))
    print('far field relative to the median (nJy): '
          + ', '.join(f'{n} {np.nanmedian(grid[n][far]):+.2f}'
                      for n in names))
    # the flatness table for the star-subtracted states
    res = [STATE_OF[('res', c)] for c, _ in columns]
    print(f'box medians less the pedestal (nJy) by distance to the '
          f'nearest G < {GBRIGHT} star:')
    print('  distance (px)    ncell ' + ' '.join(f'{n:>18s}' for n in res))
    for lo, hi in zip(DBINS[:-1], DBINS[1:]):
        sel = (dist >= lo) & (dist < hi) & np.isfinite(grid[res[0]])
        vals = [np.nanmedian(grid[n][sel]) if sel.any() else np.nan
                for n in res]
        print(f'  {lo:5.0f}-{min(hi, 9999):5.0f} {sel.sum():8d} '
              + ' '.join(f'{v:+18.2f}' for v in vals))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=2.5, foreground='white')]
    s = max(1, args.smooth)
    extent = [x0, x0 + (nx // s) * s * cell, y0, y0 + (ny // s) * s * cell]
    ncol = len(columns)
    fig, axes = plt.subplots(2, ncol, figsize=(5.3 * ncol + 1, 10.4),
                             sharex=True, sharey=True, squeeze=False)

    def smoothed(m, s):
        """the mean over s x s blocks of boxes, nan-aware"""
        if s <= 1:
            return m
        my, mx = m.shape[0] // s, m.shape[1] // s
        b = m[:my * s, :mx * s].reshape(my, s, mx, s).transpose(0, 2, 1, 3)
        b = b.reshape(my, mx, -1)
        ok = np.isfinite(b).sum(axis=2) >= 0.5 * s * s
        out = np.nanmean(np.where(ok[:, :, None], b, np.nan), axis=2)
        out[~ok] = np.nan
        return out

    for i, (r, rlabel) in enumerate(ROWS):
        for j, (c, clabel) in enumerate(columns):
            ax = axes[i, j]
            im = ax.imshow(smoothed(grid[STATE_OF[(r, c)]], args.smooth),
                           origin='lower', vmin=-args.vmax, vmax=args.vmax,
                           cmap='RdBu_r', extent=extent,
                           interpolation='nearest')
            for x, y, g in stars:
                if args.marks == 'none':
                    break
                ax.plot(x, y, 'o', mfc='none', mec='k', ms=4)
                if g < args.label_gmax and args.marks == 'all':
                    ax.text(x + 4, y, f'G {g:.1f}', fontsize=7, color='k',
                            va='center', path_effects=halo)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(clabel, fontsize=11)
            if i == 1:
                ax.set_xlabel('focal plane x (mm)')
            if j == 0:
                ax.set_ylabel(f'{rlabel}\nfocal plane y (mm)', fontsize=11)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label('nJy')
    shown = f'{box} px boxes' if s == 1 else \
        f'{box} px boxes (median per box) averaged {s} x {s}'
    where = f'raft {args.raft.upper()}' if args.raft else 'the focal plane'
    fig.suptitle(f'visit {args.visit}, {where}: sky-subtracted images in '
                 f'{shown}, '
                 'sources masked, each panel minus its median'
                 )
    fig.savefig(args.outfile, dpi=110, bbox_inches='tight')
    print('wrote', args.outfile)

    if args.cut is None:
        return
    # the cut through the brightest star along focal-plane x: the
    # median over a band of rows per step of cells
    xs, ys, gs = min(stars, key=lambda t: t[2])
    j = int((ys - y0) / cell)
    band, step = 8, 4
    n = nx // step
    xcut = x0 + (np.arange(n) + 0.5) * step * cell
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, clabel in columns:
        rows = grid[STATE_OF[('res', c)]][j - band:j + band + 1, :n * step]
        cut = np.nanmedian(rows.reshape(rows.shape[0], n, step)
                           .transpose(1, 0, 2).reshape(n, -1), axis=1)
        ax.plot(xcut - xs, cut, 'o-', ms=3, lw=1, label=clabel)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(f'focal plane x from the G {gs:.1f} star (mm)')
    ax.set_ylabel('nJy')
    ax.set_ylim(-2 * args.vmax, 2 * args.vmax)
    ax.legend(fontsize=8)
    ax.set_title(f'visit {args.visit}: cut through the G {gs:.1f} star, '
                 f'stars removed\nmedian over a {(2 * band + 1) * cell:.1f} '
                 f'mm band per {step * cell:.2f} mm step, each state minus '
                 'its level far from the bright stars', fontsize=9)
    fig.tight_layout()
    fig.savefig(args.cut, dpi=110)
    print('wrote', args.cut)


if __name__ == '__main__':
    main()
