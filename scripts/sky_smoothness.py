"""
how smooth is the real sky on 64-512 px scales?

The sky plane a run wrote (the 256 px pass plus the 64 px passes,
stars excluded) carries the sky, the star wings the star model did
not reach, and the box statistics of the noise.  Subtract the
canonical prediction of every Gaia star within FAR px of the image
(the wings beyond the template extents), fit a cubic across the
image outside the wide star masks, and measure the rms of the
residual's cell means at 64, 128, 256 and 512 px; do the same on
a noise-only sky plane made with the same passes and masks.  The
excess of the real over the noise-only rms is the sky's own
structure that a cubic misses (or wing residue).

usage: python sky_smoothness.py FILE [--visit V --detector D]
       [--gaia-file F] [--fine-grow 128]
FILE is a clean-*.fits with images (coadd) or a visit run file
"""
import argparse
import numpy as np
import rustfits

FAR = 3000
BIN = 8
SCALES = (64, 128, 256, 512)


def cubic_basis(shape, ok):
    ny, nx = shape
    gy, gx = np.mgrid[0:ny, 0:nx]
    u = (gx / nx - 0.5).ravel()
    v = (gy / ny - 0.5).ravel()
    cols = [u ** i * v ** j for i in range(4) for j in range(4 - i)]
    return np.stack(cols, 1)


def fit_cubic(img, ok):
    """cubic across the image on the ok pixels, on BIN-binned cells"""
    ny, nx = img.shape
    my, mx = ny // BIN, nx // BIN
    c = img[:my * BIN, :mx * BIN].reshape(my, BIN, mx, BIN)
    o = ok[:my * BIN, :mx * BIN].reshape(my, BIN, mx, BIN)
    n = o.sum(axis=(1, 3))
    mean = np.where(n > 0, (c * o).sum(axis=(1, 3)) / np.maximum(n, 1), 0.0)
    good = (n >= 0.5 * BIN * BIN).ravel()
    X = cubic_basis((my, mx), None)
    sol = np.linalg.lstsq(X[good], mean.ravel()[good], rcond=None)[0]
    model = (X @ sol).reshape(my, mx)
    return mean, model, n >= 0.5 * BIN * BIN


def scale_rms(resid, good):
    """rms of the cell means at each scale (cells at least half good)"""
    out = {}
    my, mx = resid.shape
    for s in SCALES:
        k = s // BIN
        cy, cx = my // k, mx // k
        r = resid[:cy * k, :cx * k].reshape(cy, k, cx, k)
        g = good[:cy * k, :cx * k].reshape(cy, k, cx, k)
        n = g.sum(axis=(1, 3))
        m = (r * g).sum(axis=(1, 3)) / np.maximum(n, 1)
        w = n >= 0.5 * k * k
        out[s] = float(np.std(m[w])) if w.sum() > 3 else np.nan
    return out


def amp_means(resid, good, shape, grid=(2, 8)):
    """per-amplifier means of the residual (detector layout)"""
    my, mx = resid.shape
    out = np.full(grid, np.nan)
    for i in range(grid[0]):
        for j in range(grid[1]):
            c = np.s_[i * my // grid[0]:(i + 1) * my // grid[0],
                      j * mx // grid[1]:(j + 1) * mx // grid[1]]
            g = good[c]
            if g.sum() > 100:
                out[i, j] = resid[c][g].mean()
    return out


def main():
    from lsst_mdet.gaia import gaia_pixel_positions, read_gaia_file
    from lsst_mdet.patchfiles import SimpleBox
    from lsst_mdet.starsub import select_stars
    from lsst_mdet.defaults import DM_NO_DATA
    from lsst_starsub.template import read_canonical_wing
    from lsst_starsub.visit import (
        VisitExposure, build_wide_star_mask, make_visit_butler,
        render_canonical_stars, sky_background,
    )
    from lsst_starsub.coadd import coadd_data_id, load_cell_coadd
    from lsst_mdet.wcs import ButlerWcs
    from scipy import ndimage

    p = argparse.ArgumentParser()
    p.add_argument('file')
    p.add_argument('--visit', type=int)
    p.add_argument('--detector', type=int)
    p.add_argument('--tract', type=int, default=7275)
    p.add_argument('--patch', type=int, default=55)
    p.add_argument('--gaia-file', required=True)
    p.add_argument('--canonical', required=True)
    p.add_argument('--repo', default=None)
    p.add_argument('--collection', default=None)
    p.add_argument('--fine-grow', type=float, default=None,
                   help='bright-star exclusion of the run (128 for the '
                        'coadd clean runs)')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--aux-file', help='file with the var and mask planes '
                   '(the cell file for a coadd clean run)')
    p.add_argument('--save', help='save the cubic residual maps (npz)')
    args = p.parse_args()

    with rustfits.FITS(args.file) as f:
        sky = f['sky'].read().astype('f8')
        starmask = f['starmask'].read().astype(bool)
        names = [h.extname for h in f]
        var = f['var'].read() if 'var' in names else None
        mask = f['mask'].read() if 'mask' in names else None
        skycorr = f['skycorr'].read().astype('f8') if 'skycorr' in names \
            else None
    if args.aux_file:
        with rustfits.FITS(args.aux_file) as f:
            var = f['var'].read()
            mask = f['mask'].read()
    shape = sky.shape
    print(f'{args.file}: sky plane {shape}, level {np.median(sky):.1f} nJy')

    # geometry and the far census
    from lsst_starsub.visit import VISIT_REPO, VISIT_COLLECTION
    repo = args.repo or VISIT_REPO
    coll = args.collection or VISIT_COLLECTION
    butler = make_visit_butler(repo, coll)
    if args.visit is not None:
        from lsst_starsub.visit import INSTRUMENT
        did = dict(instrument=INSTRUMENT, visit=args.visit,
                   detector=args.detector)
        exp = butler.get('preliminary_visit_image', dataId=did)
        wcs = ButlerWcs(exp.getWcs())
        bb = exp.getBBox()
        calib = float(exp.getPhotoCalib().getCalibrationMean())
        var = exp.variance.array * calib ** 2
        if mask is None:
            mask = exp.mask.array
        del exp
    else:
        mc = load_cell_coadd(butler, coadd_data_id(args.tract, args.patch, 'i'))
        wcs = ButlerWcs(mc.wcs)
        bb = mc.inner_bbox
        del mc
    x0, y0 = bb.getBeginX(), bb.getBeginY()
    big = SimpleBox(x0 - FAR, x0 + shape[1] + FAR, y0 - FAR,
                    y0 + shape[0] + FAR)
    gaia = read_gaia_file(args.gaia_file, wcs, big, gmax=16.0)
    x, y = gaia_pixel_positions(gaia, wcs, SimpleBox(x0, x0 + shape[1],
                                                     y0, y0 + shape[0]))
    stars = np.zeros(gaia.size, dtype=[('x', 'f8'), ('y', 'f8'), ('G', 'f8')])
    stars['x'], stars['y'], stars['G'] = x, y, gaia['phot_g_mean_mag']
    on = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    print(f'    {gaia.size} Gaia stars G < 16 within {FAR} px, {on.sum()} on '
          f'the image; brightest off-image G {stars["G"][~on].min():.1f}')
    canonical = read_canonical_wing(args.canonical)
    wings = render_canonical_stars(shape, stars, canonical, gsub=16.0)

    # the usable area: outside the wide star masks of the census
    mask0 = np.zeros(shape, dtype='i4') if mask is None else \
        (mask[:, :, 0] if mask.ndim == 3 else mask)
    census = select_stars(gaia, x, y, mask0, gsub=19.0)
    wide = build_wide_star_mask(census, shape)
    good = ~wide & ~starmask & ((mask0 & DM_NO_DATA) == 0)
    if var is not None:
        good &= np.isfinite(var) & (var > 0)
    sig = float(np.sqrt(np.median(var[good]))) if var is not None else np.nan
    print(f'    usable fraction {good.mean():.2f}, sky sigma {sig:.2f} nJy')

    results = {}
    for label, img in (('sky plane', sky), ('sky plane - wings', sky - wings)):
        mean, model, g = fit_cubic(img, good)
        resid = mean - model
        results[label] = scale_rms(resid, g)
        if label == 'sky plane - wings':
            amps = amp_means(resid, g, shape)
            saved = dict(resid=resid, good=g, sig=sig)
    if skycorr is not None:
        mean, model, g = fit_cubic(skycorr, good)
        results['pipeline skyCorr'] = scale_rms(mean - model, g)
        saved['skycorr_resid'] = mean - model

    # the noise-only reference with the same passes and masks
    if var is not None:
        rng = np.random.default_rng(args.seed)
        noise = (rng.normal(size=shape) * np.sqrt(np.where(
            np.isfinite(var) & (var > 0), var, 0.0))).astype('f4')
        m3 = np.zeros(shape + (1,), dtype='i4')
        m3[:, :, 0] = mask0
        vexp = VisitExposure(image=noise, variance=var, mask=m3, band='i',
                             backgrounds={}, noise=np.zeros(shape, 'f4'))
        dstar = ndimage.distance_transform_edt(~starmask)
        fine = dstar < 12
        if args.fine_grow is not None:
            from lsst_mdet.starsub import RESTORE_GMAX, build_star_mask
            bright = census[census['G'] < RESTORE_GMAX]
            if bright.size:
                bsm, _ = build_star_mask(bright, mask0, verbose=False)
                fine |= ndimage.distance_transform_edt(~bsm) < args.fine_grow
        nsky = sky_background(vexp, exclude=wide, bw=256)
        nsky += sky_background(vexp, exclude=fine, bw=64)
        nsky += sky_background(vexp, exclude=fine, bw=64)
        mean, model, g = fit_cubic(nsky.astype('f8'), good)
        results['noise only'] = scale_rms(mean - model, g)

    print('\n    rms of the cell means after a cubic across the image '
          '[10^-3 sigma]:')
    print('    ' + 'scale [px]'.ljust(22) + ''.join(f'{s:>10d}' for s in SCALES))
    for label, r in results.items():
        print('    ' + label.ljust(22)
              + ''.join(f'{1e3 * r[s] / sig:10.1f}' for s in SCALES))
    if args.save:
        np.savez(args.save, **saved)
    if args.visit is not None:
        print('\n    per-amplifier means of the cubic residual '
              '(sky plane - wings) [10^-3 sigma]:')
        for row in amps:
            print('    ' + ' '.join(f'{1e3 * v / sig:7.1f}' for v in row))
        print(f'    rms over amps {1e3 * np.nanstd(amps) / sig:.1f}')


if __name__ == '__main__':
    main()
