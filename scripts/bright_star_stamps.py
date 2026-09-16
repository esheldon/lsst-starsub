"""
Stamps of the brightest stars of a visit, for the ghost-disk stack.

For every on-image star brighter than --gmax in the visit's pass-1
census (the tract run's pass1/ files), loads its detector with the
pipeline's initial background put back (the raw calibrated sky), masks
the unusable pixels, the other census stars' circles and the detected
sources, cuts a stamp of --half px around the star and bins it --bin x
--bin (mean of the usable pixels, nan where none).  Writes one file
per visit with the stamps and a table (visit, detector, G, x, y, the
focal-plane position in mm, the fwhm).

usage: python bright_star_stamps.py VISIT RUNDIR OUTFILE
           [--gmax 7.5] [--half 1400] [--bin 4]
"""
import argparse
import glob
import os

import numpy as np
import rustfits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('visit', type=int)
    p.add_argument('rundir', help='the visit run dir with pass1/')
    p.add_argument('outfile')
    p.add_argument('--gmax', type=float, default=7.5)
    p.add_argument('--half', type=int, default=1400)
    p.add_argument('--bin', type=int, default=4)
    p.add_argument('--gaia-dir',
                   default='/sdf/home/e/esheldon/oh/starsub-visits/gaia')
    p.add_argument('--repo', default='dp2_prep_future')
    p.add_argument('--collection', default='LSSTCam/runs/DRP/DP2')
    args = p.parse_args()

    from lsst.daf.butler import Butler
    import lsst.afw.cameraGeom as cg
    import lsst.geom as geom
    from lsst_starsub.census import (
        build_star_mask, field_segmentation, patch_census,
    )
    from lsst_starsub.visit.exposure import (
        load_gaia_for_exposure, load_visit_exposure, restore_background,
        measure_coadd_fwhm,
    )

    # the bright on-image stars per detector from the pass-1 census
    targets = {}
    for f in sorted(glob.glob(os.path.join(
            args.rundir, 'pass1', f'profiles-*-{args.visit}-*.fits'))):
        det = int(f[:-5].rsplit('-', 1)[1])
        stars = rustfits.read(f, ext='gaia_stars')
        sel = (stars['on_image'] == 1) & (stars['G'] < args.gmax)
        if sel.any():
            targets[det] = stars[sel]
    nstar = sum(len(v) for v in targets.values())
    print(f'visit {args.visit}: {nstar} stars brighter than G {args.gmax} '
          f'on {len(targets)} detectors')
    if nstar == 0:
        return

    butler = Butler(args.repo, collections=args.collection)
    camera = butler.get('camera', instrument='LSSTCam')
    gaia_file = os.path.join(args.gaia_dir,
                             f'gaia-dr3-visit-{args.visit}.fits')
    half, b = args.half, args.bin
    n = (2 * half) // b
    stamps = []
    rows = []
    for det, bright in targets.items():
        vexp = load_visit_exposure(butler, args.visit, det)
        gaia = load_gaia_for_exposure(vexp, gaia_file=gaia_file, gmax=19.0)
        mask0 = vexp.mask.array[:, :, 0]
        stars, starmask, comps, dstar, x, y = patch_census(
            gaia, vexp.wcs, vexp.bbox, mask0, gsub=19.0,
        )
        restore_background(vexp, which='initial')
        image = vexp.image.array.astype('f8')
        good = vexp.good
        # sources: the segmentation of the flattened image
        seg = field_segmentation(
            image - np.median(image[good]), good, vexp.sky_sigma,
        )
        fwhm = measure_coadd_fwhm(vexp)
        tr = camera[det].getTransform(cg.PIXELS, cg.FOCAL_PLANE)
        ny, nx = image.shape
        for st in bright:
            # this star's own circle stays in (its inner part is
            # saturated and masked by the pipeline anyway); the other
            # stars' circles go
            others = stars[~((np.abs(stars['x'] - st['x']) < 1)
                             & (np.abs(stars['y'] - st['y']) < 1))]
            omask, _ = build_star_mask(others, mask0, verbose=False)
            usable = good & ~omask & (seg == 0)
            ix, iy = int(round(float(st['x']))), int(round(float(st['y'])))
            stamp = np.full((2 * half, 2 * half), np.nan)
            x0, x1 = max(0, ix - half), min(nx, ix + half)
            y0, y1 = max(0, iy - half), min(ny, iy + half)
            sub = np.where(usable[y0:y1, x0:x1], image[y0:y1, x0:x1], np.nan)
            stamp[y0 - (iy - half):y1 - (iy - half),
                  x0 - (ix - half):x1 - (ix - half)] = sub
            binned = stamp.reshape(n, b, n, b).transpose(0, 2, 1, 3)
            binned = binned.reshape(n, n, -1)
            with np.errstate(all='ignore'):
                ok = np.isfinite(binned).sum(axis=2) >= 0.5 * b * b
                out = np.nanmean(np.where(ok[:, :, None], binned, np.nan),
                                 axis=2)
            out[~ok] = np.nan
            stamps.append(out.astype('f4'))
            fp = tr.applyForward(geom.Point2D(float(st['x']), float(st['y'])))
            rows.append((args.visit, det, float(st['G']), float(st['x']),
                         float(st['y']), fp.x, fp.y,
                         -1.0 if fwhm is None else float(fwhm),
                         float(vexp.sky_sigma)))
            print(f'    det {det}: G {st["G"]:.2f} at ({st["x"]:.0f}, '
                  f'{st["y"]:.0f}), fp ({fp.x:.0f}, {fp.y:.0f}) mm')
    table = np.array(rows, dtype=[
        ('visit', 'i8'), ('detector', 'i4'), ('G', 'f8'), ('x', 'f8'),
        ('y', 'f8'), ('fpx', 'f8'), ('fpy', 'f8'), ('fwhm', 'f8'),
        ('sky_sigma', 'f8'),
    ])
    hdr = {'HALF': half, 'BIN': b}
    with rustfits.FITS(args.outfile, 'w+') as fits:
        fits.write_image(np.array(stamps), extname='stamps', header=hdr)
        fits.write_table(table, extname='stars')
    print('wrote', args.outfile)


if __name__ == '__main__':
    main()
