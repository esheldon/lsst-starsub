"""
A focal-plane mosaic of one visit as a JPEG.

Every detector of the visit is loaded with its stored background
added back (the sky as observed), binned BIN x BIN, and placed on a
common tangent plane about the boresight through its WCS.  Two
images are written: the sky as observed with a linear stretch about
the median, and the delivered background-subtracted image with an
asinh stretch.

usage: python visit_mosaic.py VISIT OUTDIR [BAND]
"""

import sys
import numpy as np

BIN = 16
SCALE = 0.2 * BIN * 1.15  # mosaic pixel in arcsec, a little coarser
# than a binned pixel so the placement
# leaves no holes


def tangent(ra, dec, ra0, dec0):
    """Gnomonic projection to arcsec about (ra0, dec0)."""
    ra, dec, ra0, dec0 = [np.deg2rad(v) for v in (ra, dec, ra0, dec0)]
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(
        ra - ra0
    )
    x = np.cos(dec) * np.sin(ra - ra0) / cosc
    y = (
        np.cos(dec0) * np.sin(dec)
        - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    ) / cosc
    return np.rad2deg(x) * 3600, np.rad2deg(y) * 3600


def main():
    from lsst_starsub.visit.exposure import make_visit_butler
    from lsst_starsub.site import INSTRUMENT
    from lsst_starsub.visit.forward import load_raw_exposure
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    visit = int(sys.argv[1])
    outdir = sys.argv[2]
    band = sys.argv[3] if len(sys.argv) > 3 else 'z'
    butler = make_visit_butler('dp2_prep_future', 'LSSTCam/runs/DRP/DP2')
    summary = butler.get(
        'visit_summary', dataId=dict(instrument=INSTRUMENT, visit=visit)
    )
    recs = [r for r in summary if r.getWcs() is not None]
    # the boresight: the mean of the detector centers
    cens = np.array(
        [
            [
                r.getWcs()
                .pixelToSky(r.getBBox().getCenterX(), r.getBBox().getCenterY())
                .getRa()
                .asDegrees(),
                r.getWcs()
                .pixelToSky(r.getBBox().getCenterX(), r.getBBox().getCenterY())
                .getDec()
                .asDegrees(),
            ]
            for r in recs
        ]
    )
    ra0 = np.rad2deg(
        np.arctan2(
            np.mean(np.sin(np.deg2rad(cens[:, 0]))),
            np.mean(np.cos(np.deg2rad(cens[:, 0]))),
        )
    )
    dec0 = float(np.mean(cens[:, 1]))
    half = 2.0 * 3600  # arcsec: the focal plane is 3.5 deg across
    n = int(2 * half / SCALE)
    sky = np.full((n, n), np.nan, dtype='f4')
    dlv = np.full((n, n), np.nan, dtype='f4')
    for k, r in enumerate(recs):
        det = int(r['id'])
        try:
            raw, bglist, calib, meta = load_raw_exposure(butler, visit, det)
        except Exception as err:
            print(f'    detector {det}: {err!r}')
            continue
        img = raw.image.array
        ny, nx = img.shape
        my, mx = ny // BIN, nx // BIN
        b = (
            img[: my * BIN, : mx * BIN]
            .reshape(my, BIN, mx, BIN)
            .mean(axis=(1, 3))
        )
        bg = bglist.getImage().array
        d = (
            (img - bg)[: my * BIN, : mx * BIN]
            .reshape(my, BIN, mx, BIN)
            .mean(axis=(1, 3))
        )
        wcs = r.getWcs()
        bb = r.getBBox()
        gy, gx = (np.mgrid[0:my, 0:mx] + 0.5) * BIN
        ra, dec = wcs.pixelToSkyArray(
            (gx + bb.getBeginX()).ravel().astype('f8'),
            (gy + bb.getBeginY()).ravel().astype('f8'),
            degrees=True,
        )
        x, y = tangent(ra, dec, ra0, dec0)
        ix = ((x + half) / SCALE).astype(int)
        iy = ((y + half) / SCALE).astype(int)
        ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        sky[iy[ok], ix[ok]] = (b.ravel() * calib)[ok]
        dlv[iy[ok], ix[ok]] = (d.ravel() * calib)[ok]
        if k % 20 == 0:
            print(f'    {k + 1} of {len(recs)} detectors')
    med = np.nanmedian(sky)
    print(
        f'visit {visit}: sky median {med:.0f} nJy, range over the mosaic '
        f'{np.nanpercentile(sky, 1):.0f} .. {np.nanpercentile(sky, 99):.0f}'
    )
    for name, arr, lo, hi, label in (
        (
            'sky',
            sky,
            med * 0.85,
            med * 1.15,
            'sky as observed, linear +-15 percent about the median',
        ),
        (
            'delivered',
            np.arcsinh(dlv / (0.02 * med)),
            -1.0,
            4.0,
            'delivered image, asinh stretch',
        ),
    ):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
        ax.imshow(
            arr,
            vmin=lo,
            vmax=hi,
            cmap='gray',
            origin='lower',
            extent=(-half / 3600, half / 3600, -half / 3600, half / 3600),
        )
        ax.set_title(
            f'visit {visit} {band}: {label}; east left, north up '
            f'(tangent plane, degrees)',
            fontsize=11,
        )
        ax.invert_xaxis()
        out = f'{outdir}/mosaic-{visit}-{band}-{name}.jpg'
        fig.savefig(out, dpi=150, pil_kwargs={'quality': 85})
        plt.close(fig)
        print('wrote', out)


if __name__ == '__main__':
    main()
