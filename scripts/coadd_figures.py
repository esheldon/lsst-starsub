"""
The visit route on a patch coadd, in pictures.

From a correction coadd file (lsst-starsub-correction-coadd: coadd,
correction, clean, wfrac) and the visit-route clean file of the
same patch (lsst-starsub-cell-clean --star-model visit: starmask,
gaia_stars), three figures in the patch directory:

- coadd-patch-{patch}.png: the whole patch, delivered, clean, and
  clean with the star masks set to zero, 2 x 2 binned on an asinh
  stretch.  The clean panels show the saturated cores at zero: the
  coadd holds the pipeline's interpolation there and the model
  subtracts a full core, so the true values are pits (PIT_SIGMA)
- coadd-correction-{patch}.png: 32 px box medians of the delivered
  coadd with the stars masked, the correction, and the clean coadd,
  each minus its median
- coadd-brightest-{patch}.png: the brightest star at full
  resolution, delivered against clean, asinh stretch

usage: python coadd_figures.py PATCHDIR TRACT PATCH BAND
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rustfits

BOX = 32
STRETCH = 200.0   # nJy: the asinh knee of the image panels
HALF = 500        # px: the half size of the brightest-star cutout
GCIRCLE = 12.0    # circles on the box maps for stars brighter than this
PIT_SIGMA = 10.0  # clean pixels below this many sigma are saturated cores


def box_medians(img, box):
    """box medians of an image, NaN where the box has no data"""
    ny, nx = (img.shape[0] // box) * box, (img.shape[1] // box) * box
    return np.nanmedian(img[:ny, :nx].reshape(ny // box, box, nx // box, box),
                        axis=(1, 3))


def asinh(img, knee):
    """an asinh stretch symmetric about zero"""
    return np.arcsinh(img / knee)


def main():
    pdir, tract, patch, band = sys.argv[1:5]
    tract, patch = int(tract), int(patch)
    stem = f'{tract:05d}-{patch:02d}-{band}.fits'
    cfile = os.path.join(pdir, f'correction-{stem}')
    vfile = os.path.join(pdir, f'clean-none-visit-{stem}')
    with rustfits.FITS(cfile) as f:
        coadd = f['coadd'].read().astype('f8')
        corr = f['correction'].read().astype('f8')
        clean = f['clean'].read().astype('f8')
        wfrac = f['wfrac'].read().astype('f8')
    starmask = rustfits.read(vfile, ext='starmask')
    stars = rustfits.read(vfile, ext='gaia_stars')
    wf = float(np.nanmedian(wfrac))
    title = f'tract {tract} patch {patch} {band}'

    # for display: the saturated cores, where the coadd holds the
    # pipeline's interpolation and the model subtracts a full core,
    # are pits far below anything real; shown at zero
    sig = 1.4826 * np.nanmedian(np.abs(clean - np.nanmedian(clean)))
    pits = clean < -PIT_SIGMA * sig
    shown = np.where(pits, 0.0, clean)
    print(f'{pits.sum()} pixels below -{PIT_SIGMA:.0f} sigma ({sig:.1f} nJy) '
          f'shown at zero')

    # the whole patch; the star masks (the saturated cores are pits
    # in the clean coadd: the model core over the flat saturated data)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.0))
    panels = (('delivered coadd', coadd),
              ('clean coadd (visit route), saturated cores at zero', shown),
              ('clean coadd, star masks set to zero',
               np.where(starmask > 0, 0.0, clean)))
    for ax, (lab, img) in zip(axes, panels):
        b = box_medians(img, 2)
        ax.imshow(asinh(b, STRETCH), origin='lower', cmap='gray',
                  vmin=asinh(-2 * STRETCH, STRETCH),
                  vmax=asinh(50 * STRETCH, STRETCH),
                  extent=(0, img.shape[1], 0, img.shape[0]))
        ax.set_title(f'{lab}, asinh stretch, knee {STRETCH:.0f} nJy',
                     fontsize=9)
        ax.set_xlabel('x (px)')
    axes[0].set_ylabel('y (px)')
    fig.suptitle(f'{title}: the patch, 2 x 2 binned')
    fig.tight_layout()
    out = os.path.join(pdir, f'coadd-patch-{patch}.png')
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print('wrote', out)

    # the box medians
    masked = coadd.copy()
    masked[starmask > 0] = np.nan
    panels = (
        ('delivered coadd (object background restored), stars masked',
         box_medians(masked, BOX)),
        ('the correction coadd (per-visit sky + star models)',
         box_medians(corr, BOX)),
        ('the clean coadd = delivered + correction',
         box_medians(np.where(starmask > 0, np.nan, clean), BOX)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    bright = stars[stars['G'] < GCIRCLE]
    for ax, (lab, b) in zip(axes, panels):
        b = b - np.nanmedian(b)
        ext = (0, coadd.shape[1] / 1e3, 0, coadd.shape[0] / 1e3)
        im = ax.imshow(b, origin='lower', cmap='RdBu_r', vmin=-4, vmax=4,
                       extent=ext)
        for x, y, g in zip(bright['x'], bright['y'], bright['G']):
            rad = 0.02 + 0.01 * (GCIRCLE - g)
            ax.add_patch(plt.Circle((x / 1e3, y / 1e3), rad, fill=False,
                                    color='k', lw=0.5))
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel('kpx')
    axes[0].set_ylabel('kpx')
    fig.colorbar(im, ax=axes, shrink=0.85,
                 label=f'{BOX} px box medians, each panel minus its '
                       f'median (nJy)')
    fig.suptitle(f'{title}: the visit route on the coadd (weight fraction '
                 f'corrected {wf:.2f}); circles: stars brighter than G '
                 f'{GCIRCLE:.0f}')
    out = os.path.join(pdir, f'coadd-correction-{patch}.png')
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print('wrote', out)

    # the brightest star at full resolution
    k = int(np.argmin(stars['G']))
    x, y, g = float(stars['x'][k]), float(stars['y'][k]), float(stars['G'][k])
    ix, iy = int(round(x)), int(round(y))
    x0, x1 = max(0, ix - HALF), min(coadd.shape[1], ix + HALF)
    y0, y1 = max(0, iy - HALF), min(coadd.shape[0], iy + HALF)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    for ax, (lab, img) in zip(axes, (('delivered coadd', coadd),
                                     ('clean coadd (visit route)', shown))):
        ax.imshow(asinh(img[y0:y1, x0:x1], STRETCH), origin='lower',
                  cmap='gray', vmin=asinh(-2 * STRETCH, STRETCH),
                  vmax=asinh(50 * STRETCH, STRETCH),
                  extent=(x0, x1, y0, y1))
        ax.set_title(f'{lab}, G {g:.1f} star, asinh stretch to '
                     f'{STRETCH:.0f} nJy', fontsize=9)
        ax.set_xlabel('x (px)')
    axes[0].set_ylabel('y (px)')
    fig.suptitle(f'{title}: the brightest star on the patch, full resolution')
    fig.tight_layout()
    out = os.path.join(pdir, f'coadd-brightest-{patch}.png')
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    main()
