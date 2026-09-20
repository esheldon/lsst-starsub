"""
One detector before and after: the pipeline's delivered image, its sky
with our star model removed, and our residual, binned and stretched to
show the wings and the sky at the nJy level.

usage: python detector_before_after.py IMGFILE OUT.png [BIN]
"""
import sys

import numpy as np

from raft_sky_checks import detector_name, read_detector

VMAX = 30.0       # nJy, asinh stretch


def binned(image, usable, nbin):
    ny, nx = image.shape
    my, mx = ny // nbin, nx // nbin
    im = np.where(usable, image, np.nan)[:my * nbin, :mx * nbin]
    im = im.reshape(my, nbin, mx, nbin).transpose(0, 2, 1, 3)
    return np.nanmean(im.reshape(my, mx, -1), axis=2)


def main():
    fname, out = sys.argv[1], sys.argv[2]
    nbin = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    d = read_detector(fname)
    meta = d['meta'][0]
    usable = ((d['mask'] & 1) == 0) & np.isfinite(d['var']) & (d['var'] > 0)
    panels = [
        ('delivered', 'pipeline: delivered image (its sky removed)'),
        ('res_delivered', 'pipeline sky, lsst-starsub star model removed'),
        ('res_starsub', 'lsst-starsub sky and star model removed'),
    ]
    stars = d['stars']
    bright = stars[(stars['G'] < 12) & (stars['on_image'] == 1)]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (key, title) in zip(axes, panels):
        im = binned(d[key], usable, nbin)
        far = np.nanmedian(im)
        show = np.arcsinh((im - far) / VMAX * 3)
        lim = np.arcsinh(3)
        h = ax.imshow(show, origin='lower', vmin=-lim, vmax=lim,
                      cmap='RdBu_r', interpolation='nearest',
                      extent=[0, im.shape[1] * nbin / 1000,
                              0, im.shape[0] * nbin / 1000])
        for st in bright:
            ax.plot(st['x'] / 1000, st['y'] / 1000, 'o', mfc='none',
                    mec='k', ms=5)
            ax.text(st['x'] / 1000 + 0.1, st['y'] / 1000, f'G {st["G"]:.1f}',
                    fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('kpx')
    axes[0].set_ylabel('kpx')
    cb = fig.colorbar(h, ax=axes, fraction=0.02, pad=0.02)
    ticks = [-30, -10, -3, 0, 3, 10, 30]
    cb.set_ticks(np.arcsinh(np.array(ticks) / VMAX * 3))
    cb.set_ticklabels([str(t) for t in ticks])
    cb.set_label('nJy (asinh stretch)')
    fig.suptitle(f'visit {meta["visit"]} detector '
                 f'{detector_name(meta["detector"])}: {nbin} px bins, '
                 'each panel minus its median', fontsize=11)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
