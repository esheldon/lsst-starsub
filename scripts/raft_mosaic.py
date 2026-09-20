"""
The sky around bright stars on one raft, the pipeline's against ours,
as images: three sky treatments across, before and after the star
subtraction down.

Mosaics the 3 x 3 raft of full-image pass-2 outputs (raft_sky_checks
conventions) binned to BOX px medians with the sources masked and
each panel's far-field pedestal removed, on one color scale in nJy.
Columns: the pipeline's delivered image (its per-detector polynomial
sky removed), the same with skyCorr applied (the focal-plane sky
model), and ours (our sky removed).  Top row: the stars in, the
brighter ones masked to their mask radius.  Bottom row: our star model
removed.

usage: python raft_mosaic.py IMGDIR VISIT OUT.png DET0
"""
import glob
import os
import sys

import numpy as np

from raft_sky_checks import (
    GBRIGHT, PITCH, box_medians, raft_name, read_detector,
)

BOX = 32
VMAX = 4.0        # nJy
COLUMNS = [('delivered', 'pipeline: per-detector polynomial sky'),
           ('skycorr', 'pipeline: focal-plane sky (skyCorr)'),
           ('starsub', 'lsst-starsub: stars + mesh sky')]
ROWS = [('img', 'usable', 'stars in'),
        ('res', 'usable', 'star model removed')]


def main():
    imgdir, out = sys.argv[1], sys.argv[3]
    visit = int(sys.argv[2])
    det0 = int(sys.argv[4])
    dets = {}
    for k in range(9):
        det = det0 + k
        files = glob.glob(os.path.join(imgdir, f'*-{visit}-{det:03d}.fits'))
        if not files:
            continue
        d = read_detector(files[0])
        d['row'], d['col'] = k // 3, k % 3
        dets[det] = d
    ny, nx = next(iter(dets.values()))['delivered'].shape
    mny, mnx = ny // BOX, nx // BOX
    pitch = PITCH // BOX
    keys = [f'{r}_{c}' for r, _, _ in ROWS for c, _ in COLUMNS]
    masks = {f'{r}_{c}': m for r, m, _ in ROWS for c, _ in COLUMNS}
    shape = (2 * pitch + mny, 2 * pitch + mnx)
    big = {k: np.full(shape, np.nan) for k in keys}
    stars = []
    for d in dets.values():
        r0, c0 = d['row'] * pitch, d['col'] * pitch
        for k in keys:
            med, xc, yc = box_medians(d[k], d[masks[k]], BOX)
            big[k][r0:r0 + mny, c0:c0 + mnx] = med
        st = d['stars']
        st = st[(st['G'] < GBRIGHT) & (st['on_image'] == 1)]
        for x, y, g in zip(st['x'], st['y'], st['G']):
            stars.append((x + d['col'] * PITCH, y + d['row'] * PITCH, g))
    # the far field of every panel: beyond 2400 px of the bright stars
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    xx, yy = (xx + 0.5) * BOX, (yy + 0.5) * BOX
    dist = np.full(xx.shape, np.inf)
    for x, y, g in stars:
        dist = np.minimum(dist, np.hypot(xx - x, yy - y))
    far = dist > 2400
    for k in keys:
        big[k] -= np.nanmedian(big[k][far])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    halo = [patheffects.withStroke(linewidth=2.5, foreground='white')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6),
                             sharex=True, sharey=True)
    extent = [0, shape[1] * BOX / 1000, 0, shape[0] * BOX / 1000]
    for i, (r, _, rlabel) in enumerate(ROWS):
        for j, (c, clabel) in enumerate(COLUMNS):
            ax = axes[i, j]
            im = ax.imshow(big[f'{r}_{c}'], origin='lower', vmin=-VMAX,
                           vmax=VMAX, cmap='RdBu_r', extent=extent,
                           interpolation='nearest')
            for x, y, g in stars:
                ax.plot(x / 1000, y / 1000, 'o', mfc='none', mec='k', ms=4)
                ax.text(x / 1000 + 0.2, y / 1000, f'G {g:.1f}', fontsize=8,
                        color='k', va='center', path_effects=halo)
            if i == 0:
                ax.set_title(clabel, fontsize=11)
            if i == 1:
                ax.set_xlabel('kpx')
            if j == 0:
                ax.set_ylabel(f'{rlabel}\nkpx', fontsize=11)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label('nJy')
    fig.suptitle(f'visit {visit}, raft {raft_name(det0)}: sky-subtracted '
                 f'images in {BOX} px boxes (median per box, sources '
                 'masked), each panel minus its level far from the bright '
                 'stars')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
