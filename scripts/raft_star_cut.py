"""
A cut through the brightest star of a raft: the sky-subtracted,
star-subtracted images along x, the pipeline's against lsst-starsub's.

The companion of raft_mosaic.py: the same 3 x 3 raft of full-image
pass-2 outputs, binned to BOX px box medians with the sources masked,
the median over a band of rows through the star per STEP boxes along
x, each state minus its level far from the bright stars.

usage: python raft_star_cut.py IMGDIR VISIT OUT.png DET0
"""
import glob
import os
import sys

import numpy as np

from raft_sky_checks import (
    GBRIGHT, PITCH, box_medians, raft_name, read_detector,
)

BOX = 32
STEP = 4          # boxes per point along the cut (128 px)
BAND = 8          # boxes each side of the star's row (a 544 px band)
VMAX = 8.0        # nJy
STATES = [('res_delivered', 'pipeline: per-detector polynomial sky'),
          ('res_skycorr', 'pipeline: focal-plane sky (skyCorr)'),
          ('res_starsub', 'lsst-starsub: stars + mesh sky')]


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
    shape = (2 * pitch + mny, 2 * pitch + mnx)
    big = {s: np.full(shape, np.nan) for s, _ in STATES}
    stars = []
    for d in dets.values():
        r0, c0 = d['row'] * pitch, d['col'] * pitch
        for s, _ in STATES:
            med, xc, yc = box_medians(d[s], d['usable'], BOX)
            big[s][r0:r0 + mny, c0:c0 + mnx] = med
        st = d['stars']
        st = st[(st['G'] < GBRIGHT) & (st['on_image'] == 1)]
        for x, y, g in zip(st['x'], st['y'], st['G']):
            stars.append((x + d['col'] * PITCH, y + d['row'] * PITCH, g))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    xx, yy = (xx + 0.5) * BOX, (yy + 0.5) * BOX
    dist = np.full(xx.shape, np.inf)
    for x, y, g in stars:
        dist = np.minimum(dist, np.hypot(xx - x, yy - y))
    far = dist > 2400
    for s, _ in STATES:
        big[s] -= np.nanmedian(big[s][far])

    x0, y0, g0 = min(stars, key=lambda t: t[2])
    j = int(y0 // BOX)
    n = shape[1] // STEP
    xs = (np.arange(n) + 0.5) * STEP * BOX

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, label in STATES:
        rows = big[s][j - BAND:j + BAND + 1, :n * STEP]
        cut = np.nanmedian(rows.reshape(rows.shape[0], n, STEP)
                           .transpose(1, 0, 2).reshape(n, -1), axis=1)
        ax.plot((xs - x0) / 1000, cut, 'o-', ms=3, lw=1, label=label)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(f'x from the G {g0:.1f} star (kpx)')
    ax.set_ylabel('nJy')
    ax.set_ylim(-VMAX, VMAX)
    ax.legend(fontsize=8)
    ax.set_title(f'visit {visit}, raft {raft_name(det0)}: cut through the '
                 f'G {g0:.1f} star, stars removed\n'
                 f'median over a {(2 * BAND + 1) * BOX} px band per '
                 f'{STEP * BOX} px step, each state minus its level far '
                 'from the bright stars', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print('wrote', out)


if __name__ == '__main__':
    main()
