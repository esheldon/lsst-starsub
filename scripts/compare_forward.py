"""
restored - none (polynomial coadd structure) vs the forward-model
response coadds (per-visit templates, canonical wing) around the
census stars, as images (no pixel noise), mean over stars per G bin

usage: python compare_forward.py CELLDIR FORWARDDIR PATCH [PATCH ...]
"""
import sys
import numpy as np
import rustfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
from lsst_mdet.starsub import circle_radius
from lsst_starsub.profiles import dmask_edges

celldir, fdir = sys.argv[1], sys.argv[2]
patches = [int(p) for p in sys.argv[3:]]
edges = dmask_edges()
rc = 0.5 * (edges[1:] + edges[:-1])
gbins = [(6, 11), (11, 13), (13, 14), (14, 15.2), (15.2, 16)]
kinds = [('restored - none', 'b', 'g'), ('forward per-visit', 'r', 'r'),
         ('forward canonical', 'c', 'b')]
acc = {g: {k[0]: [] for k in kinds} for g in gbins}


def profile(arr, ir, ok):
    return np.array([
        np.mean(arr[(ir == k) & ok]) if ((ir == k) & ok).sum() > 20
        else np.nan for k in range(rc.size)
    ])


import os
used = []
for patch in patches:
    if not os.path.exists(f'{fdir}/forward-canonical-07275-{patch:02d}-i.fits'):
        continue
    used.append(patch)
    with rustfits.FITS(f'{celldir}/cell-07275-{patch:02d}-i.fits') as f:
        none = f['none'].read()
        restored = f['restored'].read()
        stars = f['gaia_stars'].read()
    imgs = {'restored - none': restored - none}
    for name, stem in (('forward per-visit', 'forward'),
                       ('forward canonical', 'forward-canonical')):
        try:
            with rustfits.FITS(f'{fdir}/{stem}-07275-{patch:02d}-i.fits') as f:
                imgs[name] = f['rcoadd'].read()
        except Exception:
            pass
    ny, nx = none.shape
    for st in stars:
        g = float(st['G'])
        if g >= 16 or not st['on_image']:
            continue
        rad = float(circle_radius(g))
        m = int(edges[-1] + rad) + 2
        ix, iy = int(round(st['x'])), int(round(st['y']))
        x0, x1 = max(0, ix - m), min(nx, ix + m + 1)
        y0, y1 = max(0, iy - m), min(ny, iy + m + 1)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(gy - st['y'], gx - st['x']) - rad
        ir = np.digitize(rr, edges) - 1
        ok = np.ones(rr.shape, dtype=bool)
        for name in imgs:
            ok &= np.isfinite(imgs[name][y0:y1, x0:x1])
        for glo, ghi in gbins:
            if glo <= g < ghi:
                for name in imgs:
                    acc[(glo, ghi)][name].append(
                        profile(imgs[name][y0:y1, x0:x1], ir, ok),
                    )

print('mean over stars, nJy, at d - r_mask = 55 / 205 / 455 px')
print(f'{"G bin":>10s} {"n":>3s} | ' + ' | '.join(
    f'{k[0]:>22s}' for k in kinds))
fig, axes = plt.subplots(1, len(gbins), figsize=(4.2 * len(gbins), 4))
for ax, g in zip(axes, gbins):
    parts = []
    n = 0
    for name, _, color in kinds:
        a = np.array(acc[g][name])
        if a.size == 0:
            parts.append(f'{"-":>22s}')
            continue
        n = a.shape[0]
        mn = np.nanmean(a, axis=0)
        err = np.nanstd(a, axis=0) / np.sqrt(n)
        parts.append(f'{mn[5]:6.3f} {mn[20]:6.3f} {mn[45]:6.3f}  ')
        ax.errorbar(rc, mn, yerr=err, fmt='.-', ms=3, color=color,
                    label=name)
    print(f'{g[0]:4.1f}-{g[1]:4.1f} {n:3d} | ' + ' | '.join(parts))
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'G {g[0]}-{g[1]} ({n} stars)')
    ax.set_xlabel('d - r_mask [px]')
    ax.set_ylabel('nJy')
    ax.set_ylim(-0.3, 1.0)
axes[0].legend(fontsize=7)
pstr = '-'.join(str(p) for p in used) if len(used) <= 4 else f'{len(used)}patches'
fig.suptitle(f'tract 7275 patches {pstr}: restoration vs forward models')
fig.tight_layout()
out = f'{fdir}/forward-compare-{pstr}.png'
fig.savefig(out, dpi=110)
print('wrote', out)
