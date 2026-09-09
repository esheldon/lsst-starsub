"""
the star-subtraction residuals of the clean routes side by side:
for each route (input state none / forward / restored) the stacked
d - r_mask profiles of the input, the sky-flattened and the
star-subtracted residual states, per G bin, over the given
patches.  Profiles are in units of the coadd sky sigma; the stack
is a 3-sigma clipped mean with its error

usage: python compare_clean.py CLEANDIR PATCH [PATCH ...]
"""
import os
import sys
import numpy as np
import rustfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

cdir = sys.argv[1]
patches = [int(p) for p in sys.argv[2:]]
# the routes (input state, optionally with the star model tag);
# CLEAN_ROUTES overrides, comma separated
routes = os.environ.get(
    'CLEAN_ROUTES', 'none,restored,forward',
).split(',')
gbins = [(6, 13), (13, 14), (14, 15.2), (15.2, 16), (16, 17)]
picks = [5, 10, 20, 30, 45]


def clipped_mean(a, nsig=3.0, niter=3):
    a = np.asarray(a, dtype='f8')
    out_m = np.full(a.shape[1], np.nan)
    out_e = np.full(a.shape[1], np.nan)
    for k in range(a.shape[1]):
        v = a[:, k][np.isfinite(a[:, k])]
        if v.size < 2:
            continue
        for _ in range(niter):
            m, s = np.mean(v), np.std(v)
            keep = np.abs(v - m) <= nsig * s
            if keep.all() or keep.sum() < 2:
                break
            v = v[keep]
        out_m[k] = np.mean(v)
        out_e[k] = np.std(v) / np.sqrt(v.size)
    return out_m, out_e


tables = {r: [] for r in routes}
edges = None
for patch in patches:
    for r in routes:
        f = f'{cdir}/clean-{r}-07275-{patch:02d}-i.fits'
        if not os.path.exists(f):
            continue
        with rustfits.FITS(f) as fits:
            t = fits['profiles_dmask'].read()
            e = fits['dmask_edges'].read()['edges'][0]
        edges = e
        tables[r].append(t)
tables = {r: np.concatenate(v) for r, v in tables.items() if v}
dmid = 0.5 * (edges[1:] + edges[:-1])
print('3-sigma clipped mean of the profiles [10^-3 sigma] at d - r_mask =',
      [int(dmid[p]) for p in picks])
fig, axes = plt.subplots(2, len(gbins), figsize=(4.2 * len(gbins), 8),
                         squeeze=False)
colors = {r: f'C{i}' for i, r in enumerate(routes)}
for col, (glo, ghi) in enumerate(gbins):
    for row, state in enumerate(('flat', 'residual')):
        ax = axes[row, col]
        for r, t in tables.items():
            w = (t['state'] == state) & (t['G'] >= glo) & (t['G'] < ghi)
            if w.sum() == 0:
                continue
            m, e = clipped_mean(t['prof'][w] * 1e3)
            ax.errorbar(dmid, m, yerr=e, fmt='.-', ms=3, color=colors[r],
                        label=f'{r} route ({w.sum()})')
            print(f'  G {glo:4.1f}-{ghi:4.1f} {r:9s} {state:9s} n={w.sum():4d}: '
                  + ' '.join(f'{m[p]:6.1f}+-{e[p]:4.1f}' for p in picks))
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylim(-40, 60)
        ax.set_title(f'{state}: G {glo}-{ghi}')
        ax.set_xlabel('d - r_mask [px]')
        ax.set_ylabel('10^-3 sigma')
        if col == 0:
            ax.legend(fontsize=7)
pstr = '-'.join(str(p) for p in patches) if len(patches) <= 4 \
    else f'{len(patches)}patches'
fig.suptitle(f'tract 7275 patches {pstr}: sky-flattened and star-subtracted '
             'states by input route')
fig.tight_layout()
tag = os.environ.get('CLEAN_TAG', '')
out = f'{cdir}/clean-compare{tag}-{pstr}.png'
fig.savefig(out, dpi=110)
print('wrote', out)
