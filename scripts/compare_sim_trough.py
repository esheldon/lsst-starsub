"""
the trough in the simulation: stacked d - r_mask profiles of
none - sky - stars (the polynomial's imprint) and of forward -
sky - stars (what the response coadd leaves), per G bin, over the
given sim clean files.  3-sigma clipped means in 10^-3 sky sigma

usage: python compare_sim_trough.py SIMDIR TAG PATCH [PATCH ...]
reads {SIMDIR}/clean-{TAG}-07275-{patch:02d}-i.fits (tag e.g.
sim55-forward); the seed is the patch number unless SIM_SEED is
set, in which case TAG is used verbatim
"""
import os
import sys
import numpy as np
import rustfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

sys.path.insert(0, os.path.dirname(__file__))
from compare_inject import clipped_mean  # noqa

sdir = sys.argv[1]
tag = sys.argv[2]
patches = [int(p) for p in sys.argv[3:]]
gbins = [(6, 13), (13, 14), (14, 15.2), (15.2, 16), (16, 17)]
picks = [5, 10, 20, 30, 45]

rows = []
edges = None
for patch in patches:
    t = tag if os.environ.get('SIM_SEED') else tag.replace(
        'simSEED', f'sim{patch}')
    f = f'{sdir}/clean-{t}-07275-{patch:02d}-i.fits'
    if not os.path.exists(f):
        continue
    with rustfits.FITS(f) as fits:
        rows.append(fits['trough_dmask'].read())
        edges = fits['dmask_edges'].read()['edges'][0]
table = np.concatenate(rows)
dmid = 0.5 * (edges[1:] + edges[:-1])
print('3-sigma clipped mean [10^-3 sigma] at d - r_mask =',
      [int(dmid[p]) for p in picks])
fig, axes = plt.subplots(1, len(gbins), figsize=(4.2 * len(gbins), 4),
                         squeeze=False)
for col, (glo, ghi) in enumerate(gbins):
    ax = axes[0, col]
    for i, state in enumerate(('trough_none', 'trough_forward')):
        w = (table['state'] == state) & (table['G'] >= glo) & (table['G'] < ghi)
        if w.sum() == 0:
            continue
        m, e = clipped_mean(table['prof'][w] * 1e3)
        ax.errorbar(dmid, m, yerr=e, fmt='.-', ms=3, color=f'C{i}',
                    label=f'{state} ({w.sum()})')
        print(f'  G {glo:4.1f}-{ghi:4.1f} {state:15s} n={w.sum():4d}: '
              + ' '.join(f'{m[p]:6.1f}+-{e[p]:4.1f}' for p in picks))
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(-60, 20)
    ax.set_title(f'G {glo}-{ghi}')
    ax.set_xlabel('d - r_mask [px]')
    ax.set_ylabel('10^-3 sigma')
    if col == 0:
        ax.legend(fontsize=7)
pstr = '-'.join(str(p) for p in patches) if len(patches) <= 4 \
    else f'{len(patches)}patches'
fig.suptitle(f'simulation: the trough and the response coadd, {pstr}')
fig.tight_layout()
out = f'{sdir}/sim-trough-{tag}-{pstr}.png'
fig.savefig(out, dpi=110)
print('wrote', out)
