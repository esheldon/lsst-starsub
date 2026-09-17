"""
The wing cloud pooled over the visits of a band.

Every star's flux-normalized wing profile from the per-visit
template files (the wing extension), pooled and medianed per G bin
and radius, against the canonical wing and the ghost disk.  The
question is the wing beyond the disk edge and the disk level per G
bin, which one visit cannot measure.

usage: python pooled_cloud.py TEMPLATEDIR BAND OUT.png [CANONICAL.fits]
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rustfits

from lsst_starsub.joint import disk_level
from lsst_starsub.visit.template import cloud_median, disk_annuli
from lsst_starsub.wing import read_canonical_wing

GBINS = [(6, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13.5)]
# the disk level from the step across the edge: the annuli just
# inside and just outside
IN_RANGE = (500, 800)
OUT_RANGE = (900, 1400)

tdir, band, out = sys.argv[1:4]
canonical = sys.argv[4] if len(sys.argv) > 4 else None
files = sorted(glob.glob(os.path.join(tdir, f'template-*-{band}.fits')))
wings, edges = [], None
for f in files:
    with rustfits.FITS(f) as fits:
        w = fits['wing'].read()
        e = fits['edges'].read()['edges'][0]
    if edges is None:
        edges = e
    wings.append(w[['G', 'prof']])
wing = np.concatenate(wings)
rmid = 0.5 * (edges[1:] + edges[:-1])
print(f'{band}: {len(files)} visits, {wing.size} stars')
DISK_LEVEL = disk_level(band)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax, ax2 = axes
unit_disk = disk_annuli(edges, rmid)
rows = []
for glo, ghi in GBINS:
    med, err, count = cloud_median(wing, edges, glo, ghi)
    ok = np.isfinite(med)
    pos = ok & (med > 0)
    line = ax.errorbar(rmid[pos], med[pos], yerr=err[pos], fmt='.', ms=4,
                       capsize=2, label=f'G {glo:g}-{ghi:g} ({count.max()})')
    neg = ok & (med <= 0)
    ax.plot(rmid[neg], -med[neg], 'x', ms=4, color=line.lines[0].get_color())
    # the step across the edge: mean inside minus mean outside
    sin = ok & (rmid >= IN_RANGE[0]) & (rmid < IN_RANGE[1])
    sout = ok & (rmid >= OUT_RANGE[0]) & (rmid < OUT_RANGE[1])
    step = np.mean(med[sin]) - np.mean(med[sout])
    step_err = np.hypot(np.sqrt(np.mean(err[sin] ** 2) / sin.sum()),
                        np.sqrt(np.mean(err[sout] ** 2) / sout.sum()))
    outside = np.mean(med[sout])
    rows.append((glo, ghi, count.max(), step, step_err, outside,
                 np.sqrt(np.mean(err[sout] ** 2) / sout.sum())))
    # the profile beyond the disk, linear scale
    far = ok & (rmid > 300)
    ax2.errorbar(rmid[far], med[far], yerr=err[far], fmt='.-', ms=4,
                 lw=0.8, capsize=2, label=f'G {glo:g}-{ghi:g}')

if canonical:
    r, T = read_canonical_wing(canonical)
    ax.plot(r[r > 20], T[r > 20], 'k-', lw=1, label='canonical wing')
    ax2.plot(r[r > 300], T[r > 300], 'k-', lw=1, label='canonical wing')
    ax2.plot(r[r > 300], T[r > 300] + DISK_LEVEL * disk_annuli(
        edges, np.asarray(r[r > 300], dtype='f8')), 'k--', lw=1,
        label=f'canonical + disk {DISK_LEVEL:.1e}')
ax.plot(rmid, DISK_LEVEL * unit_disk, 'k:', lw=1,
        label=f'disk {DISK_LEVEL:.1e}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('r [px]')
ax.set_ylabel('nJy per unit Gaia flux')
ax.set_title(f'{band}: pooled cloud, {len(files)} visits (x: negative)')
ax.legend(fontsize=7)
ax2.axhline(0, color='k', lw=0.5)
ax2.axvline(827, color='k', lw=0.5, ls=':')
ax2.set_xlabel('r [px]')
ax2.set_ylabel('nJy per unit Gaia flux')
ax2.set_ylim(-3000, 8000)
ax2.set_title('beyond 300 px, linear')
ax2.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=110)
print('wrote', out)

print(f'step across the disk edge ({IN_RANGE} minus {OUT_RANGE} px) and '
      f'the level outside, nJy per unit flux:')
for glo, ghi, n, step, serr, outside, oerr in rows:
    print(f'  G {glo:4g}-{ghi:4g}  n {n:5d}  step {step:8.0f} +- {serr:6.0f}'
          f'  outside {outside:8.0f} +- {oerr:6.0f}')
