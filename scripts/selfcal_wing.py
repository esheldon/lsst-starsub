"""
self-calibration of the wing shape from the joint-fit residuals:
over the clean files of a tract, the residual-to-model ratio of
the bright (G < GMAX) on-image stars in the radial bins of the
'profiles' table (12-900 px), inverse-variance stacked, gives a
multiplicative correction c(r); the wing becomes T (1 + c),
interpolated in log r, held beyond the last bin and tapered to
zero below the first usable one (the stamps region is not
corrected).  Injected stars are excluded (their truth is the
uncorrected wing).

usage: python selfcal_wing.py CANONICAL OUT.fits CLEANDIR TAG TRACT PATCH...
   or  python selfcal_wing.py CANONICAL OUT.fits CLEANDIR TAG TRACTFILE
       (a file with one tract per line, first column; all patches
       0-99 of each: the broad calibration); the correction is
       also printed per G bin (GBINS) to show any dependence on
       brightness, the applied one uses every star
"""
import os
import sys
import numpy as np
import rustfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

from lsst_mdet.starsub import circle_radius  # noqa
from lsst_starsub.template import read_canonical_wing, write_canonical_wing  # noqa

GMAX = 13.0
MIN_PIX = 50
MIN_STARS = 5
MAX_ERR = 0.03   # bins measured worse than this are not corrected

GBINS = [(6.0, 10.0), (10.0, 11.5), (11.5, 13.0)]

canonical, out = sys.argv[1], sys.argv[2]
cdir, tag = sys.argv[3], sys.argv[4]
if os.path.isfile(sys.argv[5]):
    tracts = [int(line.split()[0]) for line in open(sys.argv[5])
              if line.strip() and not line.startswith('#')]
    jobs = [(t, p) for t in tracts for p in range(100)]
else:
    tract = int(sys.argv[5])
    jobs = [(tract, int(p)) for p in sys.argv[6:]]
    tracts = [tract]
patches = jobs
r, T = read_canonical_wing(canonical)

ratios = None
gmags = None
nstar = 0
for tract, patch in jobs:
    f = f'{cdir}/clean-{tag}-{tract:05d}-{patch:02d}-i.fits'
    if not os.path.exists(f):
        continue
    with rustfits.FITS(f) as fits:
        names = [h.extname for h in fits]
        prof = fits['profiles'].read()
        edges = fits['edges'].read()['edges'][0]
        stars = fits['gaia_stars'].read()
        sig = float(fits['meta'].read()['sky_sigma'][0])
    rmid = np.sqrt(edges[:-1] * edges[1:])
    if ratios is None:
        ratios = [[] for _ in rmid]
        weights = [[] for _ in rmid]
        gmags = [[] for _ in rmid]
    injected = prof['injected'] if 'injected' in prof.dtype.names else \
        np.zeros(prof.size, dtype=int)
    w = (prof['state'] == 'residual') & (prof['G'] < GMAX) & (injected == 0)
    for row in prof[w]:
        st = stars[row['idx']]
        if not st['on_image'] or not st['A'] > 0:
            continue
        G = float(st['G'])
        model = float(st['A']) * 10.0 ** (-0.4 * G) * np.interp(rmid, r, T) / sig
        ok = ((row['npix'] >= MIN_PIX) & np.isfinite(row['prof'])
              & (rmid > circle_radius(G) + 2) & (model > 0))
        if not ok.any():
            continue
        ratio = row['prof'][ok] / model[ok]
        # the ratio's inverse variance: pixel noise sigma over
        # sqrt(npix) on the profile, divided by the model
        wt = row['npix'][ok] * model[ok] ** 2
        for j, rt, wj in zip(np.flatnonzero(ok), ratio, wt):
            ratios[j].append(float(rt))
            weights[j].append(float(wj))
            gmags[j].append(G)
        nstar += 1


def robust_mean(v, w, nsig=3.0, niter=3):
    """
    clipped weighted mean with the weights capped at their 80th
    percentile (no single star dominates a bin); the error from
    the weighted star-to-star scatter
    """
    v, w = np.asarray(v, 'f8'), np.asarray(w, 'f8')
    w = np.minimum(w, np.percentile(w, 80))
    keep = np.ones(v.size, bool)
    for _ in range(niter):
        m = np.sum(w[keep] * v[keep]) / np.sum(w[keep])
        sd = np.sqrt(np.sum(w[keep] * (v[keep] - m) ** 2) / np.sum(w[keep]))
        new = np.abs(v - m) <= nsig * max(sd, 1e-6)
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    m = np.sum(w[keep] * v[keep]) / np.sum(w[keep])
    sd = np.sqrt(np.sum(w[keep] * (v[keep] - m) ** 2) / np.sum(w[keep]))
    neff = np.sum(w[keep]) ** 2 / np.sum(w[keep] ** 2)
    return m, sd / np.sqrt(max(neff, 1.0)), int(keep.sum())


c = np.zeros(rmid.size)
cerr = np.full(rmid.size, np.nan)
cnt = np.zeros(rmid.size, dtype=int)
den = np.zeros(rmid.size)
for j in range(rmid.size):
    if len(ratios[j]) >= MIN_STARS:
        c[j], cerr[j], cnt[j] = robust_mean(ratios[j], weights[j])
        den[j] = 1.0 / cerr[j] ** 2 if cerr[j] > 0 else 0.0
usable = (cnt >= MIN_STARS) & (cerr < MAX_ERR)
tstr = ','.join(str(t) for t in tracts) if len(tracts) <= 4 \
    else f'{len(tracts)} tracts'
print(f'{nstar} stars G < {GMAX} over {len(patches)} patches ({tstr}, {tag})')
print('  r [px]   stars   correction c(r) = residual / model'
      + ''.join(f'   | G {lo:.1f}-{hi:.1f}' for lo, hi in GBINS))
for j in range(rmid.size):
    flag = '' if usable[j] else '   (unused)'
    per = ''
    gj = np.array(gmags[j])
    for lo, hi in GBINS:
        w = (gj >= lo) & (gj < hi)
        if w.sum() >= MIN_STARS:
            m, e, n = robust_mean(np.array(ratios[j])[w], np.array(weights[j])[w])
            per += f'   | {m:+.3f}+-{e:.3f} ({n:3d})'
        else:
            per += '   |          -        '
    print(f'  {rmid[j]:6.0f}  {cnt[j]:5d}   {c[j]:+7.3f} +- {cerr[j]:.3f}{flag}{per}')

# the smoothed correction: weighted 3-bin running mean over the
# usable bins, in log r; zero below the first usable bin (tapered
# over one bin), held beyond the last
cs = np.zeros(rmid.size)
for j in np.flatnonzero(usable):
    sel = [k for k in (j - 1, j, j + 1) if 0 <= k < rmid.size and usable[k]]
    ww = np.array([den[k] for k in sel])
    cs[j] = float(np.sum(ww * c[sel]) / ww.sum())
first, last = np.flatnonzero(usable)[[0, -1]]
lr = np.log(np.maximum(r, 1.0))
lrm = np.log(rmid)
corr = np.interp(lr, lrm[first:last + 1], cs[first:last + 1],
                 left=0.0, right=cs[last])
# taper in from zero over the bin below the first usable one
if first > 0:
    lo, hi = lrm[first - 1], lrm[first]
    ramp = np.clip((lr - lo) / (hi - lo), 0.0, 1.0)
    corr = np.where(lr < hi, ramp * cs[first], corr)
Tnew = T * (1.0 + corr)
write_canonical_wing(out, r, Tnew, 'i', nstar)
print('wrote', out)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(rmid[usable], c[usable], yerr=cerr[usable], fmt='o', ms=4,
            label='stacked residual / model')
ax.plot(r, corr, 'r-', lw=1, label='applied correction')
ax.axhline(0, color='k', lw=0.5)
ax.set_xscale('log')
ax.set_xlabel('r [px]')
ax.set_ylabel('c(r)')
ax.set_ylim(-0.5, 0.5)
ax.set_title(f'wing self-calibration, {tstr}: {nstar} stars G < {GMAX}')
ax.legend()
fig.tight_layout()
fig.savefig(out.replace('.fits', '.png'), dpi=110)
