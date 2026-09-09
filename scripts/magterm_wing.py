"""
the magnitude term of the wing: the joint-fit residual-to-model
ratio pooled over a tract sample in bins of magnitude and of
r / r_mask (the star's mask circle), written with the wing as
the 'magterm' HDU (lsst_starsub.wing.WingModel).  The base wing
is the file given; the term multiplies it per star.  Bins with
too few stars or errors above MAX_ERR get zero.

usage: python magterm_wing.py WING OUT.fits CLEANDIR TAG TRACTFILE
"""
import os
import sys
import numpy as np
import rustfits

from lsst_mdet.starsub import circle_radius  # noqa
from lsst_starsub.wing import WingModel, read_wing_model, write_wing_model  # noqa

GBINS = [(6.0, 10.0), (10.0, 11.5), (11.5, 13.0), (13.0, 14.5), (14.5, 17.0)]
XEDGES = np.array([1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0])
MIN_PIX = 50
MIN_STARS = 20
MAX_ERR = 0.03

wingfile, out, cdir, tag, tractfile = sys.argv[1:6]
model = read_wing_model(wingfile)
r, T = model.base()
tracts = [int(line.split()[0]) for line in open(tractfile)
          if line.strip() and not line.startswith('#')]
table = os.environ.get('PROFILE_TABLE', 'profiles')

nx = XEDGES.size - 1
acc = [[[] for _ in range(nx)] for _ in GBINS]
wacc = [[[] for _ in range(nx)] for _ in GBINS]
nstar = 0
for tract in tracts:
    for patch in range(100):
        f = f'{cdir}/clean-{tag}-{tract:05d}-{patch:02d}-i.fits'
        if not os.path.exists(f):
            continue
        with rustfits.FITS(f) as fits:
            prof = fits[table].read()
            edges = fits['edges'].read()['edges'][0]
            stars = fits['gaia_stars'].read()
            sig = float(fits['meta'].read()['sky_sigma'][0])
        rmid = np.sqrt(edges[:-1] * edges[1:])
        injected = prof['injected'] if 'injected' in prof.dtype.names \
            else np.zeros(prof.size, dtype=int)
        for row in prof[(prof['state'] == 'residual') & (injected == 0)]:
            st = stars[row['idx']]
            if not st['on_image'] or not st['A'] > 0:
                continue
            G = float(st['G'])
            gi = [i for i, (lo, hi) in enumerate(GBINS) if lo <= G < hi]
            if not gi:
                continue
            gi = gi[0]
            rad = circle_radius(G)
            # the ratio against the base wing (the term is measured
            # relative to it)
            mod = float(st['A']) * 10 ** (-0.4 * G) * np.interp(rmid, r, T) / sig
            ok = ((row['npix'] >= MIN_PIX) & np.isfinite(row['prof'])
                  & (rmid > rad + 2) & (mod > 0))
            for j in np.flatnonzero(ok):
                k = int(np.searchsorted(XEDGES, rmid[j] / rad) - 1)
                if 0 <= k < nx:
                    acc[gi][k].append(row['prof'][j] / mod[j])
                    wacc[gi][k].append(row['npix'][j] * mod[j] ** 2)
            nstar += 1


def robust_mean(v, w, nsig=3.0, niter=3):
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


ctab = np.zeros((len(GBINS), nx))
print(f'{nstar} stars over {len(tracts)} tracts ({tag}, {table}); '
      f'residual / model by G bin and r / r_mask, applied where measured to {MAX_ERR}')
print('  r/r_mask    ' + ''.join(f'| G {lo:.1f}-{hi:.1f}        ' for lo, hi in GBINS))
for k in range(nx):
    line = f'  {XEDGES[k]:4.2f}-{XEDGES[k+1]:4.2f} '
    for gi in range(len(GBINS)):
        if len(acc[gi][k]) >= MIN_STARS:
            m, e, n = robust_mean(acc[gi][k], wacc[gi][k])
            used = e < MAX_ERR
            if used:
                ctab[gi, k] = m
            line += f'| {m:+.3f}+-{e:.3f}{"" if used else "x"} ({n:4d}) '
        else:
            line += '|         -            '
    print(line)
gcen = np.array([0.5 * (lo + hi) for lo, hi in GBINS])
new = WingModel(r, T, gcen=gcen, xedges=XEDGES, ctab=ctab)
write_wing_model(out, new, 'i', nstar)
print('wrote', out)
