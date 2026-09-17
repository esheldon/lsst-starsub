"""
The canonical wing of a band from the pooled wing cloud.

The per-visit fits describe the far wing with two power laws and,
until the ghost ring was known, absorbed the ring into the aureole:
the median of those fits (canonical_from_templates.py) is 15
percent low at 100-160 px, 17 percent high at 460 px and two to
four times too bright beyond the ring.  The cloud pooled over the
band's visits (every star's flux-normalized profile from the
template files) measures the wing directly: one power law from 80
px out through and beyond the ring, with the ring on top.

The new wing keeps the old canonical inside R_LO px (the stack
median over visits, which the cloud confirms to 3 percent), takes
the pooled cloud minus the ring at the band's level
(lsst_starsub.joint.disk_level) from R_HI out to R_FAR px or the
last annulus still measured at MIN_SNR, and the fitted power law
beyond, joined by continuity.  Prints the ring level and wing slope
fitted to the pooled profile, to check DISK_LEVELS for the band.

usage: python canonical_empirical.py TEMPLATEDIR BAND OLD.fits NEW.fits OUT.png
"""
import glob
import os
import sys

import numpy as np
import rustfits

from lsst_starsub.joint import (
    DISK_EDGE, DISK_INNER, DISK_RADIUS, disk_level, disk_profile,
)
from lsst_starsub.visit.template import cloud_median
from lsst_starsub.wing import read_canonical_wing, write_canonical_wing

GBINS = [(6, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13.5),
         (13.5, 15.5)]
PEDESTAL_RMIN = 1400.0   # each bin's far level comes off as its sky bias
R_LO, R_HI = 50.0, 65.0  # the old canonical inside, blended to the cloud
R_FAR = 1600.0           # the cloud to here, the fitted power law beyond
MIN_SNR = 3.0            # ... or to the last annulus this well measured
FIT_RANGE = (70.0, 1700.0)


def annulus_mean(fn, edges, rmid):
    """area-weighted mean of fn(r) over the annulus holding each rmid"""
    out = np.zeros(rmid.size)
    for i, rc in enumerate(rmid):
        k = np.clip(np.searchsorted(edges, rc) - 1, 0, edges.size - 2)
        g = np.linspace(edges[k], edges[k + 1], 400)
        out[i] = np.sum(g * fn(g)) / np.sum(g)
    return out


def pooled_profile(files):
    """the per-annulus inverse-variance combination of the G bins'
    medians, each bin's far pedestal removed and its first (partly
    masked) annulus dropped"""
    wings, edges = [], None
    for f in files:
        with rustfits.FITS(f) as fits:
            w = fits['wing'].read()
            e = fits['edges'].read()['edges'][0]
        edges = e if edges is None else edges
        wings.append(w[['G', 'prof']])
    wing = np.concatenate(wings)
    rmid = 0.5 * (edges[1:] + edges[:-1])
    num, den = np.zeros(rmid.size), np.zeros(rmid.size)
    for glo, ghi in GBINS:
        med, err, _ = cloud_median(wing, edges, glo, ghi)
        ok = np.isfinite(med) & (err > 0)
        if not ok.any():
            continue
        far = ok & (rmid > PEDESTAL_RMIN)
        ped = np.mean(med[far]) if far.any() else 0.0
        ok[np.flatnonzero(ok)[0]] = False
        num[ok] += (med[ok] - ped) / err[ok] ** 2
        den[ok] += 1.0 / err[ok] ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return edges, rmid, num / den, 1.0 / np.sqrt(den), wing.size


def fit_law_and_ring(edges, rmid, prof, err):
    """the power law and the ring level over FIT_RANGE"""
    use = ((rmid > FIT_RANGE[0]) & (rmid < FIT_RANGE[1]) & np.isfinite(prof)
           & (err > 0))
    r, y, e = rmid[use], prof[use], err[use]
    ring = annulus_mean(disk_profile, edges, r)
    best = None
    for s in np.arange(-3.5, -2.0, 0.01):
        A = np.vstack([r ** s, ring]).T / e[:, None]
        c, *_ = np.linalg.lstsq(A, y / e, rcond=None)
        chi2 = float(np.sum((A @ c - y / e) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, s, c[0], c[1])
    return dict(chi2=best[0], npt=int(r.size), slope=best[1], amp=best[2],
                level=best[3])


def main():
    tdir, band, old, new, png = sys.argv[1:6]
    files = sorted(glob.glob(os.path.join(tdir, f'template-*-{band}.fits')))
    edges, rmid, prof, err, nstar = pooled_profile(files)
    print(f'{band}: {len(files)} visits, {nstar} stars in the cloud')
    DISK_LEVEL = disk_level(band)

    fit = fit_law_and_ring(edges, rmid, prof, err)
    print(f'  power law r^{fit["slope"]:.2f} amp {fit["amp"]:.3e} plus the '
          f'ring at {fit["level"]:.0f} nJy per unit flux (DISK_LEVELS '
          f'{DISK_LEVEL:.0f}; radii {DISK_INNER:.0f}-{DISK_RADIUS:.0f} +- '
          f'{DISK_EDGE:.0f}); chi2 {fit["chi2"]:.1f} for {fit["npt"]} '
          f'points {FIT_RANGE[0]:.0f}-{FIT_RANGE[1]:.0f} px')

    # the cloud minus the ring, corrected from annulus means to
    # midpoint values by the fitted law's ratio
    def law(r):
        return fit['amp'] * np.asarray(r, dtype='f8') ** fit['slope']

    ring = annulus_mean(disk_profile, edges, rmid)
    corr = law(rmid) / annulus_mean(law, edges, rmid)
    emp = (prof - DISK_LEVEL * ring) * corr
    emp_err = err * corr
    ok = (np.isfinite(emp) & (rmid >= R_LO) & (rmid <= R_FAR) & (emp > 0)
          & (emp > MIN_SNR * emp_err))
    # the cloud out to the last well measured annulus, contiguous
    last = np.flatnonzero(ok)[0]
    while last + 1 < rmid.size and ok[last + 1]:
        last += 1
    ok[last + 1:] = False
    r_far = rmid[last]

    r, T_old = read_canonical_wing(old)
    T_cloud = np.exp(np.interp(np.log(r), np.log(rmid[ok]), np.log(emp[ok])))
    # continuity into the law beyond the cloud
    scale = emp[last] / law(r_far)
    T_law = scale * law(r)
    T = np.where(r <= r_far, T_cloud, T_law)
    frac = np.clip((r - R_LO) / (R_HI - R_LO), 0.0, 1.0)
    T = (1.0 - frac) * T_old + frac * T
    write_canonical_wing(new, r, T, band, len(files))
    print(f'  wrote {new}; the cloud to {r_far:.0f} px, the law beyond '
          f'scaled by {scale:.3f}')
    for rr in (50, 100, 160, 300, 460, 660, 943, 1500, 2500):
        j = np.argmin(np.abs(r - rr))
        print(f'  {rr:5d} px: old {T_old[j]:10.1f} new {T[j]:10.1f} '
              f'ratio {T[j] / T_old[j]:.3f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    pos = np.isfinite(prof) & (prof > 0)
    ax.errorbar(rmid[pos], prof[pos], yerr=err[pos], fmt='k.', ms=4,
                capsize=2, label='pooled cloud')
    ax.errorbar(rmid[ok], emp[ok], yerr=emp_err[ok], fmt='b.', ms=4,
                capsize=2, label='cloud minus ring')
    ax.plot(r[r > 10], T_old[r > 10], 'r-', lw=1, label='old canonical')
    ax.plot(r[r > 10], T[r > 10], 'g-', lw=1, label='new canonical')
    ax.plot(r[r > 10], T[r > 10] + DISK_LEVEL * disk_profile(r[r > 10]),
            'g--', lw=0.8, label='new + ring')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r [px]')
    ax.set_ylabel('nJy per unit Gaia flux')
    ax.set_title(f'{band}: {len(files)} visits, {nstar} stars')
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.errorbar(rmid[ok], emp[ok] / np.interp(rmid[ok], r, T_old),
                yerr=emp_err[ok] / np.interp(rmid[ok], r, T_old), fmt='b.',
                ms=4, capsize=2, label='(cloud minus ring) / old')
    ax.plot(r[r > 10], T[r > 10] / T_old[r > 10], 'g-', lw=1,
            label='new / old')
    ax.axhline(1, color='k', lw=0.5)
    ax.set_xscale('log')
    ax.set_ylim(0, 2)
    ax.set_xlabel('r [px]')
    ax.set_ylabel('ratio to the old canonical')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print('  wrote', png)


if __name__ == '__main__':
    main()
