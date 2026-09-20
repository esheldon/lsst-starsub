"""
The trough around bright stars: the pipeline's sky against ours.

Stacks the radial profiles (median over stars, in sky sigma, from the
profiles files of a pass-2 run) of the on-image stars by G slice for
three states of the same detector images: the pipeline's delivered
image and its warp state (the visit-level sky correction applied),
each with our star model removed, and our residual (our sky and star
model removed).  The star-subtracted pipeline states show the trough
its background left; the residual is the total package.

usage: python visit_trough_compare.py INDIR VISIT OUT.png [DETECTORS]
"""
import glob
import os
import sys

import numpy as np
import rustfits

from lsst_starsub.visit.profiles import stack_profiles

SLICES = [(6.0, 9.0), (9.0, 11.0), (11.0, 13.0), (13.0, 15.0)]
STATES = ['delivered_starsub', 'skycorr_starsub', 'residual']
# files written before 2026-09-15 name the skycorr states warp
OLD_NAMES = {'warp': 'skycorr', 'warp_starsub': 'skycorr_starsub'}


def load(indir, visit, keep=None):
    """the radial profiles of every detector's on-image stars"""
    files = sorted(glob.glob(
        os.path.join(indir, f'profiles-*-{visit}-*.fits')
    ))
    if keep is not None:
        files = [f for f in files
                 if int(f[:-5].rsplit('-', 1)[1]) in keep]
    tables = []
    edges = None
    for f in files:
        with rustfits.FITS(f) as fits:
            t = fits['profiles'].read()
            stars = fits['gaia_stars'].read()
            edges = fits['edges'].read()['edges'][0]
        t = t[stars['on_image'][t['idx']] == 1]
        for old, new in OLD_NAMES.items():
            t['state'][t['state'] == old] = new
        tables.append(t)
    return edges, np.concatenate(tables), len(files)


def main():
    indir, out = sys.argv[1], sys.argv[3]
    visit = int(sys.argv[2])
    keep = None
    if len(sys.argv) > 4:
        keep = {int(line) for line in open(sys.argv[4]) if line.strip()}
    edges, t, nfile = load(indir, visit, keep)
    have = set(np.unique(t['state']))
    states = [s for s in STATES if s in have]
    print(f'{nfile} detectors, states {states}')
    rmid = np.sqrt(edges[:-1] * edges[1:])
    show = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    print('radii (px): ' + ' '.join(f'{rmid[i]:6.0f}' for i in show))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = dict(
        delivered_starsub='pipeline sky (delivered), lsst-starsub stars '
                          'removed',
        skycorr_starsub='pipeline sky (skyCorr applied), lsst-starsub '
                        'stars removed',
        residual='lsst-starsub sky and stars removed',
    )
    fig, axes = plt.subplots(1, len(SLICES), figsize=(4.4 * len(SLICES), 4.2))
    for ax, (g0, g1) in zip(axes, SLICES):
        sel = (t['G'] >= g0) & (t['G'] < g1)
        nstar = 0
        lo = 0.0
        for state in states:
            ts = t[sel & (t['state'] == state)]
            if ts.size == 0:
                continue
            nstar = ts.size
            med, count = stack_profiles(ts, state, g0, g1, normalize=False)
            print(f'G {g0:4.1f}-{g1:4.1f} {state:18s} ({ts.size:4d} stars): '
                  + ' '.join(f'{med[i] * 1e3:+6.0f}' for i in show)
                  + '   (10^-3 sigma)')
            ax.plot(rmid, med * 1e3, 'o-', ms=3, lw=1,
                    label=labels.get(state, state))
            lo = min(lo, np.nanmin(med * 1e3))
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xscale('log')
        ax.set_xticks([20, 50, 100, 200, 500])
        ax.set_xticklabels(['20', '50', '100', '200', '500'])
        ax.set_ylim(min(-30, 1.15 * lo), 60)
        ax.set_title(f'G {g0:.0f}-{g1:.0f}, {nstar} stars', fontsize=10)
        ax.set_xlabel('r (px)')
    axes[0].set_ylabel('median profile (10^-3 sky sigma)')
    axes[0].legend(fontsize=7, loc='lower right')
    fig.suptitle(f'visit {visit}: the sky around the stars, stars removed, '
                 'the pipeline vs lsst-starsub', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print('wrote', out)


if __name__ == '__main__':
    main()
