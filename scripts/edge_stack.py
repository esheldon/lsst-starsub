"""
The gain of the visit scheme's second pass on the stars across a
detector edge: their residual profiles on the neighboring detector,
pinned to the prediction (pass 1) versus at the amplitude consolidated
from the detector that constrains them (pass 2).

Stacks the d - r_mask residual profiles (median over stars, in sky
sigma) of the off-image stars from both passes' profiles files, by G
slice, and the on-image stars for reference; prints the stacks and
writes a png.

usage: python edge_stack.py DIR_A DIR_B VISIT OUT.png [DETECTORS]

DETECTORS, a file with one detector per line, restricts the stacks.

DIR_A and DIR_B are two runs with the same census (e.g. pass 2 at the
prediction, amplitudes all 1, and pass 2 at the consolidated
amplitudes); their base names label the curves.
"""
import glob
import os
import sys

import numpy as np
import rustfits

from lsst_starsub.visit.profiles import stack_profiles

SLICES = [(6.0, 13.0), (13.0, 15.0), (15.0, 17.0)]


def load(indir, visit, keep=None):
    """the dmask residual profiles of every detector, with on_image"""
    pattern = os.path.join(indir, f'profiles-*-{visit}-*.fits')
    files = sorted(glob.glob(pattern))
    if keep is not None:
        files = [f for f in files
                 if int(f[:-5].rsplit('-', 1)[1]) in keep]
    tables = []
    edges = None
    for f in files:
        with rustfits.FITS(f) as fits:
            t = fits['profiles_dmask'].read()
            stars = fits['gaia_stars'].read()
            edges = fits['dmask_edges'].read()['edges'][0]
        t = t[t['state'] == 'residual']
        on = stars['on_image'][t['idx']]
        from numpy.lib import recfunctions as rfn
        t = rfn.append_fields(t, 'on_image', on.astype('i2'), usemask=False)
        tables.append(t)
    return edges, np.concatenate(tables), len(files)


def main():
    d1, d2, out = sys.argv[1], sys.argv[2], sys.argv[4]
    visit = int(sys.argv[3])
    keep = None
    if len(sys.argv) > 5:
        keep = {int(line) for line in open(sys.argv[5]) if line.strip()}
    edges, t1, n1 = load(d1, visit, keep)
    _, t2, n2 = load(d2, visit, keep)
    dmid = 0.5 * (edges[:-1] + edges[1:])
    print(f'pass 1: {n1} detectors, {t1.size} residual rows; pass 2: {n2}, '
          f'{t2.size}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(SLICES), figsize=(5 * len(SLICES), 4),
                            sharey=True)
    for ax, (glo, ghi) in zip(axs, SLICES):
        for label, t, style in ((os.path.basename(d1.rstrip('/')), t1, 'r-'),
                                (os.path.basename(d2.rstrip('/')), t2, 'b-')):
            off = t[t['on_image'] == 0]
            med, cnt = stack_profiles(off, 'residual', glo, ghi, min_stars=3,
                                  normalize=False)
            nstar = int(np.max(cnt)) if cnt.size else 0
            ax.plot(dmid, med * 1000, style, label=f'{label}, off-image '
                                                   f'({nstar} stars)')
            print(f'G {glo}-{ghi} {label} off-image: '
                  + ' '.join(f'{v * 1000:+.0f}' for v in med[:8]))
        on = t2[t2['on_image'] == 1]
        med, cnt = stack_profiles(on, 'residual', glo, ghi, min_stars=3,
                                  normalize=False)
        ax.plot(dmid, med * 1000, 'k--',
                label=f'{os.path.basename(d2.rstrip("/"))}, on-image '
                      f'({int(np.max(cnt))} stars)')
        print(f'G {glo}-{ghi} {os.path.basename(d2.rstrip("/"))} on-image: '
              + ' '.join(f'{v * 1000:+.0f}' for v in med[:8]))
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(f'G {glo:.0f}-{ghi:.0f}')
        ax.set_xlabel('d - r_mask (px)')
        ax.set_xlim(0, 300)
        ax.legend(fontsize=8)
    axs[0].set_ylabel('median residual (10^-3 sky sigma)')
    fig.suptitle(f'visit {visit}: residuals of the edge stars')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print('wrote', out)


if __name__ == '__main__':
    main()
