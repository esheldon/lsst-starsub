"""
The star-subtraction routes side by side on one patch coadd.

Stacks the d - r_mask residual profiles (median over stars, in units
of the coadd sky sigma) of the on-image census stars by G slice from
the clean files of several routes (lsst-starsub-cell-clean: the
coadd-level joint fit, the visit route's correction coadd, ...), with
the delivered coadd's own profile ('none') for reference, and prints
the numbers.

usage: python compare_routes.py OUT.png LABEL=CLEAN.fits [LABEL=CLEAN.fits ...]
"""
import sys

import rustfits

from lsst_starsub.visit.profiles import stack_profiles

SLICES = [(6.0, 13.0), (13.0, 15.0), (15.0, 17.0), (17.0, 19.0)]


def main():
    out = sys.argv[1]
    runs = [arg.split('=', 1) for arg in sys.argv[2:]]
    tables = {}
    edges = None
    for label, f in runs:
        with rustfits.FITS(f) as fits:
            t = fits['profiles_dmask'].read()
            stars = fits['gaia_stars'].read()
            edges = fits['dmask_edges'].read()['edges'][0]
        tables[label] = t[stars['on_image'][t['idx']] == 1]
    dmid = 0.5 * (edges[:-1] + edges[1:])
    show = [0, 1, 2, 3, 5, 8, 12, 20, 30]
    show = [i for i in show if i < dmid.size]
    print('d - r_mask (px): ' + ' '.join(f'{dmid[i]:6.0f}' for i in show))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SLICES), figsize=(4.4 * len(SLICES), 4.2))
    for ax, (g0, g1) in zip(axes, SLICES):
        first = True
        for label, t in tables.items():
            sel = (t['G'] >= g0) & (t['G'] < g1)
            if first:
                med, cnt = stack_profiles(t[sel], 'none', g0, g1,
                                          normalize=False)
                print(f'G {g0:4.1f}-{g1:4.1f} {"delivered":22s} '
                      f'({int(cnt.max()):4d} stars): '
                      + ' '.join(f'{med[i] * 1e3:+6.0f}' for i in show))
                ax.plot(dmid, med * 1e3, 'k--', lw=1, label='delivered coadd')
                first = False
            med, cnt = stack_profiles(t[sel], 'residual', g0, g1,
                                      normalize=False)
            print(f'G {g0:4.1f}-{g1:4.1f} {label:22s} '
                  f'({int(cnt.max()):4d} stars): '
                  + ' '.join(f'{med[i] * 1e3:+6.0f}' for i in show))
            ax.plot(dmid, med * 1e3, 'o-', ms=3, lw=1, label=label)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlim(dmid[0] * 0.8, dmid[-1] * 1.2)
        ax.set_ylim(-40, 40)
        ax.set_title(f'G {g0:.0f}-{g1:.0f}', fontsize=10)
        ax.set_xlabel('d - r_mask (px)')
    axes[0].set_ylabel('median residual profile (10^-3 coadd sky sigma)')
    axes[0].legend(fontsize=7)
    fig.suptitle('the residual around the stars on the patch coadd, by '
                 'route', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print('wrote', out)


if __name__ == '__main__':
    main()
