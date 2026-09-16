"""
The stacked bright-star profile: the ghost disk's shape and edge.

Reads the stamp files of bright_star_stamps.py, subtracts from each
stamp the median of its far annulus (--far-in to --far-out px) and a
plane fit there, measures the star's plateau level (the median in
--lvl-in to --lvl-out px), normalizes the stamp by it, and stacks the
normalized stamps (median per pixel).  Prints the stacked radial
profile in fine annuli, the edge radius (where the profile crosses
half the plateau) per quadrant, and the plateau levels against G and
field radius; writes a png of the stack and the profiles.

usage: python bright_star_stack.py OUT.png STAMPS.fits [STAMPS.fits ...]
           [--gmin 0] [--gmax 7.5]
"""
import argparse

import numpy as np
import rustfits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('outfile')
    p.add_argument('files', nargs='+')
    p.add_argument('--gmin', type=float, default=0.0)
    p.add_argument('--gmax', type=float, default=7.5)
    p.add_argument('--far-in', type=float, default=1100)
    p.add_argument('--far-out', type=float, default=1380)
    p.add_argument('--lvl-in', type=float, default=500)
    p.add_argument('--lvl-out', type=float, default=700)
    p.add_argument('--min-far', type=int, default=2000,
                   help='usable binned pixels needed in the far annulus')
    args = p.parse_args()

    stamps, stars = [], []
    half = b = None
    for f in args.files:
        with rustfits.FITS(f) as fits:
            hdu = fits['stamps']
            s = hdu.read()
            half, b = int(hdu.header['HALF']), int(hdu.header['BIN'])
            t = fits['stars'].read()
        stamps.append(s)
        stars.append(t)
    stamps = np.concatenate(stamps)
    stars = np.concatenate(stars)
    n = stamps.shape[1]
    yy, xx = np.mgrid[0:n, 0:n]
    dx = (xx + 0.5) * b - half
    dy = (yy + 0.5) * b - half
    rr = np.hypot(dx, dy)
    th = np.degrees(np.arctan2(dy, dx))
    far = (rr >= args.far_in) & (rr < args.far_out)
    lvl = (rr >= args.lvl_in) & (rr < args.lvl_out)

    keep, normed, levels = [], [], []
    for k, (s, st) in enumerate(zip(stamps, stars)):
        if not (args.gmin <= st['G'] < args.gmax):
            continue
        ok = np.isfinite(s)
        if (ok & far).sum() < args.min_far or (ok & lvl).sum() < 200:
            continue
        # a plane through the far annulus, the local sky
        A = np.c_[dx[ok & far], dy[ok & far], np.ones((ok & far).sum())]
        coef = np.linalg.lstsq(A, s[ok & far], rcond=None)[0]
        plane = coef[0] * dx + coef[1] * dy + coef[2]
        d = s - plane
        level = np.nanmedian(d[lvl])
        if not level > 0:
            continue
        keep.append(k)
        levels.append(level)
        normed.append(d / level)
    normed = np.array(normed)
    levels = np.array(levels)
    kept = stars[keep]
    print(f'{len(keep)} of {stars.size} stars stacked (G {args.gmin}-'
          f'{args.gmax}, a far annulus {args.far_in:.0f}-{args.far_out:.0f} '
          f'px with {args.min_far} usable binned px)')
    print('plateau level (nJy) at %.0f-%.0f px by star: G, level, field '
          'radius (mm), fwhm' % (args.lvl_in, args.lvl_out))
    order = np.argsort(kept['G'])
    for i in order:
        r_fp = np.hypot(kept['fpx'][i], kept['fpy'][i])
        print(f'   G {kept["G"][i]:5.2f}  {levels[i]:6.1f}  {r_fp:6.1f}  '
              f'{kept["fwhm"][i]:.2f}  (visit {kept["visit"][i]} det '
              f'{kept["detector"][i]})')
    flux = 10.0 ** (-0.4 * kept['G'])
    ratio = levels / flux
    spread = 0.5 * np.subtract(*np.percentile(ratio, [84, 16]))
    print(f'level per unit Gaia flux: median {np.median(ratio):.3g}, half '
          f'16-84 range {spread / np.median(ratio):.2f} of it')

    stack = np.nanmedian(normed, axis=0)
    count = np.isfinite(normed).sum(axis=0)
    edges = np.arange(200, args.far_out, 25)
    mids = 0.5 * (edges[:-1] + edges[1:])
    prof = np.array([np.nanmedian(stack[(rr >= a) & (rr < c)])
                     for a, c in zip(edges[:-1], edges[1:])])
    print('\nstacked profile in units of the plateau, 25 px annuli:')
    for a, v in zip(mids, prof):
        bar = '#' * int(max(0, min(60, v * 40)))
        print(f'   {a:6.0f} px {v:+6.3f} {bar}')
    # the edge: where the profile crosses 0.5, per quadrant
    print('half-plateau edge radius (px): whole stack and per quadrant')
    for name, sel in [('all', np.ones(rr.shape, bool)),
                      ('-x', (th >= 135) | (th < -135)),
                      ('-y', (th >= -135) & (th < -45)),
                      ('+x', (th >= -45) & (th < 45)),
                      ('+y', (th >= 45) & (th < 135))]:
        pq = np.array([np.nanmedian(stack[sel & (rr >= a) & (rr < c)])
                       for a, c in zip(edges[:-1], edges[1:])])
        cross = np.flatnonzero((mids[:-1] > args.lvl_out) & (pq[:-1] >= 0.5)
                               & (pq[1:] < 0.5))
        if cross.size:
            i = cross[0]
            r_edge = mids[i] + (pq[i] - 0.5) / (pq[i] - pq[i + 1]) * 25
            print(f'   {name:4s} {r_edge:6.0f}')
        else:
            print(f'   {name:4s}   none')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    ext = [-half, half, -half, half]
    h = axes[0].imshow(stack, origin='lower', vmin=-0.3, vmax=1.5,
                       cmap='RdBu_r', extent=ext, interpolation='nearest')
    axes[0].set_title(f'stack of {len(keep)} stars, in units of each '
                      'star\'s plateau', fontsize=9)
    axes[0].set_xlabel('px')
    axes[0].set_ylabel('px')
    fig.colorbar(h, ax=axes[0], fraction=0.046)
    axes[1].imshow(count, origin='lower', extent=ext, cmap='viridis',
                   interpolation='nearest')
    axes[1].set_title('stars contributing per pixel', fontsize=9)
    axes[1].set_xlabel('px')
    axes[2].plot(mids, prof, 'o-', ms=3)
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].axhline(1, color='k', lw=0.5, ls='--')
    axes[2].set_xlabel('r (px)')
    axes[2].set_ylabel('plateau units')
    axes[2].set_title('stacked radial profile', fontsize=9)
    fig.suptitle(f'bright stars G {args.gmin}-{args.gmax}: the raw sky with '
                 'the pipeline background put back, a far-annulus plane '
                 'removed, each star normalized by its plateau')
    fig.tight_layout()
    fig.savefig(args.outfile, dpi=110)
    print('wrote', args.outfile)


if __name__ == '__main__':
    main()
