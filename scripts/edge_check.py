"""
Are the sky model's one-sided edge nodes worse than its interior?

A per-detector sky is constrained at a detector edge from one side
only.  This compares, from the box maps of a pass-2 run placed on the
focal plane, the residual (data minus sky model, stars removed) in
strips on the two sides of every gap between adjacent detectors,
against the same statistic across lines through detector interiors,
where both sides are one fit.  If the gap mismatch is no larger than
the interior mismatch, the edges lose nothing.  Also reports the jump
of the data and of each sky model across the gaps, for the pipeline's
two skies and lsst-starsub's.

usage: python edge_check.py INDIR VISIT [--detectors FILE] [--out PNG]
"""
import argparse

import numpy as np

from focal_plane_mosaic import load_focal_plane

STRIP = 4          # cells (boxes) on each side of a line
ALONG = 8          # cells per bin along the line
GAP_MAX = 8.0      # mm: neighbors are closer than this
SKIES = [('delivered', 'sky_delivered', 'delivered_starsub',
          'pipeline: per-detector polynomial sky'),
         ('skycorr', 'sky_skycorr', 'skycorr_starsub',
          'pipeline: focal-plane sky (skyCorr)'),
         ('starsub', 'sky_starsub', 'residual',
          'lsst-starsub: stars + mesh sky')]


def strips(grid, i0, j0, axis, n, x0, y0, cell):
    """
    The medians of two strips on either side of a line, binned along it.

    axis 'x': the line is vertical at column index i0 (between i0 - 1
    and i0), the strips span rows j0 .. j0 + n; axis 'y': horizontal.

    Returns
    -------
    a, b: arrays
        The strip medians per bin on the low and high side
    """
    if axis == 'x':
        lo = grid[j0:j0 + n, i0 - STRIP:i0]
        hi = grid[j0:j0 + n, i0:i0 + STRIP]
    else:
        lo = grid[i0 - STRIP:i0, j0:j0 + n].T
        hi = grid[i0:i0 + STRIP, j0:j0 + n].T
    nb = n // ALONG
    a = np.nanmedian(lo[:nb * ALONG].reshape(nb, -1), axis=1)
    b = np.nanmedian(hi[:nb * ALONG].reshape(nb, -1), axis=1)
    return a, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('indir')
    p.add_argument('visit', type=int)
    p.add_argument('--detectors', default=None)
    p.add_argument('--out', default=None, help='a png of the histograms')
    p.add_argument('--repo', default='dp2_prep_future')
    p.add_argument('--collection', default='LSSTCam/runs/DRP/DP2')
    args = p.parse_args()

    keep = None
    if args.detectors is not None:
        keep = {int(line) for line in open(args.detectors) if line.strip()}
    names = sorted({n for _, s, r, _ in SKIES for n in (s, r)}
                   | {'delivered'})
    fp = load_focal_plane(args.indir, args.visit, names, keep=keep,
                          repo=args.repo, collection=args.collection)
    g = fp.grid
    cell = fp.cell
    # the source-masked data themselves
    g['data'] = g['delivered'] + g['sky_delivered']

    def cells(v, origin):
        return int(round((v - origin) / cell))

    # adjacent pairs from the detector bounds: B to the right of A
    # (axis x) or above A (axis y), overlapping along the other axis
    pairs = []
    dets = sorted(fp.bounds)
    for a in dets:
        ax0, ax1, ay0, ay1 = fp.bounds[a]
        for b in dets:
            if b == a:
                continue
            bx0, bx1, by0, by1 = fp.bounds[b]
            if 0 < bx0 - ax1 < GAP_MAX and min(ay1, by1) - max(ay0, by0) > 0:
                pairs.append((a, b, 'x', bx0 - ax1))
            if 0 < by0 - ay1 < GAP_MAX and min(ax1, bx1) - max(ax0, bx0) > 0:
                pairs.append((a, b, 'y', by0 - ay1))
    gaps = np.array([d for _, _, _, d in pairs])
    print(f'{len(pairs)} adjacent pairs, gaps {gaps.min():.2f}-'
          f'{gaps.max():.2f} mm; strips {STRIP * cell:.2f} mm each side, '
          f'{ALONG * cell:.2f} mm bins along the line')

    # across the gaps
    jump = {k: [] for k in ['data'] + [s for _, s, _, _ in SKIES]}
    mism = {k: [] for k, _, _, _ in SKIES}
    for a, b, axis, gap in pairs:
        ax0, ax1, ay0, ay1 = fp.bounds[a]
        bx0, bx1, by0, by1 = fp.bounds[b]
        if axis == 'x':
            # the strips: A's last columns end at ax1, B's first start
            # at bx0; rows over the overlap
            j0 = cells(max(ay0, by0), fp.y0)
            n = cells(min(ay1, by1), fp.y0) - j0
            ia, ib = cells(ax1, fp.x0), cells(bx0, fp.x0)
        else:
            j0 = cells(max(ax0, bx0), fp.x0)
            n = cells(min(ax1, bx1), fp.x0) - j0
            ia, ib = cells(ay1, fp.y0), cells(by0, fp.y0)

        def two(name):
            if axis == 'x':
                lo = g[name][j0:j0 + n, ia - STRIP:ia]
                hi = g[name][j0:j0 + n, ib:ib + STRIP]
            else:
                lo = g[name][ia - STRIP:ia, j0:j0 + n].T
                hi = g[name][ib:ib + STRIP, j0:j0 + n].T
            nb = n // ALONG
            va = np.nanmedian(lo[:nb * ALONG].reshape(nb, -1), axis=1)
            vb = np.nanmedian(hi[:nb * ALONG].reshape(nb, -1), axis=1)
            return va - vb

        jump['data'].extend(two('data'))
        for k, sky, res, _ in SKIES:
            jump[sky].extend(two(sky))
            mism[k].extend(two(res))

    # across interior lines: the vertical and horizontal center lines
    # of every detector
    inner = {k: [] for k, _, _, _ in SKIES}
    for d in dets:
        dx0, dx1, dy0, dy1 = fp.bounds[d]
        for axis in ('x', 'y'):
            if axis == 'x':
                i0 = cells(0.5 * (dx0 + dx1), fp.x0)
                j0 = cells(dy0, fp.y0)
                n = cells(dy1, fp.y0) - j0
            else:
                i0 = cells(0.5 * (dy0 + dy1), fp.y0)
                j0 = cells(dx0, fp.x0)
                n = cells(dx1, fp.x0) - j0
            for k, _, res, _ in SKIES:
                a, b = strips(g[res], i0, j0, axis, n, fp.x0, fp.y0, cell)
                inner[k].extend(a - b)

    def rms(v):
        v = np.array(v)
        v = v[np.isfinite(v)]
        return np.sqrt(np.mean(v ** 2)), v.size

    print('\njump across the gaps (low side - high side, nJy), rms over '
          'bins:')
    r, n = rms(jump['data'])
    print(f'  {"data (sources masked)":42s} {r:6.2f}   ({n} bins)')
    for k, sky, _, label in SKIES:
        r, n = rms(jump[sky])
        print(f'  {label:42s} {r:6.2f}')
    print('\nresidual mismatch (low side - high side of the residual, '
          'nJy), rms over bins:')
    print(f'  {"":42s} {"gaps":>8s} {"interior":>10s}')
    for k, _, _, label in SKIES:
        rg, ng = rms(mism[k])
        ri, ni = rms(inner[k])
        print(f'  {label:42s} {rg:8.2f} {ri:10.2f}   '
              f'({ng} gap bins, {ni} interior bins)')

    if args.out is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(-6, 6, 49)
    for k, sky, _, label in SKIES:
        axes[0].hist(np.clip(jump[sky], -6, 6), bins=bins, histtype='step',
                     label=label)
    axes[0].hist(np.clip(jump['data'], -6, 6), bins=bins, histtype='step',
                 color='k', lw=1.5, label='data')
    axes[0].set_xlabel('jump across the gap (nJy)')
    axes[0].set_ylabel('bins')
    axes[0].legend(fontsize=7)
    axes[0].set_title('sky model jumps across the gaps', fontsize=10)
    for k, _, _, label in SKIES:
        rg, _ = rms(mism[k])
        ri, _ = rms(inner[k])
        line, = axes[1].hist(np.clip(mism[k], -6, 6), bins=bins,
                             histtype='step',
                             label=f'{label}: gaps {rg:.2f}')[2][:1]
        axes[1].hist(np.clip(inner[k], -6, 6), bins=bins, histtype='step',
                     ls='--', color=line.get_edgecolor(),
                     label=f'    interior {ri:.2f}')
    axes[1].set_xlabel('residual mismatch across the line (nJy)')
    axes[1].legend(fontsize=7)
    axes[1].set_title('residual mismatch: gaps (solid) vs interior '
                      'lines (dashed), rms in nJy', fontsize=10)
    fig.suptitle(f'visit {args.visit}: the sky at the detector edges')
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
