"""
cli/stack: stack the per-star d - r_mask profiles of the
lsst-starsub-visit outputs over detectors, per image state and
G slice, in the layout of the coadd dual-state stacks (mean and
median across stars, units of 10^-3 sky sigma), and write the
plot and a table of the stacks
"""
import os

import numpy as np

# the coadd dual-state G slices
G_SLICES = [
    (6.0, 13.0), (13.0, 14.0), (14.0, 15.2), (15.2, 16.0),
    (16.0, 17.0),
]


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'files', nargs='+', help='lsst-starsub-visit fits outputs',
    )
    parser.add_argument('--output', required=True, help='pdf/png')
    parser.add_argument(
        '--states', nargs='+', default=['delivered', 'warp', 'flat'],
    )
    parser.add_argument(
        '--table', help='write the stacks to this fits file',
    )
    parser.add_argument('--title', default='')
    parser.add_argument(
        '--ref', default='global', choices=['global', 'local'],
        help='reference each profile to the ambient level '
             '(global) or to the star\'s own local annulus '
             '(local; rows without one are dropped)',
    )
    parser.add_argument(
        '--clip', type=float, default=3.0,
        help='sigma clipping of the mean across stars per bin '
             '(0 disables)',
    )
    return parser.parse_args()


def load_profiles(files):
    """
    the concatenated profiles_dmask tables with a file index,
    and the common edges
    """
    import rustfits

    tables = []
    edges = None
    for i, fname in enumerate(files):
        with rustfits.FITS(fname) as fits:
            t = fits['profiles_dmask'].read()
            e = fits['dmask_edges'].read()['edges'][0]
        if edges is None:
            edges = e
        elif not np.allclose(edges, e):
            raise ValueError(f'{fname}: different dmask edges')
        ext = np.zeros(t.size, dtype=t.dtype.descr + [('file', 'i4')])
        for name in t.dtype.names:
            ext[name] = t[name]
        ext['file'] = i
        tables.append(ext)
    return edges, np.concatenate(tables)


def clipped_mean(vals, clip, niter=5):
    """
    sigma-clipped mean and its error of a 1-d array (NaNs
    ignored); clip <= 0 disables the clipping
    """
    v = vals[np.isfinite(vals)]
    if v.size < 2:
        return np.nan, np.nan
    keep = np.ones(v.size, dtype=bool)
    if clip > 0:
        for _ in range(niter):
            m = np.median(v[keep])
            s = 1.4826 * np.median(np.abs(v[keep] - m))
            if not s > 0:
                break
            new = np.abs(v - m) <= clip * s
            if np.array_equal(new, keep):
                break
            keep = new
    vk = v[keep]
    return float(vk.mean()), float(vk.std() / np.sqrt(vk.size))


def stack(table, state, glo, ghi, ref='global', clip=3.0):
    """
    clipped mean, its error, median and count across stars for
    one state and G slice, in 10^-3 sky sigma, with the profile
    referenced globally (as stored) or to each star's local
    level
    """
    w = (
        (table['state'] == state)
        & (table['G'] >= glo) & (table['G'] < ghi)
    )
    profs = table['prof'][w] * 1.0e3
    if ref == 'local':
        loc = table['local'][w] * 1.0e3
        has = np.isfinite(loc)
        profs = profs[has] - loc[has][:, np.newaxis]
    nb = profs.shape[1]
    count = np.sum(np.isfinite(profs), axis=0)
    mean = np.full(nb, np.nan)
    err = np.full(nb, np.nan)
    med = np.full(nb, np.nan)
    for k in range(nb):
        if count[k] >= 2:
            mean[k], err[k] = clipped_mean(profs[:, k], clip)
            med[k] = np.nanmedian(profs[:, k])
    nstars = int(profs.shape[0])
    return mean, err, med, count, nstars


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as mplt

    args = get_args()
    edges, table = load_profiles(args.files)
    dmid = 0.5 * (edges[:-1] + edges[1:])
    print(f'{len(args.files)} files, {table.size} profile rows')

    nsl = len(G_SLICES)
    ncol = 3
    nrow = int(np.ceil(nsl / ncol))
    fig, axs = mplt.subplots(
        nrows=nrow, ncols=ncol, figsize=(5 * ncol, 4 * nrow),
        squeeze=False,
    )
    rows = []
    colors = mplt.rcParams['axes.prop_cycle'].by_key()['color']
    for k, (glo, ghi) in enumerate(G_SLICES):
        ax = axs[k // ncol][k % ncol]
        nst = 0
        for j, state in enumerate(args.states):
            mean, err, med, count, nstars = stack(
                table, state, glo, ghi, ref=args.ref, clip=args.clip,
            )
            nst = max(nst, nstars)
            c = colors[j % len(colors)]
            ax.errorbar(
                dmid, mean, yerr=err, color=c, marker='o', ms=2,
                lw=1, label=f'{state} (mean)',
            )
            ax.plot(
                dmid, med, color=c, ls='--', lw=1,
                label=f'{state} (median)',
            )
            rows.append((state, glo, ghi, nstars, mean, err, med, count))
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylim(-60, 100)
        ax.set_xlabel('d - r_mask [pixels]')
        ax.set_ylabel('stacked profile [10^-3 sigma_sky]')
        ax.set_title(f'G {glo}-{ghi} ({nst} stars)')
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(nsl, nrow * ncol):
        axs[k // ncol][k % ncol].set_axis_off()
    fig.suptitle(
        (args.title or 'visit-level wing profile stacks')
        + f' [{args.ref} reference, clip {args.clip:g}]'
    )
    fig.tight_layout()
    if not os.path.splitext(args.output)[1]:
        args.output += '.png'
    print('writing:', args.output)
    fig.savefig(args.output, dpi=110)

    # a text summary at a few distances
    picks = [0, 5, 10, 20, 30, 45]
    print('stacked mean [10^-3 sigma] at d - r_mask =',
          [int(dmid[p]) for p in picks if p < dmid.size])
    for state, glo, ghi, nstars, mean, err, med, count in rows:
        vals = ' '.join(
            f'{mean[p]:6.1f}+-{err[p]:4.1f}'
            for p in picks if p < mean.size
        )
        print(f'  G {glo:4.1f}-{ghi:4.1f} {state:9s} n={nstars:4d}: {vals}')

    if args.table:
        import rustfits
        nb = dmid.size
        out = np.zeros(len(rows), dtype=[
            ('state', 'U12'), ('glo', 'f4'), ('ghi', 'f4'),
            ('nstars', 'i4'), ('mean', 'f4', nb), ('err', 'f4', nb),
            ('median', 'f4', nb), ('count', 'i4', nb),
        ])
        for i, row in enumerate(rows):
            out[i] = row
        edges_t = np.zeros(1, dtype=[('edges', 'f8', edges.size)])
        edges_t['edges'][0] = edges
        print('writing:', args.table)
        with rustfits.FITS(args.table, 'w+') as fits:
            fits.write_table(out, extname='stacks')
            fits.write_table(edges_t, extname='edges')

    if not os.path.exists(args.output):
        raise RuntimeError('plot not written')


if __name__ == '__main__':
    main()
