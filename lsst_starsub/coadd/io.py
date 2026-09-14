"""
The output files of the visit and clean tools, and their summary plot.
"""
import numpy as np


def write_visit_file(fname, vexp, res, states, edges, ptable,
                     meta, dmask=None):
    """
    Write the per-detector output file of the visit tool.

    The image states, mask planes, the census with amplitudes, the
    profile table, and the background layers as delivered by the
    pipeline.

    Parameters
    ----------
    fname: str
        The output file
    vexp: VisitExposure
        The exposure, for its stored layers, mask and variance
    res: dict
        From handle_stars_visit
    states: dict
        name -> image state
    edges: array
        The profile annulus edges
    ptable: structured array
        The per-star profile table
    meta: dict
        The run identity, written as a one-row table
    dmask: (dedges, dtable), optional
        The d - r_mask profile edges and table

    Extensions
    ----------
    delivered, flat, residual: the image states (nJy)
    sky: the total sky model subtracted by the characterization
    star_model: the summed star model
    restored: what was added back from the stored layers
    initial_coarse, initial_fine, skycorr: the stored layers
    mask: the coadd-convention mask plane
    starmask: bool star mask
    var: the variance plane
    gaia_stars: the census with fitted amplitude A
    profiles: per-star profiles (sky-sigma units) per state
    edges: the profile annulus edges
    meta: one-row table of the run identity
    """
    import rustfits

    print('writing:', fname)
    hdr = {k: (v if not isinstance(v, float) or np.isfinite(v)
               else -1.0) for k, v in meta.items()}
    with rustfits.FITS(fname, 'w+') as fits:
        for name in states:
            fits.write_image(
                np.ascontiguousarray(states[name], dtype='f4'),
                extname=name, compress='gzip_2', header=hdr,
            )
        for name in ('sky', 'star_model', 'restored'):
            fits.write_image(
                np.ascontiguousarray(res[name], dtype='f4'),
                extname=name, compress='gzip_2',
            )
        for name in ('initial_coarse', 'initial_fine', 'skycorr'):
            fits.write_image(
                np.ascontiguousarray(
                    vexp.backgrounds[name], dtype='f4',
                ),
                extname=name, compress='gzip_2',
            )
        fits.write_image(
            np.ascontiguousarray(vexp.mask.array[:, :, 0]),
            extname='mask', compress='gzip_2',
        )
        fits.write_image(
            res['starmask'].astype('u1'), extname='starmask',
            compress='gzip_2',
        )
        fits.write_image(
            np.ascontiguousarray(vexp.variance.array, dtype='f4'),
            extname='var', compress='gzip_2',
        )
        fits.write_table(res['star_table'], extname='gaia_stars')
        fits.write_table(ptable, extname='profiles')
        edges_t = np.zeros(1, dtype=[('edges', 'f8', edges.size)])
        edges_t['edges'][0] = edges
        fits.write_table(edges_t, extname='edges')
        if dmask is not None:
            dedges, dtable = dmask
            fits.write_table(dtable, extname='profiles_dmask')
            dedges_t = np.zeros(
                1, dtype=[('edges', 'f8', dedges.size)],
            )
            dedges_t['edges'][0] = dedges
            fits.write_table(dedges_t, extname='dmask_edges')
        meta_t = _meta_table(meta)
        fits.write_table(meta_t, extname='meta')


def write_profiles_file(fname, dedges, dtable, meta, star_table=None,
                        rtable=None):
    """
    Write the small per-detector output: the profiles alone.

    The d - r_mask profile table and edges (read by
    lsst-starsub-stack), the run meta, and optionally the census with
    amplitudes and the radial profile table.

    Parameters
    ----------
    fname: str
        The output file
    dedges: array
        The d - r_mask annulus edges
    dtable: structured array
        The per-star d - r_mask profile table
    meta: dict
        The run identity, written as a one-row table
    star_table: structured array, optional
        The census with amplitudes
    rtable: (edges, ptable), optional
        The radial profile edges and table
    """
    import rustfits

    print('writing:', fname)
    dedges_t = np.zeros(1, dtype=[('edges', 'f8', dedges.size)])
    dedges_t['edges'][0] = dedges
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(dtable, extname='profiles_dmask')
        fits.write_table(dedges_t, extname='dmask_edges')
        fits.write_table(_meta_table(meta), extname='meta')
        if star_table is not None:
            fits.write_table(star_table, extname='gaia_stars')
        if rtable is not None:
            edges, ptable = rtable
            fits.write_table(ptable, extname='profiles')
            edges_t = np.zeros(1, dtype=[('edges', 'f8', edges.size)])
            edges_t['edges'][0] = edges
            fits.write_table(edges_t, extname='edges')


def _meta_table(meta):
    """
    Turn a meta dict into a one-row table.

    Parameters
    ----------
    meta: dict
        name -> bool, int, float or str

    Returns
    -------
    table: structured array
    """
    dtype = []
    for k, v in meta.items():
        if isinstance(v, bool):
            dtype.append((k, 'i2'))
        elif isinstance(v, (int, np.integer)):
            dtype.append((k, 'i8'))
        elif isinstance(v, (float, np.floating)):
            dtype.append((k, 'f8'))
        else:
            dtype.append((k, f'U{max(1, len(str(v)))}'))
    t = np.zeros(1, dtype=dtype)
    for k, v in meta.items():
        t[k][0] = v
    return t


def plot_summary(png, vexp, res, states, edges, ptable):
    """
    Plot the summary: stacked profiles per state and the brightest star.

    The stacked flux-normalized profiles per state in G slices, and
    the brightest on-image star in each state.

    Parameters
    ----------
    png: str
        The output file
    vexp: VisitExposure
        The exposure
    res: dict
        From handle_stars_visit
    states: dict
        name -> image state
    edges: array
        The profile annulus edges
    ptable: structured array
        The per-star profile table
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as mplt
    from ..visit.profiles import stack_profiles

    rmid = 0.5 * (edges[:-1] + edges[1:])
    stars = res['stars']
    slices = [(7.0, 12.0), (12.0, 14.0), (14.0, 16.0), (16.0, 17.0)]
    names = list(states.keys())
    ncol = max(len(slices), len(names))

    fig, axs = mplt.subplots(
        nrows=2, ncols=ncol, figsize=(5 * ncol, 9),
    )
    for ax, (glo, ghi) in zip(axs[0], slices):
        nplot = 0
        for name in names:
            med, count = stack_profiles(
                ptable, name, glo, ghi, normalize=False,
            )
            n = int(count.max()) if count.size else 0
            if n == 0 or not np.isfinite(med).any():
                continue
            nplot += 1
            ax.plot(rmid, med, marker='o', ms=3, label=f'{name} ({n})')
        if nplot == 0:
            # an empty slice cannot carry the log axes
            ax.set_title(f'G in [{glo}, {ghi}): no stars')
            continue
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('r [px]')
        ax.set_ylabel('median profile [sky sigma]')
        ax.set_title(f'G in [{glo}, {ghi})')
        ax.legend(fontsize=8)
        ax.set_yscale('symlog', linthresh=0.05)
    for ax in axs[0][len(slices):]:
        ax.set_axis_off()

    on = stars[stars['on_image'] == 1]
    if on.size > 0:
        st = on[np.argmin(on['G'])]
        half = 600
        ix, iy = int(round(st['x'])), int(round(st['y']))
        ny, nx = vexp.image.array.shape
        y0, y1 = max(0, iy - half), min(ny, iy + half)
        x0, x1 = max(0, ix - half), min(nx, ix + half)
        sig = vexp.sky_sigma
        for ax, name in zip(axs[1], names):
            cut = states[name][y0:y1, x0:x1] / sig
            ax.imshow(
                cut, origin='lower', cmap='gray',
                vmin=-1, vmax=3, interpolation='nearest',
            )
            ax.set_title(f'{name}: G={st["G"]:.1f} star (sigma units)')
    for ax in axs[1][len(names):]:
        ax.set_axis_off()
    fig.suptitle(
        f'visit {vexp.visit} det {vexp.detector} band {vexp.band}'
    )
    fig.tight_layout()
    print('writing:', png)
    fig.savefig(png, dpi=110)
    mplt.close(fig)
