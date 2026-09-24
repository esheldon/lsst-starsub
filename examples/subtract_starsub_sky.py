"""
Example: subtract the joint-fit sky (and optionally the star model) of
a metadetection run from a DP2 patch coadd.

Each catalog file of a run, <run>/<tract>/<tract>-<patch>-mdet.fits,
carries the joint star and sky fit in four tables: starsub_meta (one
row per band: image shape, mesh spacing and node counts), starsub_sky
(the mesh node values per band, nJy), starsub_stars (the census stars
with their fitted amplitudes A_<band>) and starsub_wing (the band's
wing profile T(r), nJy per unit Gaia flux).

The sky is the bilinear interpolation of the mesh nodes, in the pixel
coordinates of the coadd's image array (the full patch bbox including
the overlap), and is a model of the coadd as the butler delivers it:
the mesh was fit with no stored background applied, i.e. after
deep_coadd.apply_background(None), which is the delivered state.  So

    delivered coadd - sky                (this script)
    delivered coadd - sky - stars        (with --stars)

is what the shear processing saw, up to two small terms that are not
stored (the ghost disks of stars brighter than G = 8 and the wings of
the stars fainter than the census depth).

The sky needs only numpy and rustfits; the star model needs
lsst_starsub (render_fit).

usage:
    python subtract_starsub_sky.py --run-dir RUN --tract 2078 --patch 80 \
        --band i --output out.fits [--stars]
"""

import numpy as np
import rustfits


def read_fit_tables(fname):
    """
    The four fit tables of a catalog file, extname -> array.
    """

    with rustfits.FITS(fname) as fits:
        return {
            ext: fits[ext].read()
            for ext in ('starsub_meta', 'starsub_stars', 'starsub_sky',
                        'starsub_wing')
        }


def render_sky(tables, band):
    """
    The sky image of a band from the mesh nodes: bilinear in x then y,
    nodes at 0, spacing, 2 spacing, ... in array pixel coordinates.
    Same result as lsst_starsub.joint.render_mesh.
    """

    meta = tables['starsub_meta']
    m = meta[meta['band'] == band]

    if m.size != 1:
        raise ValueError(f'band {band} not in the fit tables')

    m = m[0]
    ny, nx = int(m['ny']), int(m['nx'])
    spacing = float(m['spacing'])

    rows = tables['starsub_sky']
    rows = rows[rows['band'] == band]
    vals = np.zeros((int(m['nynode']), int(m['nxnode'])), dtype='f8')
    vals[rows['iy'], rows['ix']] = rows['value']

    xs = np.arange(nx, dtype='f8')
    ys = np.arange(ny, dtype='f8')
    ix = np.clip((xs // spacing).astype(int), 0, vals.shape[1] - 2)
    iy = np.clip((ys // spacing).astype(int), 0, vals.shape[0] - 2)
    fx = xs / spacing - ix
    fy = ys / spacing - iy

    rows_y = (1 - fy)[:, None] * vals[iy] + fy[:, None] * vals[iy + 1]
    sky = (1 - fx)[None, :] * rows_y[:, ix] + fx[None, :] * rows_y[:, ix + 1]
    return np.ascontiguousarray(sky, dtype='f4')


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--stars', action='store_true',
        help='also subtract the star model (needs lsst_starsub)',
    )
    parser.add_argument('--repo', default='dp2')
    parser.add_argument('--collections', default='dp2')
    parser.add_argument('--skymap', default='lsst_cells_v2')
    return parser.parse_args()


def main():
    from lsst.daf.butler import Butler

    args = get_args()

    fname = (
        f'{args.run_dir}/{args.tract:05d}/'
        f'{args.tract:05d}-{args.patch:05d}-mdet.fits'
    )
    print('reading fit tables from', fname)
    tables = read_fit_tables(fname)

    butler = Butler(args.repo, collections=[args.collections])
    coadd = butler.get('deep_coadd', dataId=dict(
        band=args.band, skymap=args.skymap, tract=args.tract, patch=args.patch,
    ))
    # the delivered state, in which the fit was made.  A fresh load is
    # already in it; this is a no-op unless a named background was
    # applied earlier
    coadd.apply_background(None)
    image = coadd.image.array.copy()

    sky = render_sky(tables, args.band)
    if sky.shape != image.shape:
        raise ValueError(
            f'the sky {sky.shape} does not match the coadd '
            f'{image.shape}'
        )

    model = sky
    if args.stars:
        from lsst_starsub.coadd.starsub import render_fit
        _, stars = render_fit(tables, args.band)
        model = sky + stars

    subtracted = image - model
    print(
        f'sky median {np.median(sky):+.2f} nJy, range '
        f'{sky.min():+.2f} to {sky.max():+.2f}'
    )

    bbox = coadd.bbox
    header = {
        'X0': bbox.x.start,
        'Y0': bbox.y.start,
        'BAND': args.band,
        'TRACT': args.tract,
        'PATCH': args.patch,
    }

    print('writing', args.output)

    with rustfits.FITS(args.output, 'w+') as fits:

        fits.write(subtracted, extname='image', header=header)
        fits.write(sky, extname='sky', header=header)

        if args.stars:
            fits.write(stars, extname='stars', header=header)


if __name__ == '__main__':
    main()
