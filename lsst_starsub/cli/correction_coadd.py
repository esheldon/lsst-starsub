"""
cli/correction_coadd: the clean coadd of a patch from the per-visit
models (coadd.correction).

Writes {outdir}/correction-{tract}-{patch}-{band}.fits with the
correction plane over the coadd's inner bbox, the weight fraction
corrected and the input count per pixel, the pipeline's coadd in the
same frame (the CellCoadd with its object background restored: the
coadd of the delivered visit images) and the clean coadd, coadd plus
correction, with the input table and the run meta.
"""
import os


def get_args():
    """
    Parse the command line.

    Returns
    -------
    args: argparse.Namespace
    """
    import argparse
    from . import add_butler_arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--products', required=True,
                        help='the per-visit run directories: a pattern '
                             'with {visit}, {detector}, {band}, e.g. '
                             '/dir/{visit}/pass2/profiles-07275-55-{band}-'
                             '{visit}-{detector:03d}.fits')
    parser.add_argument('--outdir', required=True)
    add_butler_arguments(parser)
    return parser.parse_args()


def main():
    """
    Build the correction coadd of one patch.
    """
    import numpy as np
    import rustfits
    from lsst.daf.butler import Butler
    from ..coadd.cellcoadd import coadd_data_id, load_cell_coadd
    from ..coadd.correction import correction_coadd
    from ..coadd.io import _meta_table

    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    butler = Butler(args.repo, collections=[args.collection])
    did = coadd_data_id(args.tract, args.patch, args.band)
    print('loading the cell coadd for its inputs and weights')
    mcoadd = load_cell_coadd(butler, did)

    def product_path(visit, detector):
        return args.products.format(visit=visit, detector=detector,
                                    band=args.band)

    res = correction_coadd(butler, mcoadd, product_path)
    x0, y0, nx, ny = res['bbox']
    stitched = mcoadd.stitch()
    coadd = np.ascontiguousarray(stitched.image.array, dtype='f4')
    if coadd.shape != (ny, nx):
        raise RuntimeError(f'the stitched coadd is {coadd.shape}, the '
                           f'inner bbox {(ny, nx)}')
    hdr = dict(X0=x0, Y0=y0, TRACT=args.tract, PATCH=args.patch,
               BAND=args.band)
    meta = dict(tract=args.tract, patch=args.patch, band=args.band,
                x0=x0, y0=y0, collection=args.collection,
                products=args.products, ninputs=int(res['inputs'].size),
                nproducts=int(res['inputs']['has_product'].sum()))
    fname = os.path.join(
        args.outdir,
        f'correction-{args.tract:05d}-{args.patch:02d}-{args.band}.fits',
    )
    print('writing:', fname)
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_image(res['correction'], extname='correction',
                         header=hdr, compress='gzip_2')
        fits.write_image(res['wfrac'], extname='wfrac', header=hdr,
                         compress='gzip_2')
        fits.write_image(res['ninput'], extname='ninput', header=hdr,
                         compress='gzip_2')
        fits.write_image(coadd, extname='coadd', header=hdr,
                         compress='gzip_2')
        fits.write_image(coadd + res['correction'], extname='clean',
                         header=hdr, compress='gzip_2')
        fits.write_table(res['inputs'], extname='inputs')
        fits.write_table(_meta_table(meta), extname='meta')
    c = res['correction']
    print(f'    correction: median {np.median(c):+.2f} nJy, 16-84 '
          f'{np.percentile(c, 16):+.2f} .. {np.percentile(c, 84):+.2f}, '
          f'range {c.min():+.1f} .. {c.max():+.1f}')


if __name__ == '__main__':
    main()
