"""
cli/forward_check: the forward model of the visit polynomial on
one input: rebuild the mask the star_background fit saw, refit
the raw sky image and compare with the stored layer 0, then
measure the polynomial's response to a star model image (the
star_model extension of an lsst-starsub-visit output file) and
profile it around the brightest census stars against the stored
polynomial's structure

Writes forward-VISIT-DET.png (profiles), forward-images-VISIT-DET.png
(the surfaces and the mask) and forward-VISIT-DET.fits (stored,
fit, response surfaces in ADU and the fit mask) to --outdir
"""
import sys
import time
import numpy as np


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
    parser.add_argument('--visit', type=int, required=True)
    parser.add_argument('--detector', type=int, required=True)
    parser.add_argument(
        '--visit-file', required=True,
        help='lsst-starsub-visit output with star_model and gaia_stars',
    )
    parser.add_argument('--outdir', default='.')
    add_butler_arguments(parser)
    parser.add_argument('--nstars', type=int, default=3,
                        help='brightest stars to profile')
    parser.add_argument('--rmax', type=float, default=2000)
    return parser.parse_args()


def annulus_profile(arr, ir, nbin, sel=None):
    """
    Take the mean of an image in annuli.

    Parameters
    ----------
    arr: array
    ir: int array
        The annulus index of each pixel
    nbin: int
        The number of annuli
    sel: bool array, optional
        Pixels used; all when None

    Returns
    -------
    prof: array
        NaN where an annulus is empty
    """
    if sel is None:
        sel = np.ones(arr.shape, dtype=bool)
    out = np.full(nbin, np.nan)
    for k in range(nbin):
        w = (ir == k) & sel
        if w.any():
            out[k] = np.mean(arr[w])
    return out


def main():
    """
    Run the forward-model check of the visit polynomial on one input.
    """
    import rustfits
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..coadd.cellcoadd import SMOOTH_ORDER, fit_smooth_surface
    from ..site import INSTRUMENT
    from ..visit.exposure import make_visit_butler
    from ..visit import forward

    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()
    visit, det = args.visit, args.detector
    butler = make_visit_butler(args.repo, args.collection)

    t0 = time.time()
    raw, bglist, calib, meta = forward.load_raw_exposure(butler, visit, det)
    prelim = butler.get(
        'preliminary_visit_image',
        dataId=dict(instrument=INSTRUMENT, visit=visit, detector=det),
    )
    print(f'loaded in {time.time() - t0:.0f} s; calib {calib:.4f} nJy/ADU')
    print(f'adaptive threshold {meta["adaptive_threshold"]:.1f}, '
          f'raw median {np.median(raw.image.array):.1f} ADU')

    mask, fracs = forward.reconstruct_fit_mask(raw, prelim, meta)
    print(f'fit mask: detected fraction {fracs["detected_fraction"]:.4f} '
          f'(first pass dilated {fracs["psf_dilated_fraction"]:.4f})')

    with rustfits.FITS(args.visit_file) as fits:
        star_model = fits['star_model'].read() / calib
        stars = fits['gaia_stars'].read()

    res = forward.polynomial_response(raw, mask, star_model)
    stored = forward.stored_surface(bglist)
    fit0 = res['fit_raw']
    resp = res['response']

    def structure(a):
        """The image minus its smooth surface."""
        return a - fit_smooth_surface(a, SMOOTH_ORDER)

    sstruct, fstruct, rstruct = structure(stored), structure(fit0), \
        structure(resp)
    diff = fit0 - stored
    print(f'refit - stored: mean {diff.mean() * calib:.4f} rms '
          f'{diff.std() * calib:.4f} nJy; structure rms stored '
          f'{sstruct.std() * calib:.4f} refit {fstruct.std() * calib:.4f} '
          f'difference {(fstruct - sstruct).std() * calib:.4f}')
    print(f'response rms {resp.std() * calib:.4f} nJy, structure '
          f'{rstruct.std() * calib:.4f}')

    ny, nx = stored.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    badbits = mask.getPlaneBitMask(
        forward.DETECTED_PLANES + ['BAD', 'EDGE', 'NO_DATA'],
    )
    unmasked = (mask.array & badbits) == 0
    edges = np.arange(0, args.rmax + 1, 50)
    rc = 0.5 * (edges[1:] + edges[:-1])
    nbin = len(rc)

    ib = np.argsort(stars['G'])[:args.nstars]
    fig, axes = plt.subplots(2, len(ib), figsize=(5 * len(ib), 8),
                             squeeze=False)
    for col, i in enumerate(ib):
        rr = np.hypot(xx - stars['x'][i], yy - stars['y'][i])
        rmask = float(rr[unmasked].min())
        ir = np.digitize(rr, edges) - 1
        prof = dict(
            stored=annulus_profile(sstruct, ir, nbin) * calib,
            refit=annulus_profile(fstruct, ir, nbin) * calib,
            response=annulus_profile(rstruct, ir, nbin) * calib,
            model=annulus_profile(star_model, ir, nbin, unmasked) * calib,
        )
        print(f'star G {stars["G"][i]:.2f} at ({stars["x"][i]:.0f}, '
              f'{stars["y"][i]:.0f}): mask radius {rmask:.0f} px')
        print('     r   stored    refit response  model(unmasked)  [nJy]')
        for k in range(0, nbin, max(1, nbin // 10)):
            print(f'  {rc[k]:5.0f} {prof["stored"][k]:8.3f} '
                  f'{prof["refit"][k]:8.3f} {prof["response"][k]:8.3f} '
                  f'{prof["model"][k]:10.3f}')
        ax = axes[0, col]
        for name in ('stored', 'refit', 'response'):
            ax.plot(rc, prof[name], label=name)
        ax.plot(rc, prof['stored'] - prof['response'], 'k--',
                label='stored - response')
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(rmask, color='gray', ls=':', label='mask radius')
        ax.set_title(f'G {stars["G"][i]:.2f}')
        ax.set_xlabel('r [px]')
        ax.set_ylabel('structure [nJy]')
        ax.legend(fontsize=7)
        ax = axes[1, col]
        ax.semilogy(rc, prof['model'], label='star model, unmasked px')
        ax.semilogy(rc, np.abs(prof['response']), label='|response|')
        ax.axvline(rmask, color='gray', ls=':')
        ax.set_xlabel('r [px]')
        ax.set_ylabel('nJy')
        ax.legend(fontsize=7)
    fig.suptitle(f'{visit} {det}')
    fig.tight_layout()
    pname = f'{args.outdir}/forward-{visit}-{det:03d}.png'
    fig.savefig(pname, dpi=110)
    print('wrote', pname)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ims = [('stored structure', sstruct), ('refit structure', fstruct),
           ('refit - stored', diff), ('response structure', rstruct),
           ('stored - response', sstruct - rstruct), ('fit mask', None)]
    v = 3 * np.std(sstruct * calib)
    for ax, (name, arr) in zip(axes.ravel(), ims):
        if arr is None:
            ax.imshow(~unmasked, origin='lower', cmap='gray')
        else:
            ax.imshow(arr * calib, origin='lower', vmin=-v, vmax=v,
                      cmap='RdBu_r')
        ax.set_title(name)
    fig.tight_layout()
    pname = f'{args.outdir}/forward-images-{visit}-{det:03d}.png'
    fig.savefig(pname, dpi=90)
    print('wrote', pname)

    fname = f'{args.outdir}/forward-{visit}-{det:03d}.fits'
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_image(stored.astype('f4'), extname='stored',
                         header={'calib': calib})
        fits.write_image(fit0.astype('f4'), extname='refit')
        fits.write_image(resp.astype('f4'), extname='response')
        fits.write_image(mask.array, extname='fitmask')
    print('wrote', fname)


if __name__ == '__main__':
    main()
