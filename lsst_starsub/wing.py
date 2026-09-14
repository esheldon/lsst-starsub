"""
The wing model, its file format and the star renderers.

The radial wing T(r) in nJy per unit Gaia flux, core included, out
to 3000 px: the canonical wing of a band, calibrated once from the
visit templates (lsst_starsub.visit.template) and checked across
tracts by the broad calibration.  Every star has the same shape;
the joint fit scales it per star.  This module is the interface
between the visit project, which makes the wing files
(write_canonical_wing), and the coadd project, which reads them
(read_wing_model) and renders the census stars from them
(render_canonical_stars, on render_wing_image).
"""
import numpy as np

from .census import GSUB


# stars rendered for the response: the polynomial only sees the
# wings outside the fit mask, and below this the wings are
# negligible there
RENDER_GMAX = 16.0
# render each star out to where its wing drops below this (ADU)
RENDER_EPS = 0.02
# rows per block when rendering a star's window
RENDER_BLOCK = 256


class WingModel(object):
    """
    The radial wing shape shared by every star.
    """
    def __init__(self, r, T):
        self.r = np.asarray(r, dtype='f8')
        self.T = np.asarray(T, dtype='f8')

    def __iter__(self):
        # `r, T = model` works wherever a plain (r, T) is expected
        return iter((self.r, self.T))

    def profile(self, G):
        """
        Return (r, T) for a star of magnitude G.

        (r, T) for a star of magnitude G: the same for all
        """
        return self.r, self.T


def read_wing_model(fname):
    """
    Read a wing file into a WingModel.
    """
    r, T = read_canonical_wing(fname)
    return WingModel(r, T)


def render_wing_image(shape, x, y, gmag, rt, k_in, calib,
                      gmax=RENDER_GMAX, eps=RENDER_EPS, amps=None):
    """
    the summed star image (ADU) on a detector from the radial
    template

    Parameters
    ----------
    shape: (ny, nx)
    x, y, gmag: arrays
        Detector-frame positions and Gaia G of the stars
    rt: (r, T), or an object with .profile(G) -> (r, T)
        From radial_template, or a lsst_starsub.wing.WingModel
        with a magnitude term
    k_in: float
        nJy per unit gaia flux per template unit
    calib: float
        nJy per ADU
    gmax: float, optional
        Render stars brighter than this
    eps: float, optional
        Each star's window ends where its wing falls below this
        (ADU)
    amps: array, optional
        Per-star amplitude factors (1 = the prediction)

    Returns
    -------
    image (ny, nx) f4, and the number of stars rendered
    """
    ny, nx = shape
    profile_fn = getattr(rt, 'profile', None)
    if profile_fn is None:
        r, T = rt
    image = np.zeros((ny, nx), dtype='f4')
    n = 0
    if amps is None:
        amps = np.ones(len(x))
    for xk, yk, gk, ak in zip(x, y, gmag, amps):
        if not gk < gmax or not ak > 0:
            continue
        if profile_fn is not None:
            r, T = profile_fn(float(gk))
        amp = ak * k_in * 10.0 ** (-0.4 * gk) / calib
        prof = amp * T
        below = np.flatnonzero(prof < eps)
        rmax = float(r[below[0]]) if below.size else float(r[-1])
        ix, iy = int(round(xk)), int(round(yk))
        m = int(np.ceil(rmax)) + 1
        x0, x1 = max(0, ix - m), min(nx, ix + m + 1)
        y0, y1 = max(0, iy - m), min(ny, iy + m + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        # the window in blocks of rows: the radii and the
        # interpolated profile are double precision, and the
        # window of the brightest stars is the whole image
        dx = np.arange(x0, x1, dtype='f8') - xk
        dy = np.arange(y0, y1, dtype='f8') - yk
        for r0 in range(0, dy.size, RENDER_BLOCK):
            r1 = min(dy.size, r0 + RENDER_BLOCK)
            rr = np.hypot(dy[r0:r1, None], dx[None, :])
            image[y0 + r0:y0 + r1, x0:x1] += np.interp(
                rr, r, prof, right=0.0,
            ).astype('f4')
        n += 1
    return image, n


def render_canonical_stars(shape, stars, canonical, gsub=GSUB, amps=None,
                           verbose=True):
    """
    the census stars' images (nJy) from the canonical wing, a pure
    prediction: 10^(-0.4 G) T(r) with T in nJy per unit Gaia flux,
    times the per-star amplitudes when given
    """
    image, n = render_wing_image(
        shape, stars['x'], stars['y'], stars['G'], canonical, 1.0, 1.0,
        gmax=gsub, eps=0.005, amps=amps,
    )
    if verbose:
        print(f'    canonical star model: {n} stars rendered, '
              f'max {image.max():.0f} nJy')
    return image


def write_canonical_wing(fname, r, T, band, nvisit):
    import rustfits

    tab = np.zeros(r.size, dtype=[('r', 'f8'), ('T', 'f8')])
    tab['r'], tab['T'] = r, T
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(tab, extname='wing',
                         header={'band': band, 'nvisit': int(nvisit)})


def read_canonical_wing(fname):
    import rustfits

    with rustfits.FITS(fname) as fits:
        tab = fits['wing'].read()
    return tab['r'].astype('f8'), tab['T'].astype('f8')
