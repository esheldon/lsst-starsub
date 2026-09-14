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

    Parameters
    ----------
    r: array
        Radii in px
    T: array
        The wing at r, nJy per unit Gaia flux, core included
    """
    def __init__(self, r, T):
        self.r = np.asarray(r, dtype='f8')
        self.T = np.asarray(T, dtype='f8')

    def profile(self, G=None):
        """
        Return (r, T) for a star of magnitude G.

        The same for every star; G is accepted so a model with a
        magnitude term can take its place.

        Parameters
        ----------
        G: float, optional
            The star's Gaia G, unused

        Returns
        -------
        r, T: arrays
        """
        return self.r, self.T


def profile_of(canonical, G=None):
    """
    Return the radial profile (r, T) of a wing given either way.

    Parameters
    ----------
    canonical: (r, T) or WingModel
        A plain radial table, or a model with profile(G)
    G: float, optional
        The star's Gaia G, for a model with a magnitude term

    Returns
    -------
    r, T: arrays
    """
    profile_fn = getattr(canonical, 'profile', None)
    if profile_fn is None:
        r, T = canonical
        return r, T
    return profile_fn(G)


def read_wing_model(fname):
    """
    Read a wing file into a WingModel.

    Parameters
    ----------
    fname: str
        The wing file (write_canonical_wing)

    Returns
    -------
    wing: WingModel
    """
    r, T = read_canonical_wing(fname)
    return WingModel(r, T)


def render_wing_image(shape, x, y, gmag, rt, k_in, calib,
                      gmax=RENDER_GMAX, eps=RENDER_EPS, amps=None):
    """
    Render the summed star image of an image from a radial wing.

    Each star is 10^(-0.4 G) k_in T(r) / calib times its amplitude,
    drawn out to the radius where it falls below eps.

    Parameters
    ----------
    shape: (ny, nx)
        The image shape
    x, y, gmag: arrays
        Image-frame positions and Gaia G of the stars
    rt: (r, T) or WingModel
        The radial wing (profile_of)
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
    image: array (ny, nx) f4
        The summed star image
    n: int
        The number of stars rendered
    """
    ny, nx = shape
    image = np.zeros((ny, nx), dtype='f4')
    n = 0
    if amps is None:
        amps = np.ones(len(x))
    for xk, yk, gk, ak in zip(x, y, gmag, amps):
        if not gk < gmax or not ak > 0:
            continue
        r, T = profile_of(rt, float(gk))
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


def render_canonical_stars(shape, stars, canonical, gsub=None, amps=None,
                           verbose=True):
    """
    Render the census stars from the canonical wing, in nJy.

    The pure prediction 10^(-0.4 G) T(r), with T in nJy per unit Gaia
    flux, times the per-star amplitudes when given.

    Parameters
    ----------
    shape: (ny, nx)
        The image shape
    stars: structured array
        The census, with x, y and G
    canonical: (r, T) or WingModel
        The wing
    gsub: float, optional
        Render the stars brighter than this; None for every star
    amps: array, optional
        Per-star amplitude factors, 1 the prediction
    verbose: bool, optional
        Print the number rendered and the peak

    Returns
    -------
    image: array (ny, nx) f4
    """
    image, n = render_wing_image(
        shape, stars['x'], stars['y'], stars['G'], canonical, 1.0, 1.0,
        gmax=np.inf if gsub is None else gsub, eps=0.005, amps=amps,
    )
    if verbose:
        print(f'    canonical star model: {n} stars rendered, '
              f'max {image.max():.0f} nJy')
    return image


def write_canonical_wing(fname, r, T, band, nvisit):
    """
    Write a canonical wing file.

    Parameters
    ----------
    fname: str
        The output file
    r, T: arrays
        The wing, nJy per unit Gaia flux at radius r px
    band: str
        The band, recorded in the header
    nvisit: int
        The number of visits the wing was pooled from, recorded in
        the header
    """
    import rustfits

    tab = np.zeros(r.size, dtype=[('r', 'f8'), ('T', 'f8')])
    tab['r'], tab['T'] = r, T
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(tab, extname='wing',
                         header={'band': band, 'nvisit': int(nvisit)})


def read_canonical_wing(fname):
    """
    Read a canonical wing file.

    Parameters
    ----------
    fname: str
        The wing file

    Returns
    -------
    r, T: arrays
        The wing, nJy per unit Gaia flux at radius r px
    """
    import rustfits

    with rustfits.FITS(fname) as fits:
        tab = fits['wing'].read()
    return tab['r'].astype('f8'), tab['T'].astype('f8')
