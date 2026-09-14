"""
the per-visit template as a radial wing, and the star renderer

radial_template turns a pooled per-visit template (the stack
profile inside the junction, the two-power-law halo beyond) into
(r, T); render_wing_image draws census stars from such a radial
wing, with optional per-star amplitudes.  The canonical wing per
band is the median over visits of k_in T(r)
"""

import numpy as np


# stars rendered for the response: the polynomial only sees the
# wings outside the fit mask, and below this the wings are
# negligible there
RENDER_GMAX = 16.0
# render each star out to where its wing drops below this (ADU)
RENDER_EPS = 0.02
RENDER_RMAX = 3000.0
# rows per block when rendering a star's window
RENDER_BLOCK = 256
# the junction between the stack profile and the analytic halo
# (extend_template_halo's r_blend, r_join)
R_BLEND = 40.0
R_JOIN = 44.0


def radial_template(tmpl):
    """
    the template as a radial table (r, T) in core-normalized
    template units: the denoised stack profile inside R_BLEND,
    the analytic halo (inner law plus aureole) beyond R_JOIN,
    a linear blend between, as extend_template_halo builds the
    2-d array

    Parameters
    ----------
    tmpl: dict
        From lsst_starsub.template.read_template_file

    Returns
    -------
    r, T: arrays, r from 0 to RENDER_RMAX
    """
    from .template import wing_law

    p = tmpl['params']
    prof = tmpl['prof']
    r = np.concatenate([
        np.arange(0.0, 60.0, 0.5),
        np.logspace(np.log10(60.0), np.log10(RENDER_RMAX), 400)[1:],
    ])
    halo = wing_law(r, p['slope'], p['ln_a'], p['aur_slope'], p['aur_amp'])
    stack = np.interp(r, np.arange(prof.size), prof)
    frac = np.clip((r - R_BLEND) / (R_JOIN - R_BLEND), 0.0, 1.0)
    T = (1.0 - frac) * stack + frac * halo
    T[r >= R_JOIN] = halo[r >= R_JOIN]
    return r, T


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
