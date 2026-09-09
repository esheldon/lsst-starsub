"""
the forward-modeled trough: the visit polynomials' response to
the star wings, carried into the cell coadd

Per input (visit, detector) the census stars' wing images are
rendered from the visit's pooled template (lsst_starsub.template:
the stack profile inside the junction, the two-power-law halo
beyond, scaled by k_in 10^(-0.4 G)), the star_background fit of
calibrateImage is rebuilt (lsst_starsub.forward) and its response
to the star image, fit(raw) - fit(raw - stars), is the part of the
stored polynomial that is star light.  The response gets the same
order-2 surface removal as the stored polynomials in the
restoration, and is coadded per cell with the same input weights
(ResponseCache plugs into polynomial_cell_coadd), giving the
forward model of restored - none
"""
import os

import numpy as np

from .coadd import COARSE_FACTOR, SMOOTH_ORDER, PolynomialCache, \
    fit_smooth_surface
from .visit import INSTRUMENT

# stars rendered for the response: the polynomial only sees the
# wings outside the fit mask, and below this the wings are
# negligible there
RENDER_GMAX = 16.0
# render each star out to where its wing drops below this (ADU)
RENDER_EPS = 0.02
RENDER_RMAX = 3000.0
# the junction between the stack profile and the analytic halo
# (extend_template_halo's r_blend, r_join)
R_BLEND = 40.0
R_JOIN = 44.0


def template_path(template_dir, visit, band):
    return os.path.join(template_dir, f'template-{int(visit)}-{band}.fits')


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
        gy, gx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(gy - yk, gx - xk)
        image[y0:y1, x0:x1] += np.interp(rr, r, prof, right=0.0).astype('f4')
        n += 1
    return image, n


class ResponseCache(PolynomialCache):
    """
    per (visit, detector) coarse renderings of the polynomial's
    response to the visit template's star wings, in the entry
    layout of PolynomialCache (coarse structure, wcs, bbox,
    level) so polynomial_cell_coadd coadds them per cell with
    the input weights

    Parameters
    ----------
    butler: Butler
    template_dir: str
        Holds template-{visit}-{band}.fits per input visit
    gaia_dir: str
        Holds the per-visit Gaia files (lsst_starsub.gaia)
    band: str
    """

    def __init__(self, butler, template_dir, gaia_dir, band,
                 factor=COARSE_FACTOR, order=SMOOTH_ORDER,
                 canonical=None):
        super().__init__(butler, factor=factor, order=order)
        self.template_dir = template_dir
        self.gaia_dir = gaia_dir
        self.band = band
        # a canonical wing file (template.write_canonical_wing):
        # the same physical wing, nJy per unit Gaia flux, for every
        # visit, in place of the per-visit templates
        self.canonical = canonical
        self._templates = {}
        self._gaia = {}
        self.stats = {}

    def _spawn(self, butler):
        return ResponseCache(
            butler, self.template_dir, self.gaia_dir, self.band,
            factor=self.factor, order=self.order,
            canonical=self.canonical,
        )

    def template(self, visit):
        from .template import read_canonical_wing, read_template_file

        visit = int(visit)
        if self.canonical is not None:
            if 'canonical' not in self._templates:
                r, T = read_canonical_wing(self.canonical)
                self._templates['canonical'] = ((r, T), dict(k_in=1.0))
            return self._templates['canonical']
        if visit not in self._templates:
            t = read_template_file(
                template_path(self.template_dir, visit, self.band),
            )
            self._templates[visit] = (radial_template(t), t['params'])
        return self._templates[visit]

    def gaia(self, visit):
        import rustfits
        from .gaia import visit_gaia_path

        visit = int(visit)
        if visit not in self._gaia:
            self._gaia[visit] = rustfits.read(
                visit_gaia_path(self.gaia_dir, visit),
            )
        return self._gaia[visit]

    def get(self, visit, detector):
        key = (int(visit), int(detector))
        if key in self._models:
            return self._models[key]
        from lsst_mdet.gaia import gaia_from_columns, gaia_pixel_positions
        from lsst_mdet.patchfiles import SimpleBox
        from lsst_mdet.wcs import ButlerWcs
        from . import forward

        did = dict(instrument=INSTRUMENT, visit=int(visit),
                   detector=int(detector))
        raw, bglist, calib, meta = forward.load_raw_exposure(
            self.butler, visit, detector,
        )
        prelim = self.butler.get('preliminary_visit_image', dataId=did)
        mask, fracs = forward.reconstruct_fit_mask(raw, prelim, meta)

        wcs = prelim.getWcs()
        bbox = prelim.getBBox()
        ny, nx = raw.image.array.shape
        bwcs = ButlerWcs(wcs)
        box = SimpleBox(bbox.getBeginX(), bbox.getEndX(),
                        bbox.getBeginY(), bbox.getEndY())
        g = self.gaia(visit)
        gaia = gaia_from_columns(
            ra=g['ra'], dec=g['dec'], gmag=g['phot_g_mean_mag'],
            wcs=bwcs, bbox=box, gmax=RENDER_GMAX,
            pmra=g['pmra'], pmdec=g['pmdec'],
        )
        x, y = gaia_pixel_positions(gaia, bwcs, box)
        rt, params = self.template(visit)
        star, nstar = render_wing_image(
            (ny, nx), x, y, gaia['phot_g_mean_mag'], rt,
            params['k_in'], calib,
        )
        res = forward.polynomial_response(raw, mask, star)
        resp = res['response'] * calib
        coarse = np.ascontiguousarray(
            resp[::self.factor, ::self.factor], dtype='f8',
        )
        level = float(np.mean(coarse))
        coarse = coarse - fit_smooth_surface(coarse, self.order)
        self._models[key] = (coarse, wcs, bbox, level)
        self.stats[key] = dict(
            nstar=nstar, detected_fraction=fracs['detected_fraction'],
            response_rms=float(resp.std()), level=level,
        )
        print(
            f'    response {visit} {detector}: {nstar} stars rendered, '
            f'mask {fracs["detected_fraction"]:.3f}, rms '
            f'{resp.std():.3f} nJy, level {level:.3f}'
        )
        del raw, prelim, mask, star, res
        return self._models[key]
