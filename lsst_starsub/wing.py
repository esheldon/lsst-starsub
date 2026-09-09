"""
the wing model with an optional magnitude term

    T_G(r) = T(r) (1 + c(G, r / r_mask(G)))

with T the radial wing (nJy per unit Gaia flux, the canonical or
broad-calibrated one) and c a table over magnitude bins and bins
of the radius in units of the star's mask circle, measured as the
residual-to-model ratio of the joint fit pooled over a tract
sample (scripts/magterm_wing.py).  c is interpolated linearly in
log(r / r_mask) between bin centres and in G between bin centres,
zero inside the mask (x < 1) and held at the last bin outward.
A file without the 'magterm' HDU gives the plain wing for every
star
"""
import numpy as np


class WingModel(object):
    def __init__(self, r, T, gcen=None, xedges=None, ctab=None):
        self.r = np.asarray(r, dtype='f8')
        self.T = np.asarray(T, dtype='f8')
        self.gcen = None if gcen is None else np.asarray(gcen, dtype='f8')
        self.xedges = None if xedges is None else np.asarray(xedges, 'f8')
        self.ctab = None if ctab is None else np.asarray(ctab, dtype='f8')

    @property
    def has_magterm(self):
        return self.ctab is not None

    def __iter__(self):
        # `r, T = model` gives the base wing, so the code that
        # takes a plain (r, T) (injection truth, simulation,
        # template contexts) works unchanged
        return iter((self.r, self.T))

    def base(self):
        return self.r, self.T

    def correction(self, G, x):
        """c(G, x) for arrays x = r / r_mask"""
        if self.ctab is None:
            return np.zeros_like(np.asarray(x, dtype='f8'))
        xcen = np.sqrt(self.xedges[:-1] * self.xedges[1:])
        lx = np.log(np.maximum(np.asarray(x, dtype='f8'), 1e-3))
        # interpolate each magnitude row in log x, then between rows in G
        rows = np.array([
            np.interp(lx, np.log(xcen), row, left=0.0, right=row[-1])
            for row in self.ctab
        ])
        if self.gcen.size == 1:
            c = rows[0]
        else:
            G = float(np.clip(G, self.gcen[0], self.gcen[-1]))
            k = int(np.searchsorted(self.gcen, G, side='right') - 1)
            k = min(max(k, 0), self.gcen.size - 2)
            f = (G - self.gcen[k]) / (self.gcen[k + 1] - self.gcen[k])
            c = (1 - f) * rows[k] + f * rows[k + 1]
        c[np.asarray(x) < 1.0] = 0.0
        return c

    def profile(self, G):
        """(r, T_G) for a star of magnitude G"""
        if self.ctab is None:
            return self.r, self.T
        from lsst_mdet.starsub import circle_radius

        rad = float(circle_radius(float(G)))
        return self.r, self.T * (1.0 + self.correction(G, self.r / rad))


def as_wing_model(canonical):
    """a WingModel from a (r, T) tuple or a WingModel"""
    if isinstance(canonical, WingModel):
        return canonical
    r, T = canonical
    return WingModel(r, T)


def read_wing_model(fname):
    import rustfits

    with rustfits.FITS(fname) as fits:
        names = [h.extname for h in fits]
        tab = fits['wing'].read()
        r, T = tab['r'].astype('f8'), tab['T'].astype('f8')
        if 'magterm' not in names:
            return WingModel(r, T)
        mt = fits['magterm'].read()
        xe = fits['magterm_xedges'].read()['xedges'][0]
    gcen = 0.5 * (mt['glo'] + mt['ghi'])
    return WingModel(r, T, gcen=gcen, xedges=xe, ctab=mt['c'])


def write_wing_model(fname, model, band='i', nstar=0):
    import rustfits
    from .template import write_canonical_wing

    write_canonical_wing(fname, model.r, model.T, band, nstar)
    if model.ctab is None:
        return
    nx = model.xedges.size - 1
    mt = np.zeros(model.gcen.size, dtype=[('glo', 'f8'), ('ghi', 'f8'),
                                          ('c', 'f8', nx)])
    # the bins are stored by their centres' neighbours: glo/ghi
    # reconstruct gcen as their mean
    half = np.diff(model.gcen).mean() / 2 if model.gcen.size > 1 else 0.5
    mt['glo'] = model.gcen - half
    mt['ghi'] = model.gcen + half
    mt['c'] = model.ctab
    xe = np.zeros(1, dtype=[('xedges', 'f8', model.xedges.size)])
    xe['xedges'][0] = model.xedges
    with rustfits.FITS(fname, 'r+') as fits:
        fits.write_table(mt, extname='magterm')
        fits.write_table(xe, extname='magterm_xedges')
