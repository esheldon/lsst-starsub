"""
the wing model: the radial wing T(r) in nJy per unit Gaia flux,
core included, out to 3000 px (the canonical wing of
lsst_starsub.template, calibrated once per band from the visit
templates and checked across tracts by the broad calibration).
Every star has the same shape; the joint fit scales it per star
"""
import numpy as np


class WingModel(object):
    def __init__(self, r, T):
        self.r = np.asarray(r, dtype='f8')
        self.T = np.asarray(T, dtype='f8')

    def __iter__(self):
        # `r, T = model` works wherever a plain (r, T) is expected
        return iter((self.r, self.T))

    def profile(self, G):
        """(r, T) for a star of magnitude G: the same for all"""
        return self.r, self.T


def read_wing_model(fname):
    from .template import read_canonical_wing

    r, T = read_canonical_wing(fname)
    return WingModel(r, T)
