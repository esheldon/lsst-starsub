"""
The per-visit template as a radial wing.

radial_template turns a pooled per-visit template (the stack
profile inside the junction, the two-power-law halo beyond) into
(r, T), which lsst_starsub.wing.render_wing_image draws.  The
canonical wing per band is the median over visits of k_in T(r)
(lsst_starsub.visit.template.canonical_wing).
"""

import numpy as np


RENDER_RMAX = 3000.0
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
        From lsst_starsub.visit.template.read_template_file

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
