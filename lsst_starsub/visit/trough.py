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
# the visit's stack is scaled onto the canonical over this range, just
# inside the junction (visit_wing)
R_MATCH = (30.0, 40.0)


def radial_template(tmpl):
    """
    Turn a pooled template into a radial table (r, T).

    In core-normalized template units: the denoised stack profile
    inside R_BLEND, the analytic halo (inner law plus aureole) beyond
    R_JOIN, a linear blend between, as stamps.extend_template_halo
    builds the 2-d array.

    Parameters
    ----------
    tmpl: dict
        From lsst_starsub.visit.template.read_template_file

    Returns
    -------
    r, T: arrays
        r from 0 to RENDER_RMAX
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


def visit_wing(tmpl, canonical, match=R_MATCH):
    """
    Build a visit's own wing: its stack inside the junction, the canonical
    beyond.

    The pooled stack of the visit's unsaturated stars gives the core and
    inner wing at that visit's seeing; the far wing is the instrumental
    scattering, stable from visit to visit and better known from the
    survey median.  The stack is scaled onto the canonical over the
    match range just inside the junction (the median ratio), so the
    two meet: the pooled zero point k_in, fitted with the far wing profiles,
    scatters 27 percent across visits against the canonical at 40 px
    (2026-09-17), and the core amplitudes are measured against this
    wing inside 12 px, so a zero point off the canonical would put
    their outer wings off by the same factor.  Blended over
    R_BLEND-R_JOIN as radial_template blends the stack into the halo
    law.

    Parameters
    ----------
    tmpl: dict
        From lsst_starsub.visit.template.read_template_file
    canonical: WingModel or (r, T)
        The band's canonical wing
    match: (rlo, rhi), optional
        The radius range the stack is scaled onto the canonical over;
        default R_MATCH

    Returns
    -------
    wing: WingModel
        On the canonical's radial grid
    scale: float
        The factor applied to k_in times the stack
    """
    from ..wing import WingModel, profile_of

    rc, Tc = profile_of(canonical)
    r, T = radial_template(tmpl)
    k_in = float(tmpl['params']['k_in'])
    Tv = k_in * np.interp(rc, r, T)
    sel = (rc >= match[0]) & (rc <= match[1]) & (Tv > 0)
    if sel.sum() < 2:
        raise ValueError('visit_wing: no radii in the match range')
    scale = float(np.median(Tc[sel] / Tv[sel]))
    Tv = scale * Tv
    frac = np.clip((rc - R_BLEND) / (R_JOIN - R_BLEND), 0.0, 1.0)
    return WingModel(rc, (1.0 - frac) * Tv + frac * Tc), scale
