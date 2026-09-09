"""
the wing shape derived from the coadd itself, for production
without visit images (NERSC): lsst_mdet's template machinery on
a first-pass flattened coadd, the stamp stack of the G 15.5-17.5
stars for the core and inner wing, the inner power law from the
stack (22-50 px), the aureole from the G 13-15.5 profiles, joined
as extend_template_halo does; the zero point from the stamps'
core fluxes per unit Gaia flux.  The result has the canonical
wing's form, (r, T) in nJy per unit Gaia flux out to RENDER_RMAX,
so the joint fit takes it in place of the shipped file
"""
import numpy as np

from .inject import CORE_R


def coadd_wing_shape(image, good, seg, gaia, x, y, stars, band=None,
                     fwhm=None, verbose=True):
    """
    Parameters
    ----------
    image: array
        The sky-flattened coadd (nJy), stars in
    good: bool array
    seg: array
        A segmentation of image (field_segmentation)
    gaia, x, y: the Gaia extract and its patch-frame positions
    stars: structured array
        The census (for the aureole measurement)

    Returns
    -------
    (r, T), and a dict with slope, ln_a, pedestal, aur_slope,
    aur_amp, tier, nstamp, k_stamp
    """
    from lsst_mdet.starsub import (
        denoise_template, fit_halo_slope, measure_wing_profiles,
        select_template_stars, stack_star_stamps,
    )
    from .template import wing_law
    from .trough import R_BLEND, R_JOIN, RENDER_RMAX

    sel = select_template_stars(gaia, x, y, image.shape)
    tmpl, nstamp = stack_star_stamps(image, good, seg, x, y, sel)
    tmpl, prof = denoise_template(tmpl)
    slope, ln_a, ped = fit_halo_slope(prof)
    prof = prof - ped
    # the zero point: the stamps' core flux (within CORE_R, the
    # stack's normalization) per unit Gaia flux, on the same
    # stars the stack used
    k_stamp = core_zero_point(image, good, seg, x, y, sel, gaia)

    # the aureole from the mid-bright cloud with the inner law's
    # scale fixed by the zero point (lsst_mdet's tier-1 fit
    # solves the scale and the aureole jointly, which is
    # degenerate where the aureole dominates the 40-260 px cloud
    # and inflates the amplitude several times)
    rmid, med, count, naur = measure_wing_profiles(image, good, seg, stars)
    aur_slope, aur_amp, tier = fit_aureole_fixed_scale(
        rmid, med, count, naur, slope, ln_a, k_stamp,
    )

    r = np.concatenate([
        np.arange(0.0, 60.0, 0.5),
        np.logspace(np.log10(60.0), np.log10(RENDER_RMAX), 400)[1:],
    ])
    halo = wing_law(r, slope, ln_a, aur_slope, aur_amp)
    stack = np.interp(r, np.arange(prof.size), prof)
    frac = np.clip((r - R_BLEND) / (R_JOIN - R_BLEND), 0.0, 1.0)
    T = (1.0 - frac) * stack + frac * halo
    T[r >= R_JOIN] = halo[r >= R_JOIN]
    T = np.maximum(T, 0.0) * k_stamp
    params = dict(slope=slope, ln_a=ln_a, pedestal=ped, aur_slope=aur_slope,
                  aur_amp=aur_amp, tier=tier, nstamp=nstamp, k_stamp=k_stamp)
    if verbose:
        print(f'    coadd wing shape: {nstamp} stamps, inner slope '
              f'{slope:.2f}, aureole slope {aur_slope:.2f} amp '
              f'{aur_amp:.2e} (tier {tier}), zero point {k_stamp:.3e} '
              f'nJy per unit Gaia flux')
    return (r, T), params


def fit_aureole_fixed_scale(rmid, med, count, nstars, slope, ln_a, k,
                            min_stars=3):
    """
    the aureole b r^s_aur in template units, fit to the cloud
    (nJy per unit Gaia flux) divided by the zero point k with the
    inner law fixed; the slope scanned over lsst_mdet's range;
    falls back to lsst_mdet's continuity prior when the cloud is
    unusable.  Returns aur_slope, aur_amp, tier (1 fit, 3 prior)
    """
    from lsst_mdet.starsub import (
        AUR_BREAK, AUR_RMAX, AUR_SLOPE, AUR_SLOPE_MAX, AUR_SLOPE_MIN,
        AUR_SLOPE_SEP, fit_aureole,
    )

    a_in = float(np.exp(ln_a))
    usable = (np.isfinite(med) & (med > 0) & (count >= min_stars)
              & (rmid <= AUR_RMAX))
    if usable.sum() < 3 or nstars < min_stars:
        return fit_aureole(rmid, med, count, nstars, slope, ln_a)
    r = rmid[usable]
    resid = med[usable] / k - a_in * r ** slope
    lo = max(AUR_SLOPE_MIN, slope + AUR_SLOPE_SEP)
    best = None
    for s_aur in np.arange(lo, AUR_SLOPE_MAX + 1e-9, 0.02):
        basis = r ** s_aur
        b = float(np.sum(basis * resid) / np.sum(basis * basis))
        rss = float(np.sum((resid - b * basis) ** 2))
        if best is None or rss < best[0]:
            best = (rss, s_aur, b)
    _, s_aur, b = best
    if not b > 0:
        s_aur = AUR_SLOPE
        b = a_in * AUR_BREAK ** (slope - s_aur)
        return float(s_aur), float(b), 3
    return float(s_aur), float(b), 1


def core_zero_point(image, good, seg, x, y, sel, gaia):
    """
    median over the template stars of the core flux within
    CORE_R (above the local 12-20 px median) per unit Gaia flux
    """
    ny, nx = image.shape
    m = 20
    gy, gx = np.mgrid[-m:m + 1, -m:m + 1]
    rr = np.hypot(gy, gx)
    vals = []
    for k in sel:
        ix, iy = int(round(x[k])), int(round(y[k]))
        if ix < m or iy < m or ix >= nx - m or iy >= ny - m:
            continue
        cut = image[iy - m:iy + m + 1, ix - m:ix + m + 1].astype('f8')
        ok = good[iy - m:iy + m + 1, ix - m:ix + m + 1]
        segc = seg[iy - m:iy + m + 1, ix - m:ix + m + 1]
        own = segc[m, m]
        ok = ok & ((segc == 0) | (segc == own))
        ann = ok & (rr > 12) & (rr < 20)
        if ann.sum() < 20 or not ok[rr < CORE_R].all():
            continue
        loc = np.median(cut[ann])
        core = (cut - loc)[rr < CORE_R].sum()
        g = float(gaia['phot_g_mean_mag'][k])
        if core > 0:
            vals.append(core / 10.0 ** (-0.4 * g))
    if len(vals) < 3:
        raise RuntimeError('too few stars for the core zero point')
    return float(np.median(vals))
