"""
The ideal-conditions simulation of a patch through the background pass.

A patch's worth of sky on one pixel grid, seen in nvisit visits of
different seeing, with the real Gaia census of the patch as the
stars and the canonical wing as their truth; the visits go through
the calibrateImage-style star_background pass (the polynomial that
makes the trough), the coadd is their mean, and the cleaning runs
on the result with every star's truth known.

Units are nJy per pixel throughout (calibration 1).  The visit
sky level and noise follow the real i-band numbers (sky ~1900
nJy, sigma ~26 nJy per visit pixel), so the adaptive detection
threshold of the background pass (0.2 x the median sky in pixel
sigma units) lands where it does on the data.

Per visit the star image is the Gaussian PSF core of that visit's
FWHM, scaled to the canonical wing's core flux (the sum within
CORE_R, as the injection does), blended over BLEND_R0..BLEND_R1
into the canonical wing, times a per-star amplitude that is 1 or
a lognormal draw with amp_scatter.  Pixels above sat_level are
clipped and flagged SAT.
"""
import numpy as np

from ..wing import profile_of

from ..inject import BLEND_R0, BLEND_R1, CORE_R

DEFAULTS = dict(
    nvisit=20,
    fwhm_min=0.7,       # arcsec
    fwhm_max=1.3,
    pixel_scale=0.2,    # arcsec
    sky_level=1900.0,   # nJy per visit pixel
    sky_gradient=0.02,  # fractional change across the image
    noise_sigma=26.0,   # nJy per visit pixel
    sat_level=1.0e5,    # nJy per visit pixel
    render_gmax=19.0,
    detector_size=4100,  # each visit is fit on a detector-sized image
                         # with the patch at a random offset inside
                         # it (0: the patch itself)
    amp_scatter=0.0,    # lognormal sigma of the per-star wing amplitude
    eps=0.005,          # nJy; render each star out to where it falls below
)


def sky_image(shape, level, gradient, rng):
    """
    Make a smooth sky with a random linear gradient.

    Parameters
    ----------
    shape: (ny, nx)
    level: float
        The sky level, nJy
    gradient: float
        The fractional change across the image, in a random
        direction
    rng: numpy Generator

    Returns
    -------
    sky: f4 array
    """
    ny, nx = shape
    theta = rng.uniform(0, 2 * np.pi)
    gy, gx = np.mgrid[0:ny, 0:nx]
    u = ((gx - nx / 2) * np.cos(theta) + (gy - ny / 2) * np.sin(theta))
    u /= max(nx, ny)
    return (level * (1.0 + gradient * u)).astype('f4')


def gaussian_core(rr, sigma):
    """
    Evaluate a unit-flux Gaussian core at the given radii.

    Parameters
    ----------
    rr: array
        Radii in pixels
    sigma: float
        The Gaussian sigma in pixels

    Returns
    -------
    core: array
    """
    return np.exp(-0.5 * (rr / sigma) ** 2) / (2 * np.pi * sigma ** 2)


def core_factor_gaussian(sigma, canonical):
    """
    Get the factor scaling a Gaussian core to the canonical core flux.

    The canonical wing's flux within CORE_R over a unit Gaussian
    core's.

    Parameters
    ----------
    sigma: float
        The Gaussian sigma in pixels
    canonical: WingModel or (r, T)

    Returns
    -------
    factor: float
    """
    r, T = profile_of(canonical)

    m = int(np.ceil(CORE_R)) + 1
    gy, gx = np.mgrid[-m:m + 1, -m:m + 1]
    rr = np.hypot(gy, gx)

    psf_core = gaussian_core(rr, sigma)[rr < CORE_R].sum()
    can_core = np.interp(rr, r, T)[rr < CORE_R].sum()

    return can_core / psf_core


def render_stars(
    shape, x, y, G, amp, canonical, fwhm_px, eps=DEFAULTS['eps'],
    gmax=DEFAULTS['render_gmax'],
):
    """
    Render the summed star image of one visit.

    Each star is the Gaussian core of the visit's FWHM scaled to
    the canonical core flux, blended into the canonical wing over
    BLEND_R0..BLEND_R1.

    Parameters
    ----------
    shape: (ny, nx)
    x, y, G, amp: arrays
        Positions (pixels, fractional), Gaia G and the per-star
        wing amplitude factor
    canonical: WingModel or (r, T)
    fwhm_px: float
        The visit PSF FWHM in pixels
    eps: float, optional
        Render each star out to where its wing falls below this,
        nJy
    gmax: float, optional
        Render stars brighter than this

    Returns
    -------
    image: f4 array
        In nJy
    n: int
        The stars rendered
    """

    ny, nx = shape

    r, T = profile_of(canonical)
    sigma = fwhm_px / 2.3548

    C = core_factor_gaussian(sigma, canonical)
    image = np.zeros((ny, nx), dtype='f8')

    n = 0

    for xk, yk, gk, ak in zip(x, y, G, amp):

        if not gk < gmax:
            continue

        flux = 10.0 ** (-0.4 * gk)
        prof = flux * ak * T
        below = np.flatnonzero(prof < eps)

        rmax = float(r[below[0]]) if below.size else float(r[-1])
        rmax = max(rmax, 6 * sigma)
        m = int(np.ceil(rmax)) + 1

        ix, iy = int(round(xk)), int(round(yk))
        x0, x1 = max(0, ix - m), min(nx, ix + m + 1)
        y0, y1 = max(0, iy - m), min(ny, iy + m + 1)

        if x1 <= x0 or y1 <= y0:
            continue

        gy, gx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(gy - yk, gx - xk)
        f = np.clip((rr - BLEND_R0) / (BLEND_R1 - BLEND_R0), 0.0, 1.0)
        star = f * np.interp(rr, r, prof, right=0.0)
        star += (1.0 - f) * C * flux * gaussian_core(rr, sigma)
        image[y0:y1, x0:x1] += star
        n += 1

    return image.astype('f4'), n


MAX_DET_FRAC = 0.93
MIN_DET_FRAC = 0.02
AMP_GRID = (2, 8)   # amplifier layout (rows, columns) of the detector


def amp_fractions(detected, grid=AMP_GRID):
    """
    Compute the detected fraction per amplifier region.

    Parameters
    ----------
    detected: bool array
    grid: (rows, columns), optional
        The amplifier layout

    Returns
    -------
    fractions: array
        One per amplifier, row-major
    """
    ny, nx = detected.shape
    fr = []

    for i in range(grid[0]):
        for j in range(grid[1]):
            c = np.s_[i * ny // grid[0]:(i + 1) * ny // grid[0],
                      j * nx // grid[1]:(j + 1) * nx // grid[1]]
            fr.append(float(detected[c].mean()))

    return np.array(fr)


def adaptive_detection(exp, dilated, det, thresh, grow, max_iter=40):
    """
    Run calibrateImage's adaptive star-background detection loop.

    As _remeasure_star_background: detect at thresh (pixel_stdev
    units) with the footprints grown, OR in the dilated first-pass
    mask, and adjust the threshold until the detected fraction
    is between MIN_DET_FRAC and MAX_DET_FRAC, fewer than 15
    percent of the amplifiers are above the maximum (or the
    fraction is below 0.85 x the maximum), no amplifier is
    without detections, and enough footprints remain.

    Parameters
    ----------
    exp: afw ExposureF
        The raw sky image with its psf set
    dilated: lsst.afw.image.Mask
        The dilated first-pass detection mask
    det: int
        The bit mask of the DETECTED planes
    thresh: float
        The starting threshold, pixel_stdev units
    grow: float
        Footprint growth in psf sigma
    max_iter: int, optional

    Returns
    -------
    mask: lsst.afw.image.Mask
    frac: float
        The detected fraction
    thresh: float
        The final threshold
    """
    from .forward import _clear_detected, _detect

    n_amp = AMP_GRID[0] * AMP_GRID[1]
    frac = 1.0
    nfoot = 10 ** 12
    n_above = -99
    zero_amps = False

    for it in range(max_iter):
        cur = thresh

        if frac > MAX_DET_FRAC or nfoot <= 3:
            thresh = 1.07 * cur

        if n_above > 1:
            thresh = 1.1 * cur

        if frac < MIN_DET_FRAC:
            thresh = 0.8 * cur

        if zero_amps:
            thresh = 0.95 * cur

        det_exp = exp.clone()

        _clear_detected(det_exp.mask)

        res = _detect(det_exp, thresh, 1.0, grow=grow)
        mask = det_exp.mask
        mask |= dilated
        detected = (mask.array & det) != 0

        frac = float(detected.mean())
        nfoot = len(res.sources)
        minfoot = min(200, max(3, int(0.01 * res.numPosPeaks)))
        fr = amp_fractions(detected)
        n_above = int((fr > MAX_DET_FRAC).sum())
        zero_amps = bool((fr == 0).any())

        if (MIN_DET_FRAC < frac < MAX_DET_FRAC and n_above < 0.75 * n_amp
                and not zero_amps and nfoot >= minfoot):
            if (n_above < max(1, int(0.15 * n_amp))
                    or frac < 0.85 * MAX_DET_FRAC):
                break
            thresh = 1.07 * cur

    return mask, frac, thresh


def dm_background_pass(raw, var, satmask, fwhm_px, star_image=None):
    """
    Run the calibrateImage star_background pass on one visit.

    The first-pass 50 sigma detection on a 128 px sep-flattened
    image (grown 2.4 sigma, dilated 10 px), the adaptive detection
    on the raw sky image starting at 0.2 x median sky in pixel
    sigma units and adjusted as adaptive_detection does, footprints
    grown 70 psf sigma, then the weighted 6x6 Chebyshev fit with
    those planes ignored.

    Parameters
    ----------
    raw: array
        The raw sky image, nJy
    var: array
        Its variance
    satmask: bool array
        The saturated pixels
    fwhm_px: float
        The visit PSF FWHM in pixels
    star_image: array, optional
        The true star image; when given the polynomial's response
        to it is measured

    Returns
    -------
    res: dict
        fit (the surface), delivered (raw - fit), mask (bool, the
        pixels the fit ignored), threshold, detected_fraction, and
        when star_image is given response (fit(raw) - fit(raw -
        stars))
    """
    import sep
    import lsst.afw.image as afwImage
    import lsst.geom as geom
    from lsst.meas.algorithms import SingleGaussianPsf
    from .forward import (
        DETECTED_PLANES,
        PSF_DET_DILATE,
        PSF_DET_MULTIPLIER,
        PSF_DET_THRESHOLD,
        STAR_BG_GROW_SIGMA,
        _clear_detected,
        _detect,
        _dilate_detected,
        fit_star_background,
    )

    ny, nx = raw.shape
    mi = afwImage.MaskedImageF(
        geom.Box2I(
            geom.Point2I(0, 0),
            geom.Extent2I(nx, ny),
        ),
    )

    mi.image.array[:, :] = raw
    mi.variance.array[:, :] = var

    sat = mi.mask.getPlaneBitMask('SAT')
    mi.mask.array[satmask] |= sat

    exp = afwImage.ExposureF(mi)

    width = int(2 * np.ceil(4 * fwhm_px)) + 1
    exp.setPsf(SingleGaussianPsf(width, width, fwhm_px / 2.3548))

    # first pass on a plainly flattened image
    bkg0 = sep.Background(
        np.ascontiguousarray(raw, dtype='f4'),
        bw=128, bh=128,
    )

    prelim = exp.clone()
    prelim.image.array[:, :] -= bkg0.back()

    _clear_detected(prelim.mask)
    _detect(prelim, PSF_DET_THRESHOLD, PSF_DET_MULTIPLIER, grow=2.4)
    dilated = prelim.mask.clone()
    _dilate_detected(dilated, PSF_DET_DILATE)

    del prelim

    det = afwImage.Mask.getPlaneBitMask(DETECTED_PLANES)
    median_sky = float(np.median(raw))
    thresh = max(2.0, 0.2 * median_sky)
    mask, frac, thresh = adaptive_detection(
        exp, dilated, det, thresh, grow=STAR_BG_GROW_SIGMA,
    )

    fit, _ = fit_star_background(raw, mask, var)

    out = dict(
        fit=fit, delivered=raw - fit, mask=(mask.array & det) != 0,
        threshold=thresh, detected_fraction=frac,
    )

    if star_image is not None:
        fit_nostar, _ = fit_star_background(raw - star_image, mask, var)
        out['response'] = fit - fit_nostar

    return out


def simulate_visit(k, shape, x, y, G, amp, canonical, cfg, seed):
    """
    Simulate one visit through the background pass.

    Sky, stars, noise, saturation, then the background pass with
    the response to the true stars, on a detector-sized frame with
    the patch at a random offset.

    Parameters
    ----------
    k: int
        The visit index, for the printout
    shape: (ny, nx)
        The patch
    x, y, G, amp: arrays
        The stars (render_stars)
    canonical: WingModel or (r, T)
    cfg: dict
        The settings (DEFAULTS)
    seed: int

    Returns
    -------
    visit: dict
        fwhm (arcsec), delivered, response, stars, sky, raw, var,
        satmask (cut to the patch), threshold, detected_fraction,
        offset
    """
    rng = np.random.default_rng(seed)

    fwhm = float(rng.uniform(cfg['fwhm_min'], cfg['fwhm_max']))
    fwhm_px = fwhm / cfg['pixel_scale']

    # the detector frame: the patch at a random offset, so the
    # polynomial's domain and bins differ between visits as the
    # dithers make them on the data
    det = int(cfg['detector_size'])
    ny, nx = shape

    if det > max(ny, nx):
        ox = int(rng.integers(0, det - nx + 1))
        oy = int(rng.integers(0, det - ny + 1))
        big = (det, det)
    else:
        ox, oy, big = 0, 0, shape

    cut = np.s_[oy:oy + ny, ox:ox + nx]
    sky = sky_image(big, cfg['sky_level'], cfg['sky_gradient'], rng)

    stars, nstar = render_stars(
        big, x + ox, y + oy, G, amp, canonical, fwhm_px, eps=cfg['eps'],
        gmax=cfg['render_gmax'],
    )

    # the variance follows the image as on the data (Poisson in
    # electrons): noise_sigma^2 at the sky level, more on the
    # stars, which is what keeps the moderately bright stars out
    # of the adaptive detection (pixel_stdev thresholds)

    var_factor = cfg['noise_sigma'] ** 2 / cfg['sky_level']
    var = (var_factor * (sky + stars)).astype('f4')
    raw = sky + stars + rng.normal(size=big).astype('f4') * np.sqrt(var)

    satmask = raw > cfg['sat_level']
    raw[satmask] = cfg['sat_level']
    var[satmask] = var_factor * cfg['sat_level']
    bg = dm_background_pass(raw, var, satmask, fwhm_px, star_image=stars)

    print(
        f'    visit {k}: fwhm {fwhm:.2f}", {nstar} stars, sat frac '
        f'{satmask.mean():.4f}, threshold {bg["threshold"]:.1f}, '
        f'detected {bg["detected_fraction"]:.3f}',
    )

    return dict(
        fwhm=fwhm, delivered=bg['delivered'][cut],
        response=bg['response'][cut],
        stars=stars[cut], sky=sky[cut], raw=raw[cut], var=var[cut],
        satmask=satmask[cut], threshold=bg['threshold'],
        detected_fraction=bg['detected_fraction'], offset=(ox, oy),
    )


def _worker(args):
    """Run simulate_visit on a tuple of its arguments, for the pool."""
    return simulate_visit(*args)


def simulate_coadd(shape, x, y, G, amp, canonical, cfg, seed=0, nproc=1):
    """
    Simulate the visits and their equal-weight coadds.

    Parameters
    ----------
    shape: (ny, nx)
        The patch
    x, y, G, amp: arrays
        The stars (render_stars)
    canonical: WingModel or (r, T)
    cfg: dict
        The settings (DEFAULTS)
    seed: int, optional
    nproc: int, optional
        Visits simulated in parallel

    Returns
    -------
    coadd: dict
        none (coadd of the delivered visits: the trough in),
        response (coadd of the polynomial responses), stars (the
        truth star coadd), sky (the true sky coadd), raw (coadd of
        the raw visits), var, satmask (saturated in any visit),
        visits (table of fwhm, threshold, detected fraction)
    """
    from multiprocessing import Pool

    n = cfg['nvisit']
    jobs = [
        (k, shape, x, y, G, amp, canonical, cfg, seed * 1000 + k)
        for k in range(n)
    ]
    sums = {
        k: np.zeros(shape, dtype='f8')
        for k in ('none', 'response', 'stars', 'sky', 'raw', 'var')
    }

    satmask = np.zeros(shape, dtype=bool)

    rows = []

    if nproc > 1:
        pool = Pool(nproc)
        it = pool.imap(_worker, jobs)
    else:
        it = map(_worker, jobs)

    for v in it:
        sums['none'] += v['delivered']
        sums['response'] += v['response']
        sums['stars'] += v['stars']
        sums['sky'] += v['sky']
        sums['raw'] += v['raw']
        sums['var'] += v['var']
        satmask |= v['satmask']
        rows.append((v['fwhm'], v['threshold'], v['detected_fraction']))

    if nproc > 1:
        pool.close()
        pool.join()

    out = {k: (s / n).astype('f4') for k, s in sums.items()}
    out['var'] = (sums['var'] / n ** 2).astype('f4')
    out['satmask'] = satmask
    out['visits'] = np.array(rows, dtype=[
        ('fwhm', 'f8'), ('threshold', 'f8'), ('detected_fraction', 'f8'),
    ])

    return out
