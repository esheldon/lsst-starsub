"""
forward model of the visit polynomial's response to stars

The stored layer 0 of preliminary_visit_image_background is not
the first-pass fit of calibrateImage: _remeasure_star_background
adds that fit back, builds a detection mask at an adaptive
threshold (stored in the task metadata as adaptive_threshold_value,
in pixel_stdev units on the raw sky image) with footprints grown
by 70 psf sigma, ORs in the first-pass DETECTED plane dilated by
10 px, and refits the raw image with the star_background config
(128 px bins, MEANCLIP, weighted 6x6 Chebyshev; BAD, EDGE,
DETECTED, DETECTED_NEGATIVE, NO_DATA ignored).  SAT, SUSPECT and
SPIKE are added to the ignored planes of the pedestal fits only
(layers 1..), not of this fit: with them ignored the refit
misses the stored fit by 3-4 times more.

The polynomial's response to a star image is the fit of the raw
image minus the fit of the raw image with the star removed, mask
held fixed.  It is not the fit of the star image alone: the
weighted Chebyshev fit takes its bin weights from the scatter of
the pixel values in each bin, so the weights depend on the image
(a smooth star image gets near-zero bin variances and a different
weighting; measured on 2025071900593 detector 90 the star-alone
fit is 15 times too small).  Around the sky noise the fit is
linear to a few tenths of a percent (adding the star gives the
same response as removing it), so the difference is the response.
This module reproduces the mask and the fit, and measures the
response
"""

from ..site import INSTRUMENT

# from CalibrateImageTask._remeasure_star_background
STAR_BG_GROW_SIGMA = 70.0
PSF_DET_DILATE = 10
SIMPLE_PSF_FWHM = 4.0
SIMPLE_PSF_WIDTH = 11
PSF_DET_THRESHOLD = 10.0
PSF_DET_MULTIPLIER = 5.0
DETECTED_PLANES = ['DETECTED', 'DETECTED_NEGATIVE']


def load_raw_exposure(butler, visit, detector):
    """
    Load the raw sky image the star_background fit saw.

    The preliminary visit image (ADU) with its stored background
    added back, plus the stored BackgroundList, the calibration
    (nJy per ADU) and the calibrateImage metadata entries that set
    the fit's mask.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    visit, detector: int

    Returns
    -------
    raw: afw ExposureF
        ADU, sky in
    bglist: lsst.afw.math.BackgroundList
    calib: float
    meta: dict
        adaptive_threshold, psf_threshold, psf_multiplier,
        detected_fraction
    """
    did = dict(
        instrument=INSTRUMENT, visit=int(visit),
        detector=int(detector),
    )

    exp = butler.get('preliminary_visit_image', dataId=did)

    bglist = butler.get('preliminary_visit_image_background', dataId=did)
    calib = float(exp.getPhotoCalib().getCalibrationMean())

    md = butler.get('calibrateImage_metadata', dataId=did).to_dict()
    top = md['calibrateImage']

    # DP2's metadata lacks the psf-detection entries; the config
    # values (psf_detection thresholdValue 10, multiplier 5) were
    # what the weekly recorded on every detector checked

    meta = dict(
        adaptive_threshold=float(top['adaptive_threshold_value']),
        psf_threshold=float(top.get('psf_adaptive_threshold_value',
                                    PSF_DET_THRESHOLD)),
        psf_multiplier=float(top.get(
            'psf_adaptive_include_threshold_multiplier',
            PSF_DET_MULTIPLIER,
        )),
        detected_fraction=float(top.get('detected_mask_fraction',
                                        float('nan'))),
    )

    raw = exp.clone()
    raw.image.array[:, :] += bglist.getImage().array

    return raw, bglist, calib, meta


def _clear_detected(mask):
    """Clear the DETECTED planes of an afw Mask in place."""
    for name in DETECTED_PLANES:
        mask.clearMaskPlane(mask.getMaskPlane(name))


def _dilate_detected(mask, npix):
    """
    Grow the DETECTED planes of an afw Mask in place.

    Parameters
    ----------
    mask: lsst.afw.image.Mask
    npix: int
        The dilation in pixels
    """
    from lsst.afw.geom import SpanSet

    for name in DETECTED_PLANES:
        bit = mask.getPlaneBitMask(name)
        spans = SpanSet.fromMask(mask, bit).dilated(npix)
        spans = spans.clippedTo(mask.getBBox())
        mask.clearMaskPlane(mask.getMaskPlane(name))
        spans.setMask(mask, bit)


def _detect(exposure, threshold, multiplier, grow, clear=True):
    """
    Run SourceDetectionTask on an exposure, setting its DETECTED planes.

    Parameters
    ----------
    exposure: afw ExposureF
    threshold: float
        In pixel_stdev units
    multiplier: float
        The include-threshold multiplier
    grow: float
        Footprint growth in psf sigma
    clear: bool, optional
        Clear the DETECTED planes first

    Returns
    -------
    res: lsst.pipe.base.Struct
        The task's result (sources, numPosPeaks, ...)
    """
    import lsst.afw.table as afwTable
    from lsst.meas.algorithms import (
        SourceDetectionConfig, SourceDetectionTask,
    )

    cfg = SourceDetectionConfig()
    cfg.thresholdType = 'pixel_stdev'
    cfg.thresholdValue = float(threshold)
    cfg.includeThresholdMultiplier = float(multiplier)
    cfg.nSigmaToGrow = float(grow)
    cfg.doTempLocalBackground = False
    cfg.reEstimateBackground = False

    schema = afwTable.SourceTable.makeMinimalSchema()
    task = SourceDetectionTask(config=cfg, schema=schema)
    table = afwTable.SourceTable.make(schema)
    res = task.run(table=table, exposure=exposure, clearMask=clear)

    return res


def reconstruct_fit_mask(raw, prelim, meta):
    """
    Reconstruct the mask the star_background fit saw.

    On a clone of the raw exposure's mask.  The first-pass DETECTED
    plane (psf detection on the background-subtracted image at
    psf_threshold x multiplier, grown 2.4 sigma) is rebuilt on the
    preliminary image and dilated by 10 px; the star-background
    detection at the stored adaptive threshold on the raw sky image
    with footprints grown by 70 sigma is ORed with it.

    Parameters
    ----------
    raw: ExposureF
        Sky image (ADU) from load_raw_exposure; its mask is not
        modified
    prelim: ExposureF
        The preliminary image (background subtracted)
    meta: dict
        From load_raw_exposure

    Returns
    -------
    mask: lsst.afw.image.Mask
    fractions: dict
        detected_fraction, psf_dilated_fraction
    """
    import lsst.afw.image as afwImage

    # first pass: on the background-subtracted image, with the
    # simple Gaussian psf the task had installed at that point
    # (install_simple_psf: fwhm 4 px, width 11)
    from lsst.meas.algorithms import SingleGaussianPsf

    psf_exp = prelim.clone()
    psf_exp.setPsf(SingleGaussianPsf(
        SIMPLE_PSF_WIDTH, SIMPLE_PSF_WIDTH, SIMPLE_PSF_FWHM / 2.3548,
    ))

    _clear_detected(psf_exp.mask)
    _detect(psf_exp, meta['psf_threshold'], meta['psf_multiplier'],
            grow=2.4)
    dilated = psf_exp.mask.clone()
    _dilate_detected(dilated, PSF_DET_DILATE)
    del psf_exp

    # the star-background detection on the raw sky image
    det_exp = raw.clone()
    _clear_detected(det_exp.mask)
    _detect(det_exp, meta['adaptive_threshold'], 1.0,
            grow=STAR_BG_GROW_SIGMA)
    mask = det_exp.mask
    mask |= dilated
    del det_exp

    bad = afwImage.Mask.getPlaneBitMask(['BAD', 'EDGE', 'NO_DATA'])
    det = afwImage.Mask.getPlaneBitMask(DETECTED_PLANES)
    good = (mask.array & bad) == 0
    frac = float(((mask.array & det) != 0)[good].mean())
    dfrac = float(((dilated.array & det) != 0)[good].mean())

    return mask, dict(detected_fraction=frac, psf_dilated_fraction=dfrac)


def star_background_config(stat='MEANCLIP'):
    """
    Build the star_background SubtractBackgroundConfig of calibrateImage.

    Parameters
    ----------
    stat: str, optional
        The statistic; other than MEANCLIP only for linearity tests

    Returns
    -------
    cfg: SubtractBackgroundConfig
    """
    from lsst.meas.algorithms import SubtractBackgroundConfig

    cfg = SubtractBackgroundConfig()
    cfg.statisticsProperty = stat
    cfg.undersampleStyle = 'REDUCE_INTERP_ORDER'
    cfg.binSize = 128
    cfg.algorithm = 'AKIMA_SPLINE'
    cfg.ignoredPixelMask = [
        'BAD', 'EDGE', 'DETECTED', 'DETECTED_NEGATIVE', 'NO_DATA',
    ]
    cfg.isNanSafe = False
    cfg.useApprox = True
    cfg.approxOrderX = 6
    cfg.approxOrderY = -1
    cfg.weighting = True
    return cfg


def fit_star_background(image, mask, variance, stat='MEANCLIP'):
    """
    Run the star_background fit of calibrateImage on an image.

    Parameters
    ----------
    image: ndarray
    mask: afw Mask
        From reconstruct_fit_mask
    variance: ndarray
    stat: str, optional
        See star_background_config

    Returns
    -------
    surface: ndarray
        The rendered 6x6 Chebyshev surface, same units as image
    bglist: lsst.afw.math.BackgroundList
    """
    import lsst.afw.image as afwImage
    from lsst.meas.algorithms import SubtractBackgroundTask

    cfg = star_background_config(stat)
    task = SubtractBackgroundTask(config=cfg)

    mi = afwImage.MaskedImageF(mask.getBBox())
    mi.image.array[:, :] = image
    mi.mask.array[:, :] = mask.array
    mi.variance.array[:, :] = variance
    exp = afwImage.ExposureF(mi)
    bglist = task.run(exposure=exp).background
    surface = bglist[0][0].getImageF().array.copy()

    return surface, bglist


def stored_surface(bglist):
    """
    Render layer 0 of a stored BackgroundList as the fit did.

    Parameters
    ----------
    bglist: lsst.afw.math.BackgroundList

    Returns
    -------
    surface: ndarray
    """
    return bglist[0][0].getImageF().array.copy()


def polynomial_response(raw, mask, star_image):
    """
    Measure the polynomial's response to a star image.

    The fit of the raw image minus the fit of the raw image with
    the star image removed, mask and variance held fixed (see the
    module notes on why the star image is not fit alone).

    Parameters
    ----------
    raw: ExposureF
        From load_raw_exposure
    mask: afw Mask
        From reconstruct_fit_mask
    star_image: ndarray
        Same units as raw (ADU)

    Returns
    -------
    res: dict
        fit_raw, fit_nostar, response (= fit_raw - fit_nostar)
    """
    var = raw.variance.array
    fit_raw, _ = fit_star_background(raw.image.array, mask, var)
    fit_nostar, _ = fit_star_background(
        raw.image.array - star_image, mask, var,
    )
    return dict(
        fit_raw=fit_raw,
        fit_nostar=fit_nostar,
        response=fit_raw - fit_nostar,
    )
