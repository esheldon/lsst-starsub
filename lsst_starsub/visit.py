"""
star and background handling on the visit images

The coadd route (lsst_mdet.starsub.handle_stars) works on images
whose visit-level backgrounds cannot be undone: the coadd inputs
had a 32 px box background subtracted before warping, and that
pass absorbs the bright-star wings.  Restoring the coadd's own
stored 'object' model brings back only what the coadd-level pass
took, and the 2026-09 dual-state stacks showed the restoration
then re-exposes the earlier pass's over-subtraction trough.

On the visit images every layer the pipeline subtracted is
stored and the raw sky image is recoverable exactly:

- preliminary_visit_image_background (== visit_image_background,
  verified identical on the DP2 and w_2026_32 runs): two layers.
  What was actually subtracted is NOT the bin values: layer 0
  is a 6 x 6 Chebyshev polynomial fit to 128 px bin statistics
  and layer 1 an order-0 Chebyshev (a constant) fit to 32 px
  bins.  So the delivered visit image lost wing light only on
  polynomial scales: around a G = 7.6 star the layer-0 surface
  rises by ~8 nJy (0.3 sky sigma) within 500 px over its far
  value, a fair fraction of the wing there
- skyCorr: five layers, the negation of the two above followed
  by the visit-level focal-plane sky model (interpolated 64 px
  bins of a large-scale fit, flat to 0.1 nJy near the same
  star, plus two small sky-frame terms).  The delivered
  visit_image does NOT include it; the warps do (verified:
  visit_image == calib x preliminary image to 1-2 nJy, versus
  3 nJy with skyCorr applied).  So the coadd inputs carry the
  wings essentially intact, and the wing loss seen in the
  coadds has to come from the coadd-level processing

so the star wings and the sky can be characterized together on
the restored image with a background model that stays out of
the wings by construction.  The subtraction machinery is the
coadd one (lsst_mdet.starsub), applied to an adapter presenting
the deep_coadd interface it expects; only the restoration and
the sky model differ.

Units: visit_image is in nJy (photoCalib 1), the preliminary
image and the stored backgrounds in ADU; the backgrounds are
scaled by the preliminary image's calibration mean when
rendered here.  Pixel scale 0.2 arcsec, so the pixel radius
laws of starsub carry over.

Only the butler loaders import the LSST stack; the mask
conversion, the restoration arithmetic and the characterization
are plain numpy
"""
import numpy as np

from lsst_mdet.defaults import DM_INTRP, DM_NO_DATA, DM_SAT
from lsst_mdet.patchfiles import SimpleBox
from lsst_mdet.starsub import (
    APOD_STARS, GSUB, PRE_BW, PRE_GROW, TMPL_OUT_MAX,
    build_star_mask, circle_radius, field_segmentation,
    make_star_table, measure_coadd_fwhm, select_stars,
    subtract_stars, template_out_half,
)

# the weekly reprocessing runs carry the shapelets IQ score in
# visit_detector_table; DP2 itself does not
VISIT_REPO = 'dp2_prep'
VISIT_COLLECTION = 'LSSTCam/runs/DRP/w_2026_32/DM-55677'
INSTRUMENT = 'LSSTCam'
SKYMAP = 'lsst_cells_v2'

# shapelets IQ score tiers (low is good), from the DRP team
IQ_TIERS = [
    ('low', None, 0.002),
    ('medium', 0.002, 0.005),
    ('high', 0.005, 0.02),
    ('very_high', 0.02, None),
]

# afw mask planes mapped onto the coadd mask convention the
# starsub code reads (lsst_mdet.defaults.DM_*).  Planes not
# listed pass through as clear: DETECTED (we segment ourselves),
# SUSPECT (near full well, still star light), SPIKE (diffraction
# spikes: star light we want in the fit, not holes)
MASK_NO_DATA_PLANES = (
    'BAD', 'NO_DATA', 'EDGE', 'VIGNETTED', 'STREAK',
    'CROSSTALK', 'UNMASKEDNAN',
)
MASK_SAT_PLANES = ('SAT',)
MASK_INTRP_PLANES = ('INTRP', 'CR')

# the wide sky pass on the restored image: boxes large enough
# not to follow the wings, and the stars excluded out to their
# template extents so the boxes never see them
WIDE_BW = 256
WIDE_GMAX = 16.0        # template-extent exclusion for G below
WIDE_GROW = 24          # circle margin for the fainter stars

# the joint characterization iterates: wide sky, star
# subtraction, refined sky on the star-free image, star
# amplitudes re-anchored on the refined sky
NROUND = 2


def iq_tier(score):
    """
    the IQ tier name of a shapelets IQ score (nan -> 'unknown')
    """
    if not np.isfinite(score):
        return 'unknown'
    for name, lo, hi in IQ_TIERS:
        if (lo is None or score >= lo) and (hi is None or score < hi):
            return name
    return 'unknown'


def convert_mask(mask_array, plane_dict):
    """
    the afw mask plane image as a coadd-convention (ny, nx, 1)
    mask: DM_NO_DATA for the unusable planes, DM_SAT for
    saturation, DM_INTRP for the interpolated pixels (CR
    included).  Planes absent from plane_dict are ignored

    Parameters
    ----------
    mask_array: array
        The (ny, nx) integer afw mask plane image
    plane_dict: dict
        Plane name -> bit number (afw getMaskPlaneDict)

    Returns
    -------
    (ny, nx, 1) int32 array
    """
    def bits(names):
        val = 0
        for name in names:
            if name in plane_dict:
                val |= 1 << int(plane_dict[name])
        return val

    out = np.zeros(mask_array.shape, dtype='i4')
    for planes, flag in (
        (MASK_NO_DATA_PLANES, DM_NO_DATA),
        (MASK_SAT_PLANES, DM_SAT),
        (MASK_INTRP_PLANES, DM_INTRP),
    ):
        b = bits(planes)
        if b:
            out[(mask_array & b) != 0] |= flag
    return out[:, :, np.newaxis]


class _Plane(object):
    """a plane with the .array attribute the starsub code reads"""
    def __init__(self, arr):
        self.array = arr


class _AfwPsf(object):
    """
    the afw psf presenting the lsst.images compute_kernel_image
    interface used by starsub.measure_coadd_fwhm; positions are
    detector-frame pixels
    """
    def __init__(self, afw_psf):
        self._psf = afw_psf

    def compute_kernel_image(self, x, y):
        import lsst.geom
        kim = self._psf.computeKernelImage(
            lsst.geom.Point2D(float(x), float(y)),
        )
        return _Plane(kim.array.astype('f8'))


class VisitExposure(object):
    """
    one detector of a visit as the deep_coadd-like object the
    starsub code works on: image/variance/mask planes with
    .array, bbox with .x/.y start/stop (detector frame, origin
    0), band, psf, a noise realization drawn from the variance,
    and the stored background layers rendered in nJy in
    .backgrounds, keyed

        initial_coarse  the 128 px layer of the initial model
        initial_fine    the 32 px layer (the wing absorber)
        skycorr         the visit-level sky correction (not in
                        the delivered image; in the warps)

    Built by load_visit_exposure; the plain constructor takes
    arrays so the stack-free tests can build one
    """

    def __init__(self, image, variance, mask, band, backgrounds,
                 psf=None, wcs=None, visit=None, detector=None,
                 calib=1.0, noise=None, rng=None):
        ny, nx = image.shape
        self.image = _Plane(np.ascontiguousarray(image, dtype='f4'))
        self.variance = _Plane(
            np.ascontiguousarray(variance, dtype='f4'),
        )
        self.mask = _Plane(mask)
        self.band = band
        self.backgrounds = backgrounds
        self.psf = psf
        self.wcs = wcs
        self.visit = visit
        self.detector = detector
        self.calib = calib
        self.bbox = SimpleBox(0, nx, 0, ny)

        if noise is None:
            if rng is None:
                rng = np.random.default_rng()
            var = self.variance.array
            sig = np.sqrt(np.where(
                np.isfinite(var) & (var > 0), var, 0.0,
            ))
            noise = (rng.normal(size=var.shape) * sig).astype('f4')
        self.noise_realizations = [_Plane(noise)]
        # what has been restored so far, for the accounting
        self.restored = np.zeros(image.shape, dtype='f4')

    @property
    def sky_level(self):
        """median of the initial model, nJy"""
        bg = self.backgrounds
        return float(np.nanmedian(
            bg['initial_coarse'] + bg['initial_fine'],
        ))

    @property
    def sky_sigma(self):
        """median pixel noise, nJy"""
        var = self.variance.array
        good = np.isfinite(var) & (var > 0)
        return float(np.sqrt(np.median(var[good])))

    @property
    def good(self):
        """usable pixels: finite positive variance, not NO_DATA"""
        var = self.variance.array
        mask0 = self.mask.array[:, :, 0]
        return (
            np.isfinite(var) & (var > 0)
            & ((mask0 & DM_NO_DATA) == 0)
        )


def restore_background(vexp, which='initial'):
    """
    add stored background layers back to the image in place

    Parameters
    ----------
    vexp: VisitExposure
    which: str
        'initial' both layers of the initial model (the raw sky
        image); 'fine' the 32 px layer only (the coarse sky stays
        out, the wings come back); 'none' nothing

    Returns
    -------
    the array added (zeros for 'none')
    """
    bg = vexp.backgrounds
    if which == 'initial':
        add = bg['initial_coarse'] + bg['initial_fine']
    elif which == 'fine':
        add = bg['initial_fine']
    elif which == 'none':
        add = np.zeros(vexp.image.array.shape, dtype='f4')
    else:
        raise ValueError(f'unknown restoration {which!r}')

    vexp.image.array[:, :] += add
    vexp.restored += add
    print(
        f'    restored {which} background: '
        f'median {np.nanmedian(add):.1f} nJy'
    )
    return add


def build_wide_star_mask(stars, shape):
    """
    the sky-pass exclusion: every census star out to its
    template extent when brighter than WIDE_GMAX, else its mask
    circle plus WIDE_GROW.  Bool array of the given shape
    """
    ny, nx = shape
    wide = np.zeros((ny, nx), dtype=bool)
    for st in stars:
        gmag = float(st['G'])
        if gmag < WIDE_GMAX:
            rad = float(min(template_out_half(gmag), TMPL_OUT_MAX))
        else:
            rad = float(circle_radius(gmag)) + WIDE_GROW
        ix = int(round(float(st['x'])))
        iy = int(round(float(st['y'])))
        ir = int(np.ceil(rad))
        y0, y1 = max(0, iy - ir), min(ny, iy + ir + 1)
        x0, x1 = max(0, ix - ir), min(nx, ix + ir + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        ly, lx = np.mgrid[y0:y1, x0:x1]
        wide[y0:y1, x0:x1] |= (
            np.hypot(ly - float(st['y']), lx - float(st['x'])) <= rad
        )
    return wide


def sky_background(vexp, exclude, bw):
    """
    mask-aware sep background of the current image, subtracted
    in place and returned.  Detections (1.5 sigma segmentation)
    and the exclusion zone stay out of the boxes

    Parameters
    ----------
    vexp: VisitExposure
    exclude: bool array
        Pixels kept out of the background boxes
    bw: int
        The box size

    Returns
    -------
    the background array subtracted
    """
    import sep

    image = np.ascontiguousarray(vexp.image.array, dtype='f4')
    good = vexp.good

    # the image may carry the full sky here: segment on a first
    # mask-only flattening, then fit with the detections out
    bkg0 = sep.Background(image, mask=~good | exclude, bw=bw, bh=bw)
    seg = field_segmentation(
        image - bkg0.back(), good, vexp.sky_sigma,
    )

    bad = ~good | (seg > 0) | exclude
    bkg = sep.Background(image, mask=bad, bw=bw, bh=bw)
    back = bkg.back().astype('f4')
    vexp.image.array[:, :] -= back
    print(
        f'    sky background (bw {bw}, {bad.mean():.2f} masked): '
        f'globalback {bkg.globalback:.2f} rms {bkg.globalrms:.2f}'
    )
    return back


def star_model_image(shape, slist):
    """the summed star model of a subtract_stars work list"""
    model = np.zeros(shape, dtype='f4')
    for st in slist:
        if st['A'] > 0:
            model[st['sl']] += st['A'] * st['T']
    return model


def render_canonical_stars(shape, stars, canonical, gsub=GSUB, amps=None,
                           verbose=True):
    """
    the census stars' images (nJy) from the canonical wing, a pure
    prediction: 10^(-0.4 G) T(r) with T in nJy per unit Gaia flux,
    times the per-star amplitudes when given
    """
    from .trough import render_wing_image

    image, n = render_wing_image(
        shape, stars['x'], stars['y'], stars['G'], canonical, 1.0, 1.0,
        gmax=gsub, eps=0.005, amps=amps,
    )
    if verbose:
        print(f'    canonical star model: {n} stars rendered, '
              f'max {image.max():.0f} nJy')
    return image


def handle_stars_visit(vexp, gaia, gsub=GSUB, restore='initial',
                       nround=NROUND, grow_bright=None,
                       star_model='template', canonical=None,
                       joint_spacing=None, joint_prior=None):
    """
    the joint star-wing and sky characterization of one detector

    census and star mask as on the coadds; restoration of the
    stored background layers; then nround rounds of: sky pass
    (wide boxes with the stars excluded to their template
    extents in the first round, PRE_BW boxes with the plain
    star-mask margin after), template build and joint amplitude
    solve (starsub.subtract_stars) on the sky-flattened image.
    Each round after the first re-adds the previous star model
    before the sky pass so the amplitudes are re-anchored on
    the refined sky rather than on their own residuals.  A last
    PRE_BW sky pass on the star-free image finishes.

    The image ends star-subtracted and sky-subtracted in place

    Parameters
    ----------
    vexp: VisitExposure
    gaia: array with fields
        The gaia extract for the detector
    gsub: float, optional
        Census depth
    restore: str, optional
        restore_background choice
    nround: int, optional
    grow_bright: float, optional
        When set, the PRE_BW sky passes (round 2 on, and the final
        one) also exclude this many pixels beyond the masks of the
        stars brighter than RESTORE_GMAX, as the lsst_mdet coadd
        route does (PRE_GROW_BRIGHT); without it the 64 px boxes
        sit 12 px from every mask and follow the bright stars'
        wings and any trough beyond that
    star_model: str, optional
        'template': lsst_mdet's own, built from the image's stamps
        with per-star anchor-ring amplitudes; 'canonical': the
        canonical wing (nJy per unit Gaia flux, from the visit
        templates) rendered as a pure prediction, no per-star
        fit; 'joint': the canonical shape with the amplitudes and
        a bilinear sky mesh solved together (lsst_starsub.joint),
        no rounds: the production model
    canonical: WingModel or (r, T), optional
        The canonical wing (lsst_starsub.wing.read_wing_model)
        for the canonical and joint star models
    joint_spacing: float, optional
        The sky mesh node spacing of the joint fit (default
        lsst_starsub.joint.SPACING)
    joint_prior: float, optional
        The amplitude prior width of the joint fit (default
        lsst_starsub.joint.PRIOR_SIGMA)

    Returns
    -------
    dict with
        stars, star_table, starmask, dstar, slist, fwhm,
        restored (array added back), sky (total sky model
        subtracted, nJy), star_model (the summed star model),
        delivered (the image as delivered, copy)
    """
    from scipy import ndimage
    from lsst_mdet.gaia import gaia_pixel_positions

    delivered = vexp.image.array.copy()
    mask0 = vexp.mask.array[:, :, 0]

    x, y = gaia_pixel_positions(gaia, vexp.wcs, vexp.bbox)
    stars = select_stars(gaia, x, y, mask0, gsub=gsub)
    starmask, comps = build_star_mask(stars, mask0)
    dstar = ndimage.distance_transform_edt(~starmask)
    # the fine-pass exclusion: every mask plus PRE_GROW, and the
    # bright-star masks plus grow_bright when asked for
    fine_excl = dstar < PRE_GROW
    if grow_bright is not None:
        from lsst_mdet.starsub import RESTORE_GMAX
        bright = stars[stars['G'] < RESTORE_GMAX]
        if bright.size > 0:
            bsm, _ = build_star_mask(bright, mask0, verbose=False)
            dbright = ndimage.distance_transform_edt(~bsm)
            fine_excl = fine_excl | (dbright < float(grow_bright))
            print(f'    bright-star exclusion {grow_bright:.0f} px beyond '
                  f'{bright.size} masks: {fine_excl.mean():.3f} of pixels')

    restored = restore_background(vexp, which=restore)
    wide = build_wide_star_mask(stars, mask0.shape)
    print(f'    wide star exclusion fraction {wide.mean():.3f}')

    fwhm = measure_coadd_fwhm(vexp)
    fstr = f'{fwhm:.2f}' if fwhm is not None else 'n/a'
    print(f'    psf fwhm {fstr} arcsec')

    sky = np.zeros(mask0.shape, dtype='f4')
    star_model_img = np.zeros(mask0.shape, dtype='f4')
    slist = []
    if star_model != 'template' and canonical is None:
        raise ValueError(f'star_model {star_model!r} needs the canonical wing')
    if star_model == 'joint':
        from .joint import SPACING, joint_fit
        jf = joint_fit(
            vexp.image.array, vexp.good & ~starmask, stars, canonical,
            vexp.sky_sigma,
            spacing=SPACING if joint_spacing is None else joint_spacing,
            variance=vexp.variance.array,
            **({} if joint_prior is None else dict(prior_sigma=joint_prior)),
        )
        sky += jf['sky']
        star_model_img = jf['star_model']
        vexp.image.array[:, :] -= jf['sky'] + star_model_img
        star_table = make_star_table(stars, [])
        star_table['A'] = jf['A']
        return dict(
            stars=stars, star_table=star_table, starmask=starmask,
            dstar=dstar, slist=[], restored=restored, sky=sky,
            star_model=star_model_img, delivered=delivered, fwhm=fwhm,
            joint=jf,
        )
    for iround in range(nround):
        print(f'  round {iround + 1} of {nround}')
        bw = WIDE_BW if iround == 0 else PRE_BW
        excl = wide if iround == 0 else fine_excl
        if iround > 0:
            # re-anchor: put the stars back, refit the sky
            # without them in the boxes, then refit the stars
            vexp.image.array[:, :] += star_model_img
        sky += sky_background(vexp, exclude=excl, bw=bw)

        if star_model == 'canonical':
            slist = []
            star_model_img = render_canonical_stars(
                mask0.shape, stars, canonical, gsub=gsub,
            )
            vexp.image.array[:, :] -= star_model_img
        else:
            slist = subtract_stars(
                vexp.image.array, vexp.variance.array, mask0,
                gaia, x, y, stars, comps, band=vexp.band, fwhm=fwhm,
            )
            star_model_img = star_model_image(mask0.shape, slist)

    # the refined sky on the star-free image
    sky += sky_background(vexp, exclude=fine_excl, bw=PRE_BW)

    star_table = make_star_table(stars, slist)
    return dict(
        stars=stars,
        star_table=star_table,
        starmask=starmask,
        dstar=dstar,
        slist=slist,
        restored=restored,
        sky=sky,
        star_model=star_model_img,
        delivered=delivered,
        fwhm=fwhm,
    )


def apod_starmask(dstar, width=APOD_STARS):
    """the attenuation zone as used downstream"""
    return dstar < width


# ---------------------------------------------------------------
# butler-facing loaders
# ---------------------------------------------------------------

def make_visit_butler(repo=VISIT_REPO, collection=VISIT_COLLECTION):
    from lsst.daf.butler import Butler
    return Butler(repo, collections=[collection])


def load_iq_scores(butler):
    """
    (visit, detector) -> shapelets IQ score from
    visit_detector_table; empty when the run lacks the column
    """
    t = butler.get(
        'visit_detector_table', dataId=dict(instrument=INSTRUMENT),
    )
    if 'shapeletsIqScore' not in t.colnames:
        print('    no shapeletsIqScore in visit_detector_table')
        return {}
    return {
        (int(v), int(d)): float(s)
        for v, d, s in zip(
            t['visitId'], t['detector'], t['shapeletsIqScore'],
        )
    }


def select_coadd_inputs(butler, tract, patch, band, iq=None):
    """
    the visit-detectors that went into the coadd of a patch
    (deep_coadd_input_summary_tract: the IQ-selected inputs),
    with the shapelets IQ score when a lookup is given, best
    first

    Parameters
    ----------
    butler: Butler
    tract, patch: int
    band: str
    iq: dict, optional
        (visit, detector) -> shapelets IQ score, from
        load_iq_scores

    Returns
    -------
    structured array visit, detector, weight, goodpix, iq_score
    sorted by iq_score (nan last)
    """
    t = butler.get(
        'deep_coadd_input_summary_tract',
        dataId=dict(band=band, skymap=SKYMAP, tract=tract),
    )
    w = np.asarray(t['patch']) == patch
    n = int(w.sum())
    out = np.zeros(n, dtype=[
        ('visit', 'i8'), ('detector', 'i4'),
        ('weight', 'f8'), ('goodpix', 'i8'), ('iq_score', 'f8'),
    ])
    out['visit'] = np.asarray(t['visit'])[w]
    out['detector'] = np.asarray(t['detector'])[w]
    out['weight'] = np.asarray(t['weight'])[w]
    out['goodpix'] = np.asarray(t['goodpix'])[w]
    out['iq_score'] = np.nan
    if iq is not None:
        for i in range(n):
            out['iq_score'][i] = iq.get(
                (int(out['visit'][i]), int(out['detector'][i])),
                np.nan,
            )
    order = np.argsort(np.nan_to_num(out['iq_score'], nan=np.inf))
    out = out[order]
    print(
        f'    {n} coadd inputs for {tract} {patch} {band}, '
        f'{np.isfinite(out["iq_score"]).sum()} with an IQ score'
    )
    return out


def render_background_list(bglist, layers, calib):
    """
    the sum of the selected layers of an afw BackgroundList as
    an f4 array in nJy (the stored layers are ADU), rendered
    exactly as BackgroundList.getImage does: a layer with an
    approximation style set renders from its stored polynomial,
    otherwise by interpolating its bin statistics
    """
    from lsst.afw.math import ApproximateControl

    out = None
    for i in layers:
        bg, interp, undersample, approx, *_ = bglist[i]
        if approx != ApproximateControl.UNKNOWN:
            arr = bg.getImageF().array
        else:
            arr = bg.getImageF(interp, undersample).array
        out = arr.copy() if out is None else out + arr
    return (out * calib).astype('f4')


def describe_background_list(bglist, name):
    """print the layer structure: bins, approximation, order"""
    for i, (bg, interp, undersample, approx, ox, oy, _) in enumerate(
        bglist
    ):
        st = bg.getStatsImage().image.array
        print(
            f'    {name} layer {i}: {st.shape[1]} x {st.shape[0]} '
            f'bins, approx {approx.name} order {ox} x {oy}, '
            f'interp {interp.name}'
        )


def load_visit_exposure(butler, visit, detector, rng=None):
    """
    one detector of a visit from the butler as a VisitExposure:
    the delivered visit_image (nJy) with the stored background
    layers rendered in nJy

    Parameters
    ----------
    butler: Butler
        With the collection set (VISIT_COLLECTION)
    visit, detector: int
    rng: numpy Generator, optional
        For the noise realization
    """
    from lsst_mdet.wcs import ButlerWcs

    did = dict(instrument=INSTRUMENT, visit=int(visit),
               detector=int(detector))
    print(f'    loading visit {visit} detector {detector}')
    # the preliminary image's calibration: ADU -> nJy for the
    # stored (ADU) backgrounds.  The visit_image photoCalib is
    # 1 (already nJy), so the value has to come from the
    # preliminary image
    prelim = butler.get('preliminary_visit_image', dataId=did)
    calib = float(prelim.getPhotoCalib().getCalibrationMean())
    try:
        have_vi = bool(butler.exists('visit_image', did))
    except Exception:
        have_vi = False
    exp = None
    if have_vi:
        exp = butler.get('visit_image', dataId=did)
        if not hasattr(exp.mask, 'getMaskPlaneDict'):
            # an lsst.images VisitImage (DP2 since 2026-09-09):
            # packed mask planes under other names, no getWcs or
            # getPsf; the calibrated preliminary image stands in,
            # as it did when DP2 had no visit_image at all
            print('    visit_image is not an afw exposure; '
                  'calibrating the preliminary image')
            exp = None
    if exp is None:
        # the preliminary image calibrated to nJy.  On the weekly
        # run the two differ by 1.4 nJy rms (0.04 sky sigma), the
        # final calibration's spatial variation
        exp = prelim.clone()
        exp.setMaskedImage(
            prelim.getPhotoCalib().calibrateImage(prelim.getMaskedImage())
        )
    prelim_bg = butler.get(
        'preliminary_visit_image_background', dataId=did,
    )
    skycorr = butler.get('skyCorr', dataId=did)

    if len(prelim_bg) < 2:
        raise RuntimeError(
            f'expected at least 2 initial background layers, '
            f'got {len(prelim_bg)}'
        )
    describe_background_list(prelim_bg, 'initial')
    describe_background_list(skycorr, 'skycorr')
    # layer 0 is the coarse sky; the refinement is usually one
    # layer but some detectors carry extra iterations, summed
    # here into the fine model
    nfine = len(prelim_bg) - 1
    if nfine > 1:
        print(f'    {nfine} refinement layers in the initial model')
    backgrounds = dict(
        initial_coarse=render_background_list(prelim_bg, [0], calib),
        initial_fine=render_background_list(
            prelim_bg, range(1, len(prelim_bg)), calib,
        ),
        skycorr=render_background_list(
            skycorr, range(len(skycorr)), calib,
        ),
    )
    # the per-layer rendering must reproduce what the pipeline
    # subtracted (BackgroundList.getImage)
    for name, bgl, tot in (
        ('initial', prelim_bg,
         backgrounds['initial_coarse'] + backgrounds['initial_fine']),
        ('skycorr', skycorr, backgrounds['skycorr']),
    ):
        ref = bgl.getImage().array * calib
        dmax = float(np.nanmax(np.abs(ref - tot)))
        if dmax > 1.0e-3 * max(1.0, float(np.nanstd(ref))):
            raise RuntimeError(
                f'{name} layer rendering differs from getImage '
                f'by up to {dmax:.3e} nJy'
            )
    mask = convert_mask(exp.mask.array, exp.mask.getMaskPlaneDict())

    if rng is None:
        rng = np.random.default_rng(
            [int(visit) % (2 ** 32), int(detector)],
        )

    vexp = VisitExposure(
        image=exp.image.array,
        variance=exp.variance.array,
        mask=mask,
        band=exp.getFilter().bandLabel,
        backgrounds=backgrounds,
        psf=_AfwPsf(exp.getPsf()),
        wcs=ButlerWcs(exp.getWcs()),
        visit=int(visit),
        detector=int(detector),
        calib=calib,
        rng=rng,
    )
    vexp.exposure = exp
    print(
        f'    band {vexp.band}, sky {vexp.sky_level:.0f} nJy, '
        f'sigma {vexp.sky_sigma:.1f} nJy, calib {calib:.4f}'
    )
    return vexp


def load_gaia_for_exposure(vexp, gaia_file=None, gmax=None):
    """
    the gaia extract covering the detector, from a file
    (lsst_mdet.gaia.read_gaia_file layout) or the TAP query
    """
    from lsst_mdet.gaia import GMAX, fetch_gaia, read_gaia_file

    if gmax is None:
        gmax = GMAX
    if gaia_file is not None:
        return read_gaia_file(gaia_file, vexp.wcs, vexp.bbox, gmax=gmax)
    return fetch_gaia(vexp.wcs, vexp.bbox, gmax=gmax)
