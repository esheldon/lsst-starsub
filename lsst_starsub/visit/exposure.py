"""
star and background handling on the visit images

The coadd route (lsst_starsub.stamps.handle_stars) works on images
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
coadd one (lsst_starsub.stamps), applied to an adapter presenting
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

from ..census import (
    GSUB,
    build_star_mask,
    circle_radius,
    field_segmentation,
    make_star_table,
    patch_census,
)
from ..geom import SimpleBox
from ..maskbits import DM_INTRP, DM_NO_DATA, DM_SAT
from ..site import INSTRUMENT, SKYMAP, VISIT_COLLECTION, VISIT_REPO
from ..stamps import (
    PRE_BW,
    PRE_GROW,
    TMPL_OUT_MAX,
    measure_coadd_fwhm,
    subtract_stars,
    template_out_half,
)
from ..wing import render_canonical_stars


# shapelets IQ score tiers (low is good), from the DRP team
IQ_TIERS = [
    ('low', None, 0.002),
    ('medium', 0.002, 0.005),
    ('high', 0.005, 0.02),
    ('very_high', 0.02, None),
]

# afw mask planes mapped onto the coadd mask convention the
# starsub code reads (lsst_starsub.maskbits.DM_*).  Planes not
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
# pixels whose variance is below this fraction of the detector median
# saw no light (a dead amplifier) and are flagged NO_DATA
DEAD_VAR_FRAC = 0.25
# mm: detectors whose center lies beyond this focal-plane radius are
# left out of the visit scheme for now (visit_detectors).  The wings
# there differ: on visit 2025060400354 the bright stars' wing-fit
# amplitude against the focal-plane-average wing is 1.09-1.11 inside
# 300 mm and 1.21 beyond, with the core scale 3 percent lower, the
# vignetted pupil changing the scattering (TODO step 9).  A field-radius
# term in the wing is the eventual fix; the field edge is ~320 mm
MAX_FIELD_RADIUS = 300.0
WIDE_BW = 256
WIDE_GMAX = 16.0        # template-extent exclusion for G below
WIDE_GROW = 24          # circle margin for the fainter stars
# core amplitudes outside this range (a neighbor in the aperture, a
# star off its Gaia position) are left to the fit
CORE_AMP_RANGE = (0.25, 4.0)
# the detector's own core profile for the core amplitudes: the mean of
# the unsaturated stars' stamps in this G range, each per unit Gaia
# flux, out to CORE_STACK_HALF px, scaled to the wing's flux within
# CORE_NORM_RAD px; at least CORE_STACK_MIN stars, else the wing given
# is used.  The seeing varies across a visit (1.02-1.30 arcsec over
# visit 2025060400354) and a 5 px aperture's enclosed fraction with it
# (the stars' flux over the wing's within 5 px fell from 0.88 to 0.81
# across that range), so against the visit's pooled stack the
# per-detector core scale tracked the local fwhm (correlation -0.85,
# -0.32 per arcsec).  Within 12 px the ratio is flat to 2 percent, and
# what remains is per-detector structure seen at every radius; so the
# stack supplies the core shape at the detector's seeing, and its zero
# point is the wing's at 12 px, which keeps the amplitudes relative to
# the wing they render (a stack normalized to the detector's median
# star rendered the wing 15 percent too bright on that visit)
CORE_STACK_GMIN = 15.5
CORE_STACK_GMAX = 19.0
CORE_STACK_HALF = 12
CORE_STACK_MIN = 20
CORE_NORM_RAD = 12.0
# the stack is a per-pixel mean with CORE_STACK_CLIP sigma clipping
# (CORE_STACK_NCLIP rounds, sigma from the MAD over the stars), and
# stars with a Gaia neighbor within CORE_STACK_ISOLATION px are left
# out of it: a plain mean is thrown by the neighbors and galaxies in the
# stamps (the stack's 5/12 px flux ratio scattered 2 and 4 percent
# across the detectors of two visits after its seeing trend, 0.5 and 1.1
# with the clipping, a further 0.1 with the isolation)
CORE_STACK_CLIP = 3.0
CORE_STACK_NCLIP = 2
CORE_STACK_ISOLATION = 24.0
# px: a star with another Gaia star closer than this is not measured
# from its core (the aperture would hold both) and keeps the fit
CORE_NEIGHBOR = 12.0

# the joint characterization iterates: wide sky, star
# subtraction, refined sky on the star-free image, star
# amplitudes re-anchored on the refined sky
NROUND = 2


def iq_tier(score):
    """
    Name the IQ tier of a shapelets IQ score.

    Parameters
    ----------
    score: float
        The shapelets IQ score; nan gives 'unknown'

    Returns
    -------
    name: str
        From IQ_TIERS, or 'unknown'
    """
    if not np.isfinite(score):
        return 'unknown'
    for name, lo, hi in IQ_TIERS:
        if (lo is None or score >= lo) and (hi is None or score < hi):
            return name
    return 'unknown'


def flag_dead_pixels(mask, variance, frac=None):
    """
    Flag the pixels that saw no light as NO_DATA, in place.

    A dead amplifier is not flagged by the pipeline (visit
    2025060400354 detector 001: a 512 x 2048 px block at -34 ADU with
    a variance 4 percent of its neighbors', PARTLY_VIGNETTED and
    nothing else), and the sky fit then chases a 1000 nJy hole.  The
    variance tells: pixels below frac times the detector's median
    variance had no sky in them.

    Parameters
    ----------
    mask: array (ny, nx, nplane)
        The converted mask, plane 0 modified
    variance: array
        The variance plane
    frac: float, optional
        Default DEAD_VAR_FRAC

    Returns
    -------
    ndead: int
        The pixels flagged
    """
    if frac is None:
        frac = DEAD_VAR_FRAC
    ok = np.isfinite(variance) & (variance > 0)
    if not ok.any():
        return 0
    dead = ok & (variance < frac * np.median(variance[ok]))
    ndead = int(dead.sum())
    if ndead > 0:
        mask[:, :, 0][dead] |= DM_NO_DATA
        print(f'    {ndead} pixels ({100 * ndead / dead.size:.1f} percent) '
              f'with variance below {frac:g} x the median flagged NO_DATA')
    return ndead


def convert_mask(mask_array, plane_dict):
    """
    Convert an afw mask plane image to the coadd convention.

    DM_NO_DATA for the unusable planes, DM_SAT for saturation,
    DM_INTRP for the interpolated pixels (CR included).  Planes
    absent from plane_dict are ignored.

    Parameters
    ----------
    mask_array: array
        The (ny, nx) integer afw mask plane image
    plane_dict: dict
        Plane name -> bit number (afw getMaskPlaneDict)

    Returns
    -------
    mask: (ny, nx, 1) int32 array
    """
    def bits(names):
        """The bit mask of the named planes present in plane_dict."""
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
    """A plane with the .array attribute the starsub code reads."""
    def __init__(self, arr):
        self.array = arr


class _AfwPsf(object):
    """
    An afw psf presenting the lsst.images compute_kernel_image interface.

    As stamps.measure_coadd_fwhm uses it; positions are detector-frame
    pixels.

    Parameters
    ----------
    afw_psf: lsst.afw.detection.Psf
    """
    def __init__(self, afw_psf):
        self._psf = afw_psf

    def compute_kernel_image(self, x, y):
        """
        Evaluate the psf kernel image at a position.

        Parameters
        ----------
        x, y: float
            Detector-frame pixel position

        Returns
        -------
        image: _Plane
            The kernel image as an object with .array (f8)
        """
        import lsst.geom
        kim = self._psf.computeKernelImage(
            lsst.geom.Point2D(float(x), float(y)),
        )
        return _Plane(kim.array.astype('f8'))


class VisitExposure(object):
    """
    One detector of a visit as the deep_coadd-like object the code works on.

    Image/variance/mask planes with .array, bbox with .x/.y
    start/stop (detector frame, origin 0), band, psf, a noise
    realization drawn from the variance, and the stored background
    layers rendered in nJy in .backgrounds, keyed

        initial_coarse  the 128 px layer of the initial model
        initial_fine    the 32 px layer (the wing absorber)
        skycorr         the visit-level sky correction (not in
                        the delivered image; in the warps)

    Built by load_visit_exposure; the plain constructor takes arrays
    so the stack-free tests can build one.

    Parameters
    ----------
    image, variance: (ny, nx) arrays
        Stored as f4
    mask: (ny, nx, 1) int array
        In the coadd convention (convert_mask)
    band: str
    backgrounds: dict
        The stored layers rendered in nJy, keyed as above
    psf: object, optional
        With compute_kernel_image(x, y) (_AfwPsf)
    wcs: ButlerWcs, optional
    visit, detector: int, optional
    calib: float, optional
        nJy per ADU of the preliminary image
    noise: (ny, nx) array, optional
        The noise realization; drawn from the variance when None
    rng: numpy Generator, optional
        For the noise draw
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
        """The median of the initial background model, nJy."""
        bg = self.backgrounds
        return float(np.nanmedian(
            bg['initial_coarse'] + bg['initial_fine'],
        ))

    @property
    def sky_sigma(self):
        """The median pixel noise, nJy."""
        var = self.variance.array
        good = np.isfinite(var) & (var > 0)
        return float(np.sqrt(np.median(var[good])))

    @property
    def good(self):
        """The usable pixels: finite positive variance, not NO_DATA."""
        var = self.variance.array
        mask0 = self.mask.array[:, :, 0]

        return (
            np.isfinite(var) & (var > 0)
            & ((mask0 & DM_NO_DATA) == 0)
        )


def restore_background(vexp, which='initial'):
    """
    Add stored background layers back to the image in place.

    Parameters
    ----------
    vexp: VisitExposure
    which: str, optional
        'initial' both layers of the initial model (the raw sky
        image); 'fine' the 32 px layer only (the coarse sky stays
        out, the wings come back); 'none' nothing

    Returns
    -------
    added: array
        The array added (zeros for 'none')
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
    Build the sky-pass exclusion mask.

    Every census star out to its template extent when brighter than
    WIDE_GMAX, else its mask circle plus WIDE_GROW.

    Parameters
    ----------
    stars: structured array
        The census
    shape: (ny, nx)

    Returns
    -------
    wide: bool array
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


def boxes_at(boxes, bw, x, y):
    """
    Evaluate a box-grid background at positions.

    Bilinear between the box centers at ((i + 0.5) bw, (j + 0.5) bw),
    clamped beyond the outermost centers: the evaluation
    box_background uses, so the product's sky is reproduced exactly
    from the stored boxes (lsst_starsub.visit.product).

    Parameters
    ----------
    boxes: array (my, mx)
        The box values, every box finite
    bw: int
        The box size in px
    x, y: arrays
        Positions in px, the pixel centers at integers

    Returns
    -------
    values: array, x's shape
    """
    from scipy import ndimage

    x = np.asarray(x, dtype='f8')
    coords = np.array([np.asarray(y, dtype='f8').ravel() / bw - 0.5,
                       x.ravel() / bw - 0.5])
    out = ndimage.map_coordinates(
        np.asarray(boxes, dtype='f8'), coords, order=1, mode='nearest',
    )
    return out.reshape(x.shape)


def box_background(image, usable, bw, min_frac=0.1, return_boxes=False):
    """
    A box-median background that cannot overshoot in masked regions.

    The median per bw x bw box of the usable pixels, boxes with fewer
    than min_frac of their pixels usable filled from the nearest box
    that has them, a 3 x 3 median filter over the boxes, and bilinear
    interpolation between the box centers (clamped at the edges,
    boxes_at).  sep's spline over the boxes overshot by 60 nJy in a
    detector corner that the wide star exclusion of a G 9.6 star
    covered; the joint fit's segmentation then masked the plateau as a
    source, and the hole stayed in the product.

    Parameters
    ----------
    image: array
    usable: bool array
        Pixels that count
    bw: int
        The box size in px
    min_frac: float, optional
        A box needs this fraction of usable pixels to count
    return_boxes: bool, optional
        Also return the box grid the background is interpolated from

    Returns
    -------
    back: array (f4)
        The background, the image's shape
    boxes: array (ny // bw, nx // bw)
        With return_boxes
    """
    from scipy import ndimage
    from .profiles import box_medians

    ny, nx = image.shape
    med = box_medians(image, usable, box=bw, min_frac=min_frac)
    empty = ~np.isfinite(med)
    if empty.all():
        raise ValueError('no usable box for the background')
    if empty.any():
        _, (iy, ix) = ndimage.distance_transform_edt(
            empty, return_indices=True,
        )
        med = med[iy, ix]
    med = ndimage.median_filter(med, size=3, mode='nearest')
    # box centers at (i + 0.5) bw; pixels beyond the last center are
    # clamped to it, and the partial boxes at the far edges (dropped
    # by box_medians) take the last full box's value
    yy, xx = np.mgrid[0:ny, 0:nx]
    back = boxes_at(med, bw, xx, yy).astype('f4')
    if return_boxes:
        return back, med
    return back


def sky_background(vexp, exclude, bw):
    """
    Subtract a mask-aware box background from the image in place.

    Detections (1.5 sigma segmentation) and the exclusion zone stay
    out of the boxes (box_background).

    Parameters
    ----------
    vexp: VisitExposure
    exclude: bool array
        Pixels kept out of the background boxes
    bw: int
        The box size

    Returns
    -------
    back: array
        The background subtracted
    """
    image = vexp.image.array
    good = vexp.good

    # the image may carry the full sky here: segment on a first
    # mask-only flattening, then fit with the detections out
    back0 = box_background(image, good & ~exclude, bw)
    seg = field_segmentation(image - back0, good, vexp.sky_sigma)

    bad = ~good | (seg > 0) | exclude
    back, boxes = box_background(image, ~bad, bw, return_boxes=True)
    vexp.image.array[:, :] -= back
    # the box grid of the last pass, for the product (the joint route
    # runs one pass before the fit; its boxes plus the mesh nodes
    # reproduce the sky model exactly)
    vexp.sky_boxes = boxes
    vexp.sky_bw = int(bw)

    print(
        f'    sky background (bw {bw}, {bad.mean():.2f} masked): '
        f'median {np.median(back):.2f}, range {back.min():.1f} to '
        f'{back.max():.1f}'
    )

    return back


def star_model_image(shape, slist):
    """
    Sum the star model of a subtract_stars work list.

    Parameters
    ----------
    shape: (ny, nx)
    slist: list of dict
        The work list, with sl, A and T per star

    Returns
    -------
    model: array
    """
    model = np.zeros(shape, dtype='f4')
    for st in slist:
        if st['A'] > 0:
            model[st['sl']] += st['A'] * st['T']
    return model


def handle_stars_visit(
    vexp, gaia, gsub=GSUB, restore='initial',
    nround=NROUND, grow_bright=None,
    star_model='template', canonical=None,
    joint_spacing=None, joint_prior=None,
    edge_factor=None, amplitudes=None, core_rap=None,
):
    """
    Run the joint star-wing and sky characterization of one detector.

    Census and star mask as on the coadds; restoration of the
    stored background layers; then nround rounds of: sky pass
    (wide boxes with the stars excluded to their template
    extents in the first round, PRE_BW boxes with the plain
    star-mask margin after), template build and joint amplitude
    solve (stamps.subtract_stars) on the sky-flattened image.
    Each round after the first re-adds the previous star model
    before the sky pass so the amplitudes are re-anchored on
    the refined sky rather than on their own residuals.  A last
    PRE_BW sky pass on the star-free image finishes.

    The image ends star-subtracted and sky-subtracted in place.

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
        stars brighter than RESTORE_GMAX, as the stamp-template
        coadd route does (PRE_GROW_BRIGHT); without it the 64 px boxes
        sit 12 px from every mask and follow the bright stars'
        wings and any trough beyond that
    star_model: str, optional
        'template': the stamp route's, built from the image's stamps
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
    edge_factor: float, optional
        Joint model only: the stars across the detector edge are
        in the census and fit as well, those brighter than
        lsst_starsub.joint.GFIT whose center is within edge_factor
        mask radii (circle_radius) of the image; default only the
        stars on the image are fit and the intruders follow the
        census default (bright, within STAR_MARGIN px)
    amplitudes: structured array, optional
        Joint model only: per-star amplitudes (ra, dec, A) from a
        visit-level consolidation (lsst_starsub.visit.gather); every
        census star is pinned to its value (1 where absent) and only
        the sky is fit, the second pass of the visit scheme
    core_rap: float, optional
        Joint model only: the unsaturated on-image census stars get
        their amplitude from the flux within this many px of their
        center (lsst_starsub.wing.core_amplitudes), measured on the
        flattened image with the wing given, which must then be the
        visit's own (lsst_starsub.visit.trough.visit_wing) for the
        core to be right; those stars are pinned to it and only the
        saturated and edge stars stay free.  Their table rows carry
        free = 2

    Returns
    -------
    res: dict
        stars, star_table, starmask, dstar, slist, fwhm,
        restored (array added back), sky (total sky model
        subtracted, nJy), star_model (the summed star model),
        delivered (the image as delivered, copy); with the joint
        model also joint (the joint_fit result, with A_err and free)
    """
    from scipy import ndimage

    delivered = vexp.image.array.copy()
    mask0 = vexp.mask.array[:, :, 0]
    intruders = {}
    if edge_factor is not None:
        from ..joint import GFIT
        intruders = dict(
            intruder_gmax=GFIT,
            intruder_margin=lambda gmag: edge_factor * circle_radius(gmag),
        )
    stars, starmask, comps, dstar, x, y = patch_census(
        gaia, vexp.wcs, vexp.bbox, mask0, gsub=gsub, **intruders,
    )

    # the fine-pass exclusion: every mask plus PRE_GROW, and the
    # bright-star masks plus grow_bright when asked for
    fine_excl = dstar < PRE_GROW

    if grow_bright is not None:
        from ..stamps import RESTORE_GMAX
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
        from ..joint import GFIT, SPACING, joint_fit
        # the restored image carries the whole sky; the joint fit
        # segments the sources on the image minus the star model
        # without its own first-pass sky (see lsst_starsub.joint), so
        # the coarse wide-box sky pass flattens it first, as on the
        # coadds the object background does.  The mesh then models
        # what the boxes left, jointly with the stars
        sky += sky_background(vexp, exclude=wide, bw=WIDE_BW)
        extra = dict(gfit=GFIT)
        if joint_prior is not None:
            extra['prior_sigma'] = joint_prior
        if edge_factor is not None:
            extra['free_margin'] = (
                edge_factor * circle_radius(stars['G'].astype('f8'))
            )
        core_err = None
        if core_rap is not None and amplitudes is None:
            core_amps, core_err, core_ok = core_pinned(
                vexp, stars, canonical, core_rap, edge_factor=edge_factor,
                neighbors=(x, y),
            )
            extra['amps'] = core_amps
            extra['free'] = core_free(
                stars, core_ok, vexp.image.array.shape, edge_factor,
            )
        if amplitudes is not None:
            # the second pass: every star at its consolidated
            # amplitude and disk, the sky alone refit
            extra['amps'] = match_amplitudes(stars, amplitudes)
            extra['gfit'] = -np.inf
            extra['fit_disks'] = False
            if 'D' in amplitudes.dtype.names:
                extra['disks'] = match_amplitudes(stars, amplitudes,
                                                  column='D')
        jf = joint_fit(
            vexp.image.array, vexp.good & ~starmask, stars, canonical,
            vexp.sky_sigma,
            spacing=SPACING if joint_spacing is None else joint_spacing,
            variance=vexp.variance.array, band=vexp.band,
            **extra,
        )
        sky += jf['sky']
        star_model_img = jf['star_model']
        vexp.image.array[:, :] -= jf['sky'] + star_model_img
        star_table = make_star_table(stars, [])
        star_table['A'] = jf['A']
        if core_err is not None:
            # the core-measured stars are constrained, not free: free
            # = 2 in the table, their errors from the aperture
            jf['A_err'] = np.where(core_ok, core_err, jf['A_err'])
            jf['free'] = np.where(core_ok, 2, jf['free']).astype('i2')
        return dict(
            stars=stars, star_table=star_table, starmask=starmask,
            dstar=dstar, slist=[], restored=restored, sky=sky,
            star_model=star_model_img, delivered=delivered, fwhm=fwhm,
            joint=jf, sky_boxes=(vexp.sky_boxes, vexp.sky_bw),
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


# ---------------------------------------------------------------
# butler-facing loaders
# ---------------------------------------------------------------

def core_pixels(vexp):
    """
    The pixels a core measurement may use.

    The usable pixels less the saturated and interpolated ones: an
    interpolated core (a saturated star the census did not flag, a
    cosmic ray) sums to a fraction of the star's flux.

    Parameters
    ----------
    vexp: VisitExposure

    Returns
    -------
    good: bool array
    """
    mask0 = vexp.mask.array[:, :, 0]
    return vexp.good & ((mask0 & (DM_SAT | DM_INTRP)) == 0)


def clipped_mean(stamps, nsig=None, niter=None):
    """
    The per-pixel sigma-clipped mean of a set of stamps.

    Parameters
    ----------
    stamps: array (nstamp, ny, nx)
        nan where not measured
    nsig: float, optional
        The clip in units of 1.4826 x the MAD about the median over
        the stamps; default CORE_STACK_CLIP
    niter: int, optional
        Rounds of clipping; default CORE_STACK_NCLIP

    Returns
    -------
    mean: array (ny, nx)
    """
    nsig = CORE_STACK_CLIP if nsig is None else nsig
    niter = CORE_STACK_NCLIP if niter is None else niter
    keep = np.isfinite(stamps)
    for _ in range(niter):
        kept = np.where(keep, stamps, np.nan)
        med = np.nanmedian(kept, axis=0)
        sig = 1.4826 * np.nanmedian(np.abs(kept - med), axis=0) + 1e-30
        keep = np.isfinite(stamps) & (np.abs(stamps - med) < nsig * sig)
    return np.nanmean(np.where(keep, stamps, np.nan), axis=0)


def detector_core_stack(vexp, stars, wing, gmin=None, gmax=None, half=None,
                        norm_rad=None, exclude=None):
    """
    Measure the detector's own core shape from its unsaturated stars.

    The clipped mean over the stars of their stamps per unit Gaia
    flux (clipped_mean), cut on the integer pixel nearest each star
    (no resampling; the pixel phases average out over the stars, as
    they do in an aperture sum; a per-pixel median narrows a pattern
    that shifts from star to star, and a plain mean is thrown by the
    neighbors and galaxies in the stamps), scaled so that its flux
    within norm_rad px is the wing's.  The core at this detector's
    seeing with the wing's zero point, in nJy per unit Gaia flux: an
    amplitude measured against it renders the wing with the right flux
    at norm_rad and beyond, whatever the local seeing did to the
    enclosed fraction inside.

    Parameters
    ----------
    vexp: VisitExposure
        Flattened (the sky subtracted)
    stars: structured array
        The census
    wing: WingModel or (r, T)
        The wing the amplitudes render, nJy per unit Gaia flux
    gmin, gmax: float, optional
        The stars used, unsaturated and on the image; default
        CORE_STACK_GMIN, CORE_STACK_GMAX
    half: int, optional
        The stamp half size in px; default CORE_STACK_HALF
    norm_rad: float, optional
        The radius the stack is matched to the wing within, at most
        half; default CORE_NORM_RAD
    exclude: bool array, optional
        Per census star, left out of the stack (a neighbor too close)

    Returns
    -------
    stack: array (2 half + 1, 2 half + 1) or None
        None with fewer than CORE_STACK_MIN stars
    nstar: int
        The stars stacked
    """
    from ..wing import profile_of

    gmin = CORE_STACK_GMIN if gmin is None else gmin
    gmax = CORE_STACK_GMAX if gmax is None else gmax
    half = CORE_STACK_HALF if half is None else half
    norm_rad = CORE_NORM_RAD if norm_rad is None else norm_rad
    if norm_rad > half:
        raise ValueError(f'norm_rad {norm_rad} exceeds the stamp half {half}')

    image = vexp.image.array
    good = core_pixels(vexp)
    ny, nx = image.shape
    sel = ((stars['on_image'] == 1) & (stars['is_sat'] == 0)
           & (stars['G'] >= gmin) & (stars['G'] < gmax))
    if exclude is not None:
        sel &= ~exclude
    gy, gx = np.mgrid[-half:half + 1, -half:half + 1]
    rr = np.hypot(gy, gx)
    ap = rr <= norm_rad
    stamps = []
    for st in stars[sel]:
        icx, icy = int(round(float(st['x']))), int(round(float(st['y'])))
        if icx - half < 0 or icy - half < 0 or icx + half >= nx \
                or icy + half >= ny:
            continue
        cut = np.s_[icy - half:icy + half + 1, icx - half:icx + half + 1]
        if not good[cut][ap].all():
            continue
        stamp = image[cut].astype('f8')
        stamp[~good[cut]] = np.nan
        stamp /= 10.0 ** (-0.4 * float(st['G']))
        stamps.append(stamp)
    nstar = len(stamps)
    if nstar < CORE_STACK_MIN:
        return None, nstar
    stack = clipped_mean(np.array(stamps))
    r, T = profile_of(wing, 0.5 * (gmin + gmax))
    wsum = float(np.interp(rr[ap], r, T).sum())
    stack *= wsum / stack[ap].sum()
    return stack, nstar


def core_pinned(vexp, stars, wing, rap, edge_factor=None, neighbors=None):
    """
    Measure the unsaturated on-image stars' amplitudes from their cores.

    On the flattened image (the wide-box sky pass done), each such
    star's flux within rap px over the wing's at amplitude 1, with the
    sky noise over the aperture as the error.

    Parameters
    ----------
    vexp: VisitExposure
        Flattened
    stars: structured array
        The census
    wing: WingModel
        The wing the amplitudes render: the detector's own core stack
        (detector_core_stack) takes its zero point from it, and it is
        used as is when the detector has too few stars for a stack
    rap: float
        The aperture radius in px
    edge_factor: float, optional
        Unused here; the edge stars are off the image and keep their
        fit
    neighbors: (x, y) arrays, optional
        The positions of every Gaia star near the image; a census
        star with one closer than CORE_NEIGHBOR px is not measured

    Returns
    -------
    amps: array
        Per census star, the core amplitude, 1 where not measured
    errs: array
        Their errors, nan where not measured
    ok: bool array
        Measured
    """
    from ..wing import core_amplitudes

    sel = (stars['on_image'] == 1) & (stars['is_sat'] == 0)
    crowded = None
    if neighbors is not None:
        from scipy.spatial import cKDTree

        nx_, ny_ = neighbors
        tree = cKDTree(np.c_[nx_, ny_])
        xy = np.c_[stars['x'], stars['y']]
        # the star itself is in its own list
        blended = np.array([
            len(p) > 1 for p in tree.query_ball_point(xy, CORE_NEIGHBOR)
        ])
        crowded = np.array([
            len(p) > 1
            for p in tree.query_ball_point(xy, CORE_STACK_ISOLATION)
        ])
        nblend = int((sel & blended).sum())
        sel &= ~blended
        if nblend:
            print(f'    {nblend} stars with a neighbor within '
                  f'{CORE_NEIGHBOR:g} px left to the fit')
    amps = np.ones(stars.size)
    errs = np.full(stars.size, np.nan)
    ok = np.zeros(stars.size, dtype=bool)
    # the detector's own core shape with the wing's zero point, so the
    # local seeing is in the model and not in the amplitudes
    core, nstack = detector_core_stack(vexp, stars, wing, exclude=crowded)
    if core is None:
        print(f'    core stack: only {nstack} stars, using the wing '
              'given')
        core = wing
    else:
        print(f'    core stack from {nstack} stars')
    if sel.any():
        a, e, k = core_amplitudes(
            vexp.image.array, core_pixels(vexp), stars['x'][sel],
            stars['y'][sel],
            stars['G'][sel].astype('f8'), core, rap,
            amp_range=CORE_AMP_RANGE, sky_sigma=vexp.sky_sigma,
        )
        amps[sel], errs[sel], ok[sel] = a, e, k
    print(f'    core amplitudes ({rap:g} px): {int(ok.sum())} of '
          f'{int(sel.sum())} unsaturated stars measured, median '
          f'{np.median(amps[ok]) if ok.any() else np.nan:.3f}')
    return amps, errs, ok


def core_free(stars, core_ok, shape, edge_factor):
    """
    Choose the stars the joint fit still fits after the core pinning.

    The saturated stars on the image, and with an edge factor the
    stars across the edge within edge_factor mask radii, brighter
    than lsst_starsub.joint.GFIT; the core-measured stars are pinned.

    Parameters
    ----------
    stars: structured array
        The census
    core_ok: bool array
        The core-measured stars
    shape: (ny, nx)
    edge_factor: float or None

    Returns
    -------
    free: bool array
    """
    from ..joint import GFIT

    ny, nx = shape
    x, y, G = stars['x'], stars['y'], stars['G'].astype('f8')
    m = np.zeros(stars.size)
    if edge_factor is not None:
        m = edge_factor * circle_radius(G)
    near = (x >= -m) & (x < nx + m) & (y >= -m) & (y < ny + m)
    free = near & (G < GFIT) & ~core_ok
    return free


def star_key(ra, dec):
    """
    Key stars by position, to match census rows across detectors.

    The census carries no source id; the same Gaia extract gives the
    same positions on every detector, so the position rounded to a
    micro-degree identifies the star.

    Parameters
    ----------
    ra, dec: arrays
        Degrees

    Returns
    -------
    keys: list of (int, int)
    """
    ira = np.round(np.asarray(ra, dtype='f8') * 1e6).astype('i8')
    idec = np.round(np.asarray(dec, dtype='f8') * 1e6).astype('i8')
    return list(zip(ira.tolist(), idec.tolist()))


def match_amplitudes(stars, amplitudes, column='A', default=1.0):
    """
    Look up each census star's consolidated amplitude.

    Parameters
    ----------
    stars: structured array
        The census, with ra and dec
    amplitudes: structured array
        With ra, dec and A

    Returns
    -------
    amps: array
        One per census star, 1 (the prediction) where the star is
        not in the table
    """
    table = dict(zip(star_key(amplitudes['ra'], amplitudes['dec']),
                     np.asarray(amplitudes[column], dtype='f8').tolist()))
    keys = star_key(stars['ra'], stars['dec'])
    amps = np.array([table.get(k, default) for k in keys], dtype='f8')
    nfound = sum(k in table for k in keys)
    print(f'    {column} for {nfound} of {stars.size} census stars '
          f'from the consolidated table')
    return amps


def detector_field_radius(butler, detector):
    """
    Get a detector's distance from the focal-plane center.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    detector: int

    Returns
    -------
    radius: float
        mm
    """
    from lsst.afw.cameraGeom import FOCAL_PLANE

    camera = butler.get('camera', instrument=INSTRUMENT)
    c = camera[int(detector)].getCenter(FOCAL_PLANE)
    return float(np.hypot(c.getX(), c.getY()))


def visit_detectors(butler, visit, max_radius=None):
    """
    List the detectors of a visit with a wcs and a calibration.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    visit: int
    max_radius: float, optional
        mm: leave out the detectors whose center lies beyond this
        focal-plane radius (the heavily vignetted ones); None keeps
        all, MAX_FIELD_RADIUS is the visit scheme's cut

    Returns
    -------
    detectors: list of int
        Sorted
    """
    cat = butler.get(
        'visit_summary', dataId=dict(instrument=INSTRUMENT, visit=int(visit)),
    )
    dets = sorted(
        int(rec['id']) for rec in cat
        if rec.getWcs() is not None and rec.getPhotoCalib() is not None
    )
    if max_radius is not None:
        from lsst.afw.cameraGeom import FOCAL_PLANE

        camera = butler.get('camera', instrument=INSTRUMENT)
        keep = []
        for det in dets:
            c = camera[det].getCenter(FOCAL_PLANE)
            if np.hypot(c.getX(), c.getY()) <= max_radius:
                keep.append(det)
        import sys
        print(f'    {len(dets) - len(keep)} of {len(dets)} detectors beyond '
              f'{max_radius:g} mm left out', file=sys.stderr)
        dets = keep
    return dets


def make_visit_butler(repo=VISIT_REPO, collection=VISIT_COLLECTION):
    """
    Make a butler on the visit repo.

    Parameters
    ----------
    repo: str, optional
    collection: str, optional

    Returns
    -------
    butler: lsst.daf.butler.Butler
    """
    from lsst.daf.butler import Butler
    return Butler(repo, collections=[collection])


def load_iq_scores(butler):
    """
    Load the shapelets IQ scores of the run.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler

    Returns
    -------
    iq: dict
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
    Select the visit-detectors that went into the coadd of a patch.

    From deep_coadd_input_summary_tract (the IQ-selected inputs),
    with the shapelets IQ score when a lookup is given, best first.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    tract, patch: int
    band: str
    iq: dict, optional
        (visit, detector) -> shapelets IQ score, from
        load_iq_scores

    Returns
    -------
    inputs: structured array
        visit, detector, weight, goodpix, iq_score, sorted by
        iq_score (nan last)
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
    Render the sum of selected layers of an afw BackgroundList.

    Rendered exactly as BackgroundList.getImage does: a layer with
    an approximation style set renders from its stored polynomial,
    otherwise by interpolating its bin statistics.

    Parameters
    ----------
    bglist: lsst.afw.math.BackgroundList
    layers: iterable of int
        The layer indices to sum
    calib: float
        nJy per ADU (the stored layers are ADU)

    Returns
    -------
    image: f4 array
        In nJy
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
    """
    Print the layer structure of a BackgroundList: bins, approximation, order.

    Parameters
    ----------
    bglist: lsst.afw.math.BackgroundList
    name: str
        A label for the printout
    """
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
    Load one detector of a visit from the butler as a VisitExposure.

    The delivered visit_image (nJy) with the stored background
    layers rendered in nJy.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
        With the collection set (VISIT_COLLECTION)
    visit, detector: int
    rng: numpy Generator, optional
        For the noise realization

    Returns
    -------
    vexp: VisitExposure
        With the afw exposure kept in .exposure
    """
    from ..geom import ButlerWcs

    did = dict(
        instrument=INSTRUMENT, visit=int(visit),
        detector=int(detector),
    )

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
    flag_dead_pixels(mask, exp.variance.array)

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
    Get the Gaia extract covering a detector.

    Parameters
    ----------
    vexp: VisitExposure
    gaia_file: str, optional
        A file in the lsst_starsub.gaia.read_gaia_file layout;
        the TAP query when None
    gmax: float, optional
        The depth; default lsst_starsub.gaia.GMAX

    Returns
    -------
    gaia: structured array
    """
    from ..gaia import GMAX, fetch_gaia, read_gaia_file

    if gmax is None:
        gmax = GMAX

    if gaia_file is not None:
        return read_gaia_file(gaia_file, vexp.wcs, vexp.bbox, gmax=gmax)

    return fetch_gaia(vexp.wcs, vexp.bbox, gmax=gmax)
