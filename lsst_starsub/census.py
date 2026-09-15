"""
The bright-star census and the masks built from it.

select_stars picks the Gaia stars on and near a patch; circle_radius and
build_star_mask give each a magnitude-scaled mask circle;
field_segmentation keeps neighbors out of the star fits;
apply_star_taper fades the image to zero at the mask edges; and
make_star_table and make_starmask_plane are the outputs.  diffuse_mask
turns the joint fit's diffuse regions into a mask.  Shared by the two
star routes, the stamp templates (lsst_starsub.stamps) and the joint fit
(lsst_starsub.coadd.starsub).

The apodization is the consumer's job: the star routes return the
mask circles and the distance transform off them (dstar); whoever
measures on the image applies apply_star_taper with APOD_STARS and
treats dstar < APOD_STARS, the attenuation zone, as masked (zero
weight), after combining the per-band distance fields so every band
shares one zone, as lsst_mdet.cells.load_coadds_butler does.
make_starmask_plane records that zone.
"""
import numpy as np

from .maskbits import DM_INTRP, DM_NO_DATA, DM_SAT


# every census star is subtracted with the empirical extended
# template and masked with a floored magnitude-scaled circle
# that hides the core, where a field-average template is wrong.
# The circle law follows the flagged-arm extent measured over
# 56 saturated stars in 3 patches
MASK_R15 = 45.0     # circle radius in px at G = 15
MASK_SLOPE = 0.115  # radius scales as 10^(slope * (15 - G))
MASK_RMAX = 450.0
MINRAD = 20.0       # circle floor. subtracted cores never show
STAR_MARGIN = 210   # off-patch stars whose wings still intrude
GSAT = 15.2         # G saturation threshold of these coadds
GSUB = 19.0         # subtract stars brighter than this
BG_GROW = 12        # extra star-mask margin for the background
APOD_STARS = 12.0   # taper width outside the star mask
# px: the large diffuse segments (cirrus) that the joint fit leaves
# out of its source mask, so that its sky mesh fits them, are masked
# out to this distance.  Faint objects on the cirrus are biased
# whatever the sky treatment, and the mesh fitted to it
# over-subtracts past its edge: faint objects 5-10 percent low in
# flux at 50-200 px, unbiased by 200-250 px (arm B of
# run-dp2-test-cirrus-falloff, 2026-09-12).  One node spacing of the
# mesh (lsst_starsub.joint.SPACING)
DIFFUSE_MARGIN = 256

# sep's pixel stack for the field segmentation, entries: the
# start, grown by 4 on overflow up to the maximum
SEG_PIXSTACK = int(2e6)
SEG_PIXSTACK_MAX = int(3.2e7)


def circle_radius(gmag):
    """
    Get the mask circle radius of a star from its magnitude.

    MASK_R15 at G = 15, scaled as 10^(MASK_SLOPE (15 - G)), floored at
    MINRAD and capped at MASK_RMAX.

    Parameters
    ----------
    gmag: float or array
        Gaia G magnitude

    Returns
    -------
    radius: float or array
        In pixels, the shape of gmag
    """

    return np.minimum(
        np.maximum(
            MASK_R15 * 10 ** (MASK_SLOPE * (15.0 - gmag)),
            MINRAD,
        ),
        MASK_RMAX,
    )


def select_stars(gaia, x, y, mask0, gsub=GSUB, verbose=True,
                 intruder_gmax=None, intruder_margin=None):
    """
    Select the census: on-patch stars saturated or brighter than gsub.

    Off-patch stars brighter than intruder_gmax within intruder_margin
    px of the image, whose wings reach in, are included as intruders.
    No RUWE guard: Gaia at these depths is essentially pure point
    sources, and high-RUWE binaries are still stars we want gone.
    Referred to as "the subtract-and-mask census" in various places.

    Parameters
    ----------
    gaia: array with fields
        The gaia extract (ra, dec, phot_g_mean_mag, ruwe)
    x, y: arrays
        Patch-frame pixel positions of the gaia stars
    mask0: array
        The DM mask plane, for the saturation test
    gsub: float, optional
        Census depth: unsaturated on-patch stars brighter than
        this are included
    verbose: bool, optional
        Print the census counts
    intruder_gmax: float, optional
        Off-image stars brighter than this are intruders; default
        GSAT
    intruder_margin: float or callable, optional
        Intruders within this many px of the image; a callable is
        given the star's G and returns the margin, so the wings can
        set it (e.g. a multiple of circle_radius).  Default STAR_MARGIN

    Returns
    -------
    stars: structured array
        Fields ra, dec, x, y, G, ruwe, is_sat, on_image,
        sorted brightest first
    """
    if intruder_gmax is None:
        intruder_gmax = GSAT
    if intruder_margin is None:
        intruder_margin = STAR_MARGIN

    ny, nx = mask0.shape
    sat = (mask0 & DM_SAT) != 0
    rows = []

    for k in np.argsort(gaia['phot_g_mean_mag']):

        gmag = float(gaia['phot_g_mean_mag'][k])
        ruwe = float(gaia['ruwe'][k])

        xk, yk = float(x[k]), float(y[k])
        ix, iy = int(round(xk)), int(round(yk))

        on = 0 <= ix < nx and 0 <= iy < ny

        if on:
            m = 5
            is_sat = (
                gmag < GSAT + 0.5
                and sat[max(0, iy - m):iy + m + 1,
                        max(0, ix - m):ix + m + 1].any()
            )
            if not (is_sat or gmag < gsub):
                continue
        else:
            is_sat = 0
            if gmag >= intruder_gmax:
                continue
            if callable(intruder_margin):
                margin = float(intruder_margin(gmag))
            else:
                margin = float(intruder_margin)
            if not (-margin < ix < nx + margin
                    and -margin < iy < ny + margin):
                continue

        rows.append((
            float(gaia['ra'][k]), float(gaia['dec'][k]),
            xk, yk, gmag, ruwe, int(is_sat), int(on),
        ))

    stars = np.array(rows, dtype=_get_select_stars_dtype())

    non = int(stars['on_image'].sum())

    if verbose:
        print(
            f'    census: {non} on-patch '
            f'({int(stars["is_sat"].sum())} saturated) + '
            f'{len(stars) - non} off-patch intruders'
        )

    return stars


def patch_census(gaia, wcs, bbox, mask0, gsub=GSUB, coadd=False, verbose=True,
                 intruder_gmax=None, intruder_margin=None):
    """
    Make the census and its masks for one image, the routes' preamble.

    Parameters
    ----------
    gaia: array with fields
        The gaia extract (lsst_starsub.gaia)
    wcs: ButlerWcs or FileWcs
        For the pixel positions
    bbox: box
        The image's bounding box
    mask0: array
        The DM mask plane
    gsub: float, optional
        Census depth
    coadd: bool, optional
        See build_star_mask
    verbose: bool, optional
        Print the census and masked fraction
    intruder_gmax, intruder_margin: optional
        The off-image intruder rule, see select_stars

    Returns
    -------
    stars, starmask, comps, dstar, x, y:
        The census (select_stars), the bool star mask and the labeled
        component image (build_star_mask), the distance transform off
        the mask, and every gaia star's pixel position
    """
    from scipy import ndimage
    from .gaia import gaia_pixel_positions

    x, y = gaia_pixel_positions(gaia, wcs, bbox)

    stars = select_stars(
        gaia, x, y, mask0, gsub=gsub, verbose=verbose,
        intruder_gmax=intruder_gmax, intruder_margin=intruder_margin,
    )

    starmask, comps = build_star_mask(
        stars, mask0, verbose=verbose,
        coadd=coadd,
    )

    dstar = ndimage.distance_transform_edt(~starmask)

    return stars, starmask, comps, dstar, x, y


def _get_select_stars_dtype():
    """
    Get the dtype of the census table.

    Returns
    -------
    dtype: list of (name, type)
    """
    return [
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('G', 'f4'),
        ('ruwe', 'f4'),
        ('is_sat', 'i2'),
        ('on_image', 'i2'),
    ]


def own_component_ids(comps, ix, iy):
    """
    Get the labels of the flagged mask components at a star.

    Sampled at the star's integer position and +-2 px around it, so a
    slightly displaced flag still counts as the star's own.

    Parameters
    ----------
    comps: array
        The labeled SAT/INTRP component image
    ix, iy: int
        The star's integer pixel position

    Returns
    -------
    ids: set of int
        The component labels, possibly empty
    """
    ny, nx = comps.shape

    ids = set()

    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            lid = comps[
                np.clip(iy + dy, 0, ny - 1),
                np.clip(ix + dx, 0, nx - 1)
            ]
            if lid > 0:
                ids.add(int(lid))

    return ids


def build_star_mask(stars, mask0, verbose=True, coadd=False):
    """
    Build the star mask: a circle per census star plus flagged components.

    The circles follow circle_radius; the saturated stars also get the
    flagged mask components at their positions.

    Parameters
    ----------
    stars: structured array
        The census from select_stars
    mask0: array
        The DM mask plane, for the flagged components
    verbose: bool, optional
        Print the masked fraction
    coadd: bool, optional
        True for a coadd: the components are the NO_DATA regions,
        and every one touching the mask is added.  A coadd carries
        SAT/INTRP also where other epochs still give usable data;
        NO_DATA marks where none survived (the coadd interpolates
        only there).  False for a single exposure: the SAT/INTRP
        components at the star positions

    Returns
    -------
    starmask, comps:
        The bool star mask and the labeled component image (used
        later for the per-star own-component masks)
    """
    from scipy import ndimage

    ny, nx = mask0.shape

    bits = DM_NO_DATA if coadd else (DM_SAT | DM_INTRP)
    comps, _ = ndimage.label((mask0 & bits) != 0)

    starmask = np.zeros((ny, nx), dtype=bool)
    star_ids = set()

    for st in stars:
        xk, yk = float(st['x']), float(st['y'])
        ix, iy = int(round(xk)), int(round(yk))

        if st['on_image']:
            star_ids |= own_component_ids(comps, ix, iy)

        rad = circle_radius(float(st['G']))

        ir = int(np.ceil(rad))

        y0, y1 = max(0, iy - ir), min(ny, iy + ir + 1)
        x0, x1 = max(0, ix - ir), min(nx, ix + ir + 1)

        if y1 <= y0 or x1 <= x0:
            continue

        ly, lx = np.mgrid[y0:y1, x0:x1]

        starmask[y0:y1, x0:x1] |= (
            np.hypot(ly - yk, lx - xk) <= rad
        )

    if star_ids:
        starmask |= np.isin(comps, sorted(star_ids))

    if coadd:
        # no-data regions reaching past the circles, e.g. diffraction
        # spikes that rejection removed from every epoch, but not
        # attached to the star's own component
        touch = ndimage.binary_dilation(
            starmask, structure=np.ones((3, 3), dtype=bool),
        )
        touch_ids = np.unique(comps[touch & (comps > 0)])
        if touch_ids.size > 0:
            starmask |= np.isin(comps, touch_ids)

    if verbose:
        print(f'    star mask fraction {starmask.mean():.3f}')

    return starmask, comps


def field_segmentation(image, good, sig):
    """
    Segment the sources of a whole image at 1.5 sigma.

    Parameters
    ----------
    image: array
        The image
    good: bool array
        The usable pixels
    sig: float
        The pixel noise for the detection threshold

    Returns
    -------
    seg: int array
        The segmentation map, 0 for sky
    """
    imf = np.ascontiguousarray(image, dtype='f4')
    _, seg = sep_extract(imf, 1.5, sig, ~good, retry_deblend=True)
    return seg


def sep_extract(imf, thresh, err, mask, retry_deblend=False, **kwargs):
    """
    Run sep.extract with the pixel stack managed.

    sep's pixel stack is process-global and touched in full on every
    extract call (41 bytes per entry: 2e7 entries cost 0.8 GB, and
    every later sep call in the process paid it, the per-cell
    detections included).  The stack starts at SEG_PIXSTACK, grows by
    4 on overflow up to SEG_PIXSTACK_MAX, and the previous settings
    are put back.

    Parameters
    ----------
    imf: array
        The image, contiguous float32
    thresh: float
        The detection threshold, in units of err
    err: float or array
        The pixel noise
    mask: bool array
        True for the pixels to ignore
    retry_deblend: bool, optional
        On a deblending overflow (a very bright star's wing above
        threshold exceeding the sub-object limit, seen on visit
        images) retry with a single deblend threshold, no
        sub-objects; for callers that only need the segmentation
    kwargs: dict
        Passed to sep.extract (filter_kernel, minarea, the deblend
        settings, ...)

    Returns
    -------
    objs, seg: structured array, int array
        The object table and the segmentation map (label k for
        objs[k - 1], 0 for none)
    """
    import sep

    old_stack = sep.get_extract_pixstack()
    old_sub = sep.get_sub_object_limit()
    stack = SEG_PIXSTACK

    try:
        sep.set_sub_object_limit(10240)

        while True:
            sep.set_extract_pixstack(stack)
            try:
                return sep.extract(
                    imf, thresh, err=err, mask=mask, segmentation_map=True,
                    **kwargs,
                )
            except Exception as error:
                msg = str(error)
                if 'pixel buffer full' in msg and stack < SEG_PIXSTACK_MAX:
                    stack *= 4
                    print(f'    segmentation pixel stack full; '
                          f'retrying with {stack}')
                elif ('deblending overflow' in msg and retry_deblend
                      and kwargs.get('deblend_nthresh') != 1):
                    print('    segmentation deblending overflow; '
                          'retrying without deblending')
                    kwargs = dict(kwargs, deblend_nthresh=1, deblend_cont=1.0)
                else:
                    raise
    finally:
        sep.set_extract_pixstack(old_stack)
        sep.set_sub_object_limit(old_sub)


def make_star_table(stars, slist):
    """
    Build the star table of the output file.

    The census with the fitted amplitude A filled in for the stars that
    received stamps, 0 for the others.

    Parameters
    ----------
    stars: structured array
        The census from select_stars
    slist: list of dict
        The per-star work list from stamps.subtract_stars; empty for
        a route that fills A itself

    Returns
    -------
    star_table: structured array
        The census fields plus A
    """

    star_table = np.zeros(len(stars), dtype=_get_star_table_dtype())

    for name in (
        'ra',
        'dec',
        'x',
        'y',
        'G',
        'ruwe',
        'is_sat',
        'on_image',
    ):
        star_table[name] = stars[name]

    for st in slist:
        star_table['A'][st['idx']] = st['A']

    return star_table


def _get_star_table_dtype():
    """
    Get the dtype of the star table: the census fields plus A.

    Returns
    -------
    dtype: list of (name, type)
    """
    return [
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('G', 'f4'),
        ('ruwe', 'f4'),
        ('is_sat', 'i2'),
        ('on_image', 'i2'),
        ('A', 'f8'),
    ]


def apply_star_taper(deep_coadd, dstar, width=APOD_STARS):
    """
    Apodize the star-mask regions of a coadd in place.

    The image and the noise realization are multiplied by
    taper_from_distance, zero inside the mask and rising to one over
    width px outside it, so they stay statistically matched.  Applied
    after any background determination.

    Parameters
    ----------
    deep_coadd: deep_coadd
        The coadd; its image and noise realization are modified
    dstar: array
        The distance from the star mask, patch frame
    width: float, optional
        The taper width in pixels; default APOD_STARS
    """
    taper = taper_from_distance(dstar, width)
    deep_coadd.image.array[:, :] *= taper
    deep_coadd.noise_realizations[0].array[:, :] *= taper


def make_starmask_plane(starmask, dstar, apod):
    """
    Build the three-valued star-mask plane of the output.

    0 clear; 1 the taper zone (attenuated: masked for any measurement,
    smooth enough for FFTs); 2 the star mask (zeroed when apod > 0).

    Parameters
    ----------
    starmask: bool array
        The star mask
    dstar: array
        The distance from the star mask, patch frame
    apod: float
        The taper width actually applied; 0 for no taper zone

    Returns
    -------
    plane: u1 array
        The shape of starmask
    """
    plane = np.zeros(starmask.shape, dtype='u1')

    if apod > 0:
        plane[(dstar > 0) & (dstar < apod)] = 1

    plane[starmask] = 2

    return plane


def taper_from_distance(dist, width):
    """
    Get a smooth taper as a function of the distance into valid territory.

    0 at distance 0, 1 at width and beyond, the cumulative triweight
    kernel of the cell-edge apodization, so star masks and cell edges
    share one smoothness class.

    Parameters
    ----------
    dist: array
        The distance into the valid region, px
    width: float
        The taper width, px

    Returns
    -------
    taper: array
        The shape of dist
    """

    y = (np.asarray(dist, dtype=float) - width) * (6.0 / width) + 3
    out = np.where(y > 3, 1.0, 0.0)

    w = (y >= -3) & (y <= 3)
    yy = y[w]

    out[w] = (
        -5 * yy ** 7 / 69984
        + 7 * yy ** 5 / 2592
        - 35 * yy ** 3 / 864
        + 35 * yy / 96
        + 1 / 2
    )

    return out


def diffuse_mask(starsub_fits, margin):
    """
    Get the mask of the joint fit's diffuse regions, all bands, grown.

    The union over the bands of the large diffuse segments the fit did
    not mask as sources (the fit dicts' 'diffuse'), grown by margin px.

    Parameters
    ----------
    starsub_fits: dict
        The joint fit dicts, band -> fit
    margin: float
        The growth in px

    Returns
    -------
    mask: bool array or None
        None when no band has a diffuse region
    """
    from scipy import ndimage

    masks = [
        fit['diffuse'] for fit in starsub_fits.values()
        if fit.get('diffuse') is not None
    ]

    if not masks:
        return None

    union = np.logical_or.reduce(masks)

    if not union.any():
        return None

    if margin > 0:
        union = ndimage.distance_transform_edt(~union) <= margin

    return union
