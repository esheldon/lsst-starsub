"""
per-star wing profiles across the image states

The characterization question is how much wing light the
pipeline's 32 px background pass absorbed, and how well the
joint sky-plus-template model recovers it.  Each census star's
azimuthal median profile outside its own mask is measured on
several states of the same detector image (delivered, restored
and sky-flattened, star-subtracted residual) so the curves are
directly comparable.  Profiles are in sky-sigma units and, for
the stack, additionally normalized by 10^(-0.4 G)
"""
import numpy as np

from .census import APOD_STARS, circle_radius

# log-spaced annuli from just outside the mask floor to beyond
# the template extents; radii are from the star center
R_MIN = 12.0
R_MAX = 900.0
NBIN = 20
MIN_PIX = 50    # annulus pixels needed for a median


# linear annuli in the distance beyond the mask radius, the
# binning of the coadd dual-state stacks
DMASK_MAX = 600.0
DMASK_STEP = 10.0


def radial_edges():
    return np.unique(np.round(np.logspace(
        np.log10(R_MIN), np.log10(R_MAX), NBIN + 1,
    )))


def dmask_edges():
    return np.arange(0.0, DMASK_MAX + DMASK_STEP / 2, DMASK_STEP)


def ambient_levels(states, vexp, seg, exclude):
    """
    per-state ambient reference: the median of the good,
    unsegmented pixels outside the exclusion zone.  The states
    differ by constant pedestals (the sky correction's net
    offset, the ambient reference of the subtraction) that
    would otherwise read as wing light
    """
    ok = vexp.good & (seg == 0) & ~exclude
    return {
        name: float(np.median(image[ok]))
        for name, image in states.items()
    }


def star_window(shape, x, y, edges):
    """
    the slice of the image around a star covering the outer
    annulus edge, and the radius grid on it; (None, None) when
    the window is empty
    """
    ny, nx = shape
    m = int(edges[-1]) + 2
    icx, icy = int(round(x)), int(round(y))
    x0, x1 = max(0, icx - m), min(nx, icx + m + 1)
    y0, y1 = max(0, icy - m), min(ny, icy + m + 1)
    if x1 <= x0 or y1 <= y0:
        return None, None
    gy, gx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(gy - y, gx - x)
    return np.s_[y0:y1, x0:x1], rr


def star_profile(image, ok, rr, rad, edges):
    """
    the azimuthal median profile of one star outside its mask
    circle, on its window

    Parameters
    ----------
    image: array
        The image cut to the star's window
    ok: bool array
        Usable pixels on the window (good, unsegmented or own
        segment, other stars' zones out)
    rr: array
        Radius from the star on the window
    rad: float
        The star's mask radius; annuli inside it are NaN
    edges: array
        Annulus edges

    Returns
    -------
    prof, npix: arrays over the annuli (NaN / 0 where empty)
    """
    prof = np.full(edges.size - 1, np.nan)
    npix = np.zeros(edges.size - 1, dtype=int)

    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        w = (rr >= max(lo, rad + 2.0)) & (rr < hi) & ok
        n = int(w.sum())
        npix[i] = n
        if n >= MIN_PIX:
            prof[i] = np.median(image[w])
    return prof, npix


def exclusion_radius(gmag, wide):
    """
    the radius to which another star is kept out of a profile:
    its wide (template-extent) radius, matching the ambient
    reference's exclusion, or its mask circle plus the taper
    """
    from .visit import WIDE_GMAX, WIDE_GROW
    from .stamps import TMPL_OUT_MAX, template_out_half

    if wide:
        if gmag < WIDE_GMAX:
            return float(min(template_out_half(gmag), TMPL_OUT_MAX))
        return float(circle_radius(gmag)) + WIDE_GROW
    return float(circle_radius(gmag)) + APOD_STARS


def usable_pixels(good, seg, stars, st, sl, rr, wide=True):
    """
    pixels usable for one star's profile on its window: good,
    not another detection (the star's own segments allowed),
    other stars excluded to their exclusion_radius

    Parameters
    ----------
    good: bool array
        The full-frame usable mask
    seg: array
        The full-frame segmentation
    stars: structured array
        The census
    st: record
        The star
    sl: slice
        The star's window
    rr: array
        Radius from the star on the window
    wide: bool, optional
        Exclude other stars to their wide radius (their wings
        are then out of the annuli, as they are out of the
        ambient reference); False for the mask circle plus
        taper only
    """
    segc = seg[sl]
    rad = float(circle_radius(float(st['G'])))
    own = set(np.unique(segc[rr <= rad + 2])) - {0}
    ok = good[sl] & ((segc == 0) | np.isin(segc, sorted(own)))

    ny, nx = ok.shape
    y0, x0 = sl[0].start, sl[1].start
    for ot in stars:
        if ot['x'] == st['x'] and ot['y'] == st['y']:
            continue
        orad = exclusion_radius(float(ot['G']), wide)
        ox, oy = float(ot['x']), float(ot['y'])
        # the neighbor's zone clipped to the window; the radius
        # test runs on that sub-box only (a full-window hypot per
        # neighbor made this quadratic in the star count times
        # the window area on dense fields)
        bx0 = max(0, int(np.floor(ox - orad)) - x0)
        bx1 = min(nx, int(np.ceil(ox + orad)) - x0 + 1)
        by0 = max(0, int(np.floor(oy - orad)) - y0)
        by1 = min(ny, int(np.ceil(oy + orad)) - y0 + 1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        gy, gx = np.mgrid[by0 + y0:by1 + y0, bx0 + x0:bx1 + x0]
        sub = ok[by0:by1, bx0:bx1]
        sub &= np.hypot(gy - oy, gx - ox) > orad
    return ok


LOCAL_REF = (500.0, 600.0)   # d - r_mask range of the local reference


def measure_profiles(states, vexp, stars, seg, gmax=17.0, mode='r',
                     ambient=None, wide=True, local_ref=LOCAL_REF,
                     edges=None, gmin=None, good=None):
    """
    per-star profiles on every image state, in sky-sigma units

    Parameters
    ----------
    states: dict name -> array
        The image states, all the same shape
    vexp: VisitExposure
        For the good mask and the noise
    stars: structured array
        The census
    seg: array
        A segmentation of the field (the final residual's)
    gmax: float, optional
        Measure on-image stars brighter than this
    mode: str, optional
        'r': log annuli in radius from the star (radial_edges);
        'dmask': linear annuli in the distance beyond the
        star's mask radius (dmask_edges)
    edges: array, optional
        Annulus edges overriding the mode's default (in the
        mode's convention)
    gmin: float, optional
        Measure only stars at or fainter than this
    good: bool array, optional
        Overrides vexp.good as the usable-pixel mask (e.g. with a
        detection mask applied, the star's own features included)
    ambient: dict, optional
        name -> level subtracted from each state before the
        measurement (ambient_levels)
    wide: bool, optional
        Exclude other stars to their wide radius (usable_pixels)
    local_ref: (lo, hi), optional
        The star's own local reference level is the median of
        the usable pixels between lo and hi beyond its mask
        radius; recorded per row in 'local' (same units as
        prof) so a stack can reference the profile locally
        instead of to the ambient level

    Returns
    -------
    edges, table: the annulus edges and a structured array with
    one row per (star, state): idx, G, x, y, state, prof
    (sky-sigma units, ambient-referenced), npix, local (the
    local reference level, NaN when unmeasurable), nlocal
    """
    if edges is not None:
        edges = np.asarray(edges, dtype='f8')
    elif mode == 'r':
        edges = radial_edges()
    elif mode == 'dmask':
        edges = dmask_edges()
    else:
        raise ValueError(f'unknown mode {mode!r}')
    nb = edges.size - 1
    sig = vexp.sky_sigma
    names = list(states.keys())
    maxlen = max(len(n) for n in names)
    if ambient is None:
        ambient = {name: 0.0 for name in names}

    if good is None:
        good = vexp.good
    rows = []
    for si, st in enumerate(stars):
        if not st['on_image'] or float(st['G']) >= gmax:
            continue
        if gmin is not None and float(st['G']) < gmin:
            continue
        rad = float(circle_radius(float(st['G'])))
        # absolute-radius edges for the annuli; the window
        # covers them and the local reference annulus
        redges = edges if mode == 'r' else edges + rad
        rmax = max(float(redges[-1]), rad + local_ref[1])
        sl, rr = star_window(
            seg.shape, float(st['x']), float(st['y']),
            np.array([rmax]),
        )
        if sl is None:
            continue
        ok = usable_pixels(good, seg, stars, st, sl, rr, wide=wide)
        wloc = ok & (rr >= rad + local_ref[0]) & (rr < rad + local_ref[1])
        nloc = int(wloc.sum())
        for name in names:
            cut = states[name][sl]
            prof, npix = star_profile(cut, ok, rr, rad, redges)
            local = np.nan
            if nloc >= MIN_PIX:
                local = (float(np.median(cut[wloc])) - ambient[name]) / sig
            rows.append((
                si, float(st['G']), float(st['x']), float(st['y']),
                name, (prof - ambient[name]) / sig, npix, local, nloc,
            ))

    table = np.zeros(len(rows), dtype=[
        ('idx', 'i4'), ('G', 'f4'), ('x', 'f8'), ('y', 'f8'),
        ('state', f'U{maxlen}'), ('prof', 'f4', nb),
        ('npix', 'i4', nb), ('local', 'f4'), ('nlocal', 'i4'),
    ])
    for i, row in enumerate(rows):
        table[i] = row
    return edges, table


def stack_profiles(table, state, glo, ghi, min_stars=2,
                   normalize=True):
    """
    the median over stars in [glo, ghi) of the profile for one
    state, in sky-sigma units, divided by 10^(-0.4 G) when
    normalize is set; NaN where fewer than min_stars contribute

    Returns
    -------
    med, count
    """
    w = (
        (table['state'] == state)
        & (table['G'] >= glo) & (table['G'] < ghi)
    )
    if w.sum() == 0:
        nb = table['prof'].shape[1]
        return np.full(nb, np.nan), np.zeros(nb, dtype=int)
    profs = table['prof'][w]
    if normalize:
        profs = profs / (
            10.0 ** (-0.4 * table['G'][w])
        )[:, np.newaxis]
    count = np.sum(np.isfinite(profs), axis=0)
    med = np.full(profs.shape[1], np.nan)
    wc = count >= min_stars
    if wc.any():
        med[wc] = np.nanmedian(profs[:, wc], axis=0)
    return med, count
