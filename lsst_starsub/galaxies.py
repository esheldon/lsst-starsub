"""
Large galaxies: the catalog selection and the data-driven mask.

Nearby galaxies of a fraction of an arcminute and up make giant
blend groups in the cell processing (runtime, and corrupted
measurements of everything blended with them), so their regions are
masked before detection.  The catalog (HyperLEDA, read_galaxy_file)
only says where the candidates are: its isophotal sizes are the
catalog's blue D25, not the extent of the galaxy in deep coadds (on
run-dp2-v01 the blend groups reach ~2.5 D25/2 from the center), so
the size of the mask comes from the data.  The joint star-and-sky
fit segments the image (joint.deep_segmentation) and keeps the large
sources of its last pass (the fit dicts' 'big_sources': centroids,
moment ellipses and isophotal areas); galaxy_mask matches each
catalog galaxy to the large source at its position and paints that
source's isophotal ellipse scaled by GAL_SCALE, the union over the
bands, plus the catalog's own D25 ellipse as a floor: the catalog
sizes err small, so the floor costs nothing in the usual case and
only bites when star masks or a bright neighbor have chopped the
segment (IC 5078, an edge-on Sc of 3.5 arcmin with two star masks
on its disk).  A catalog galaxy with no large source at its
position is too faint or small to matter and gets no mask.

The mask is of the zero-weight kind, like the diffuse (cirrus)
mask: the consumer leaves the pixels in the image, gives them zero
weight and clears them from the footprint; no taper is needed since
nothing is zeroed in the image.
"""

import numpy as np

# arcmin: catalog galaxies at least this large are candidates.  The
# mask size comes from the data, so a candidate that turns out small
# in the image costs nothing; the threshold only bounds the catalog
# work.  0.3 takes the whole of the HyperLEDA extract (D25 > 0.32'):
# on the run-dp2-v01 region galaxies of D25 0.36-0.49' still made
# blend groups of 50-75 objects
D25MIN = 0.3

# the mask ellipse's semi-major axis in units of the matched source's
# isophotal semi-major axis.  The segmentation is at the detection
# threshold, so the isophote is already deep: on 06342-00062 ESO
# 598-31 (D25 0.89') has an isophotal radius of 246 px, 1.85 x
# D25/2, and its blend groups in run-dp2-v01 reached ~2.5 x D25/2;
# 1.5 covers that.  To be tuned on the region
GAL_SCALE = 1.5

# px: cap on the mask's semi-major axis
GAL_RMAX = 1500.0

# the catalog position and the source centroid must agree to this
# fraction of D25/2, with a floor of GAL_MATCH_MIN px (the catalog
# positions are good to a few arcsec, the centroid of a merged
# segment can be off)
GAL_MATCH_FRAC = 0.5
GAL_MATCH_MIN = 25.0

# the catalog D25 ellipse scaled by this is kept out of the sky fit
# (catalog_exclusion): the joint fit's own exclusion of large sources
# is capped at joint.SEG_BIG_RMAX px, and the outer light of a galaxy
# larger than that is fit as sky and subtracted, a dark halo around
# it (IC 5078, 3.5 arcmin).  The catalog sizes err small, so the
# factor is generous; the mesh under the excluded region follows its
# smoothness prior
GAL_SKY_SCALE = 2.0

PIXEL_SCALE = 0.2   # arcsec per px

GALAXY_DTYPE = [
    ('pgc', 'i8'), ('ra', 'f8'), ('dec', 'f8'), ('d25_arcmin', 'f8'),
    ('logr25', 'f8'), ('pa', 'f8'), ('name', 'U24'),
]


def read_galaxy_file(fname, wcs, bbox, d25min=D25MIN):
    """
    Read the catalog galaxies that can reach a patch.

    A FITS table with ra, dec (degrees) and d25_arcmin (or logd25 in
    the HyperLEDA convention, log10 of D25 in 0.1 arcmin); logr25,
    pa, pgc and name are carried when present.  The galaxies of at
    least d25min inside the patch's circle grown by their own D25/2
    are returned, so a galaxy centered off the patch whose light
    reaches it is included.

    Parameters
    ----------
    fname: str
        The file
    wcs: ButlerWcs or FileWcs
        For the patch's sky position (pixelToSkyArray)
    bbox: box
        The patch's bounding box
    d25min: float, optional
        arcmin; default D25MIN

    Returns
    -------
    gals: structured array
        GALAXY_DTYPE, possibly empty
    """
    import rustfits

    data = rustfits.read(fname)
    columns = list(data.dtype.names)

    if 'd25_arcmin' in columns:
        d25 = np.asarray(data['d25_arcmin'], dtype='f8')
    elif 'logd25' in columns:
        d25 = 0.1 * 10.0 ** np.asarray(data['logd25'], dtype='f8')
    else:
        raise ValueError(f'{fname} has neither d25_arcmin nor logd25')

    keep = np.isfinite(d25) & (d25 >= d25min)
    data, d25 = data[keep], d25[keep]

    # the patch circle: the center to a corner, plus each galaxy's
    # own radius
    xmid = 0.5 * (bbox.x.start + bbox.x.stop)
    ymid = 0.5 * (bbox.y.start + bbox.y.stop)
    ra_c, dec_c = wcs.pixelToSkyArray(
        np.array([xmid, float(bbox.x.start)]),
        np.array([ymid, float(bbox.y.start)]),
        degrees=True,
    )
    rad = _separation(ra_c[0], dec_c[0], ra_c[1], dec_c[1]) + 0.005

    sep = _separation(
        ra_c[0], dec_c[0],
        np.asarray(data['ra'], dtype='f8'),
        np.asarray(data['dec'], dtype='f8'),
    )
    w, = np.where(sep <= rad + d25 / 120.0)

    gals = np.zeros(w.size, dtype=GALAXY_DTYPE)
    gals['ra'] = data['ra'][w]
    gals['dec'] = data['dec'][w]
    gals['d25_arcmin'] = d25[w]
    for name in ('pgc', 'logr25', 'pa', 'name'):
        if name in columns:
            gals[name] = data[name][w]
        elif name != 'name':
            gals[name] = np.nan if name != 'pgc' else 0

    print(f'    galaxies from {fname}: {gals.size} of D25 >= '
          f'{d25min:g} arcmin')
    return gals


def _separation(ra0, dec0, ra, dec):
    """great-circle separation in degrees"""
    ra0r, dec0r = np.deg2rad(ra0), np.deg2rad(dec0)
    rar, decr = np.deg2rad(ra), np.deg2rad(dec)
    cossep = (
        np.sin(dec0r) * np.sin(decr)
        + np.cos(dec0r) * np.cos(decr) * np.cos(rar - ra0r)
    )
    return np.rad2deg(np.arccos(np.clip(cossep, -1, 1)))


def galaxy_pixel_positions(gals, wcs, bbox):
    """
    Get the galaxies' patch-frame pixel positions.

    Parameters
    ----------
    gals: structured array
        From read_galaxy_file
    wcs: ButlerWcs or FileWcs
        The image's wcs
    bbox: box
        The image's bounding box

    Returns
    -------
    x, y: arrays
        Pixel positions relative to the box origin
    """
    x, y = wcs.skyToPixelArray(gals['ra'], gals['dec'], degrees=True)
    return x - bbox.x.start, y - bbox.y.start


def in_source_ellipse(src, x, y, scale=1.0):
    """
    Whether a position lies inside a source's isophotal ellipse.

    Parameters
    ----------
    src: one row of the sep table
    x, y: float
        The position
    scale: float, optional
        Scale on the ellipse; default 1, the isophote itself

    Returns
    -------
    inside: bool
    """
    a, b, theta = source_ellipse(src, scale=scale, rmax=np.inf)
    dx, dy = x - float(src['x']), y - float(src['y'])
    c, s = np.cos(theta), np.sin(theta)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def match_big_source(big, x, y, d25_arcmin):
    """
    Find the large source at a galaxy's position.

    The nearest of the sources whose centroid is within
    GAL_MATCH_FRAC x D25/2 (at least GAL_MATCH_MIN px) of the
    position, or, failing that, the nearest of the sources whose
    isophotal ellipse contains the position: a segment merged with a
    neighbor (a pair of ellipticals) or clipped by the patch edge (a
    galaxy centered just off the patch) has its centroid well away
    from the catalog position while its isophote still covers it.

    Parameters
    ----------
    big: structured array
        The sep table of the large sources (joint.deep_segmentation
        return_big), with x, y, a, b, theta, npix
    x, y: float
        The galaxy's pixel position
    d25_arcmin: float
        Its catalog D25

    Returns
    -------
    index: int or None
        Into big
    """
    if big is None or big.size == 0:
        return None

    r25 = d25_arcmin * 60 / PIXEL_SCALE / 2
    rmatch = max(GAL_MATCH_MIN, GAL_MATCH_FRAC * r25)
    d = np.hypot(big['x'] - x, big['y'] - y)
    order = np.argsort(d)
    k = int(order[0])
    if d[k] <= rmatch:
        return k

    for k in order:
        if in_source_ellipse(big[k], x, y):
            return int(k)

    return None


def source_ellipse(src, scale=GAL_SCALE, rmax=GAL_RMAX):
    """
    Get the mask ellipse of a large source.

    The isophotal ellipse has the moment axis ratio and orientation
    and the isophotal area, so its semi-major axis is
    sqrt(npix / pi) / sqrt(b / a); the mask is that times scale,
    capped at rmax.

    Parameters
    ----------
    src: one row of the sep table
    scale, rmax: float, optional
        Default GAL_SCALE, GAL_RMAX

    Returns
    -------
    a, b, theta: floats
        The semi-axes in px and the orientation (sep's theta,
        radians)
    """
    a0 = max(float(src['a']), 1.0)
    b0 = max(float(src['b']), 1.0)
    q = min(b0 / a0, 1.0)
    riso = np.sqrt(float(src['npix']) / np.pi)
    a = min(scale * riso / np.sqrt(q), rmax)
    return a, a * q, float(src['theta'])


def catalog_ellipse(gal):
    """
    Get a catalog galaxy's D25 ellipse in the patch frame.

    The semi-major axis is D25/2 in px, the axis ratio 10^-logr25
    (1 when the catalog has none) and the orientation from the
    position angle east of north (0 when none): the patch frame has
    north along +y and east along -x (right ascension decreases
    with x), so an angle of pa east of north is 90 + pa degrees
    counter-clockwise from +x, sep's theta.  (Through v0.3.0 this
    was 90 - pa, the mirror image, wrong for position angles away
    from 0 and 90.)

    Parameters
    ----------
    gal: one row of GALAXY_DTYPE

    Returns
    -------
    a, b, theta: floats
        The semi-axes in px and the orientation, radians
    """
    a = float(gal['d25_arcmin']) * 60 / PIXEL_SCALE / 2
    logr = float(gal['logr25'])
    q = 10.0 ** (-logr) if np.isfinite(logr) else 1.0
    q = min(max(q, 0.05), 1.0)
    pa = float(gal['pa'])
    if not np.isfinite(pa):
        pa = 0.0
    return a, a * q, np.deg2rad(90.0 + pa)


def catalog_exclusion(gals, x, y, shape, scale=GAL_SKY_SCALE):
    """
    Get the region of the catalog galaxies to keep out of the sky fit.

    The union of the galaxies' D25 ellipses (catalog_ellipse) scaled
    by scale; every catalog galaxy in the patch contributes, matched
    to a source or not, since the fit has not run yet.

    Parameters
    ----------
    gals: structured array
        From read_galaxy_file
    x, y: arrays
        Their pixel positions (galaxy_pixel_positions)
    shape: (ny, nx)
        The image shape
    scale: float, optional
        Default GAL_SKY_SCALE

    Returns
    -------
    mask: bool array or None
        None without galaxies
    """
    import sep

    if gals is None or gals.size == 0:
        return None

    mask = np.zeros(shape, dtype=bool)
    for k in range(gals.size):
        a, b, theta = catalog_ellipse(gals[k])
        sep.mask_ellipse(
            mask, np.array([x[k]]), np.array([y[k]]),
            np.array([scale * a]), np.array([scale * b]),
            np.array([theta]), r=1.0,
        )

    return mask if mask.any() else None


def galaxy_mask(starsub_fits, gals, x, y, shape, scale=GAL_SCALE,
                rmax=GAL_RMAX, verbose=True):
    """
    Get the mask of the catalog galaxies, all bands.

    Per band, each galaxy is matched to the large source at its
    position (match_big_source) and that source's ellipse
    (source_ellipse) is painted; the union over the bands is
    returned, with the catalog D25 ellipse (catalog_ellipse) of
    every galaxy matched in at least one band added as a floor.

    Parameters
    ----------
    starsub_fits: dict
        The joint fit dicts, band -> fit (with 'big_sources')
    gals: structured array
        From read_galaxy_file
    x, y: arrays
        Their pixel positions (galaxy_pixel_positions)
    shape: (ny, nx)
        The image shape
    scale, rmax: float, optional
        Default GAL_SCALE, GAL_RMAX
    verbose: bool, optional

    Returns
    -------
    mask: bool array or None
        None when no galaxy matched a source in any band
    table: structured array
        One row per galaxy per band matched: pgc, band, x, y (the
        source centroid), a, b (px), theta, npix
    """
    import sep

    rows = []
    mask = np.zeros(shape, dtype=bool)
    ny, nx = shape

    for band, fit in starsub_fits.items():
        big = fit.get('big_sources') if fit is not None else None
        if verbose:
            nbig = 0 if big is None else big.size
            print(f'    galaxy mask band {band}: {nbig} large sources')
            for k in range(gals.size):
                if nbig > 0:
                    d = np.hypot(big['x'] - x[k], big['y'] - y[k])
                    j = int(np.argmin(d))
                    print(f'      pgc {gals["pgc"][k]} D25 '
                          f'{gals["d25_arcmin"][k]:.2f}\' at '
                          f'({x[k]:.0f}, {y[k]:.0f}): nearest source '
                          f'{d[j]:.0f} px, npix {big["npix"][j]}')
        for k in range(gals.size):
            j = match_big_source(
                big, float(x[k]), float(y[k]), float(gals['d25_arcmin'][k]),
            )
            if j is None:
                continue
            src = big[j]
            a, b, theta = source_ellipse(src, scale=scale, rmax=rmax)
            sep.mask_ellipse(
                mask, np.array([src['x']]), np.array([src['y']]),
                np.array([a]), np.array([b]), np.array([theta]), r=1.0,
            )
            rows.append((
                int(gals['pgc'][k]), band, float(src['x']), float(src['y']),
                a, b, theta, int(src['npix']),
            ))

    table = np.array(rows, dtype=[
        ('pgc', 'i8'), ('band', 'U4'), ('x', 'f8'), ('y', 'f8'),
        ('a', 'f8'), ('b', 'f8'), ('theta', 'f8'), ('npix', 'i8'),
    ])

    # the floor: the catalog ellipse of every matched galaxy
    matched = np.isin(gals['pgc'], table['pgc']) if table.size else \
        np.zeros(gals.size, dtype=bool)
    data_frac = mask.mean()
    for k in np.flatnonzero(matched):
        a, b, theta = catalog_ellipse(gals[k])
        sep.mask_ellipse(
            mask, np.array([x[k]]), np.array([y[k]]),
            np.array([a]), np.array([b]), np.array([theta]), r=1.0,
        )

    if verbose:
        nmatched = np.unique(table['pgc']).size if table.size else 0
        print(f'    galaxy mask: {nmatched} of {gals.size} catalog galaxies '
              f'matched a large source; masked fraction {mask.mean():.4f} '
              f'({data_frac:.4f} from the data ellipses alone)')

    if not mask.any():
        return None, table

    return mask, table
