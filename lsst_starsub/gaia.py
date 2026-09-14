"""
Gaia DR3 stars: the patch extracts and positions.

The patch extracts come from the per-tract files that
lsst-starsub-make-gaia writes (read_gaia_file) or from the Gaia TAP
service (fetch_gaia), cut to a circle around the patch and converted
to one layout; gaia_pixel_positions propagates the proper motions and
maps them to patch pixels.  The per-visit extracts of the visit project are
lsst_starsub.visit.gaia.
"""

import numpy as np


# Gaia positions are queried at the catalog epoch and
# propagated by proper motion to the approximate observation
# epoch
GAIA_EPOCH = 2016.0
OBS_EPOCH = 2025.0
GMAX = 19.0         # download depth

# TAP sync endpoints serving gaiadr3.gaia_source with the same
# query dialect and csv output; tried in order.  ESA is the
# canonical archive, ARI Heidelberg a full mirror (ESA has been
# observed to reset connections during outages)
GAIA_TAP_URLS = [
    'https://gea.esac.esa.int/tap-server/tap/sync',
    'https://gaia.ari.uni-heidelberg.de/tap/sync',
]

GAIA_ADQL = (
    'SELECT source_id, ra, dec, pmra, pmdec, parallax, '
    'phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe '
    'FROM gaiadr3.gaia_source '
    "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
    "CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {rad:.4f})) "
    'AND phot_g_mean_mag < {gmax}'
)


def fetch_gaia(wcs, bbox, gmax=GMAX):
    """
    Fetch the Gaia DR3 stars of a patch from the TAP service.

    A sync query (a few seconds) of the circle centered on the patch,
    of corner radius plus a margin for the off-patch intruders, tried
    on each of GAIA_TAP_URLS in turn.

    Parameters
    ----------
    wcs: ButlerWcs or FileWcs
        For the patch's sky position
    bbox: box
        The patch's bounding box
    gmax: float, optional
        Stars brighter than this; default GMAX

    Returns
    -------
    gaia: structured array
        ra, dec, pmra, pmdec, phot_g_mean_mag, ruwe, sorted by G
    """
    import io
    import urllib.request
    import urllib.parse

    xmid = 0.5 * (bbox.x.start + bbox.x.stop)
    ymid = 0.5 * (bbox.y.start + bbox.y.stop)
    ctr = wcs.pixelToSky(xmid, ymid)
    corner = wcs.pixelToSky(
        float(bbox.x.start), float(bbox.y.start),
    )
    rad = ctr.separation(corner).asDegrees() + 0.02

    query = GAIA_ADQL.format(
        ra=ctr.getRa().asDegrees(),
        dec=ctr.getDec().asDegrees(),
        rad=rad,
        gmax=gmax,
    )
    print(query)
    data = urllib.parse.urlencode({
        'REQUEST': 'doQuery',
        'LANG': 'ADQL',
        'FORMAT': 'csv',
        'QUERY': query,
    }).encode()

    text = None
    errors = []
    for url in GAIA_TAP_URLS:
        try:
            with urllib.request.urlopen(
                url, data=data, timeout=120,
            ) as resp:
                text = resp.read().decode()
            if not text.startswith('source_id'):
                raise RuntimeError(
                    'unexpected TAP response: ' + text[:200],
                )
            break
        except (OSError, RuntimeError, UnicodeDecodeError) as err:
            # OSError covers the whole network family (URLError,
            # HTTPError, connection resets, timeouts, ssl);
            # RuntimeError is our own unexpected-response check,
            # so an error page from one mirror falls through to
            # the next
            print(f'    gaia query failed at {url}: {err}')
            errors.append(f'{url}: {err}')
            text = None
    if text is None:
        raise RuntimeError(
            'all gaia TAP services failed:\n    '
            + '\n    '.join(errors)
        )
    gaia = np.genfromtxt(
        io.StringIO(text), delimiter=',', names=True,
    )
    print(f'    gaia: {gaia.size} stars')
    return gaia


def read_gaia_file(fname, wcs, bbox, gmax=GMAX):
    """
    Read the Gaia stars of a patch from a file.

    Converted to the fetch_gaia layout so everything downstream is
    unchanged; the same circle as the TAP query is applied, plus the
    gmax cut.  FITS files (lsst-starsub-make-gaia output, or any with
    a table in the first extension) are read with rustfits, parquet
    files with pandas.  Required columns are ra, dec (degrees) and a
    G magnitude, phot_g_mean_mag or gaia_g_mag.  Proper motions pmra,
    pmdec (mas/yr, pmra including cos(dec)) are used when present,
    else set to zero and the positions used as given.  ruwe is not
    carried by the files and is set to 1 (the template
    astrometric-quality guard passes everything).

    Parameters
    ----------
    fname: str
        The file
    wcs: ButlerWcs or FileWcs
        For the patch's sky position
    bbox: box
        The patch's bounding box
    gmax: float, optional
        Stars brighter than this; default GMAX

    Returns
    -------
    gaia: structured array
        As fetch_gaia
    """
    if fname.endswith('.parq') or fname.endswith('.parquet'):
        import pandas as pd
        data = pd.read_parquet(fname)
        columns = list(data.columns)
    else:
        import rustfits
        data = rustfits.read(fname)
        columns = list(data.dtype.names)

    if 'phot_g_mean_mag' in columns:
        gmag = data['phot_g_mean_mag']
    else:
        gmag = data['gaia_g_mag']

    if 'pmra' in columns and 'pmdec' in columns:
        pmra = data['pmra']
        pmdec = data['pmdec']
    else:
        pmra = None
        pmdec = None

    gaia = gaia_from_columns(
        ra=data['ra'],
        dec=data['dec'],
        gmag=gmag,
        wcs=wcs,
        bbox=bbox,
        gmax=gmax,
        pmra=pmra,
        pmdec=pmdec,
    )
    print(f'    gaia from {fname}: {gaia.size} stars')
    return gaia


def gaia_from_columns(
    ra, dec, gmag, wcs, bbox, gmax=GMAX, pmra=None, pmdec=None,
):
    """
    Build the fetch_gaia layout from plain position and magnitude arrays.

    The same patch circle as the TAP query and the gmax cut are
    applied.

    Parameters
    ----------
    ra, dec: arrays
        Degrees
    gmag: array
        Gaia G
    wcs: ButlerWcs or FileWcs
        For the patch's sky position
    bbox: box
        The patch's bounding box
    gmax: float, optional
        Stars brighter than this; default GMAX
    pmra, pmdec: arrays, optional
        Proper motions, mas/yr, pmra including cos(dec); zero when
        absent

    Returns
    -------
    gaia: structured array
        As fetch_gaia
    """
    xmid = 0.5 * (bbox.x.start + bbox.x.stop)
    ymid = 0.5 * (bbox.y.start + bbox.y.stop)
    ctr = wcs.pixelToSky(xmid, ymid)
    corner = wcs.pixelToSky(
        float(bbox.x.start), float(bbox.y.start),
    )
    rad = ctr.separation(corner).asDegrees() + 0.02

    ra = np.asarray(ra, dtype='f8')
    dec = np.asarray(dec, dtype='f8')
    gmag = np.asarray(gmag, dtype='f8')

    ra0 = np.deg2rad(ctr.getRa().asDegrees())
    dec0 = np.deg2rad(ctr.getDec().asDegrees())
    rar = np.deg2rad(ra)
    decr = np.deg2rad(dec)
    cossep = (
        np.sin(dec0) * np.sin(decr)
        + np.cos(dec0) * np.cos(decr) * np.cos(rar - ra0)
    )
    sep = np.rad2deg(np.arccos(np.clip(cossep, -1, 1)))

    w, = np.where((sep <= rad) & (gmag < gmax))
    gaia = np.zeros(w.size, dtype=[
        ('ra', 'f8'), ('dec', 'f8'),
        ('pmra', 'f8'), ('pmdec', 'f8'),
        ('phot_g_mean_mag', 'f8'), ('ruwe', 'f8'),
    ])
    gaia['ra'] = ra[w]
    gaia['dec'] = dec[w]
    gaia['phot_g_mean_mag'] = gmag[w]
    if pmra is not None:
        gaia['pmra'] = np.asarray(pmra, dtype='f8')[w]
        gaia['pmdec'] = np.asarray(pmdec, dtype='f8')[w]
    gaia['ruwe'] = 1.0
    return gaia


def gaia_pixel_positions(gaia, wcs, bbox):
    """
    Get the stars' patch-frame pixel positions at the observation epoch.

    The proper motions are propagated from GAIA_EPOCH to OBS_EPOCH.

    Parameters
    ----------
    gaia: structured array
        The extract (fetch_gaia layout)
    wcs: ButlerWcs or FileWcs
        The image's wcs
    bbox: box
        The image's bounding box

    Returns
    -------
    x, y: arrays
        Pixel positions relative to the box origin
    """
    dt = OBS_EPOCH - GAIA_EPOCH
    pmra = np.nan_to_num(gaia['pmra'])
    pmdec = np.nan_to_num(gaia['pmdec'])
    cosd = np.cos(np.deg2rad(gaia['dec']))
    ra = gaia['ra'] + dt * pmra / 3.6e6 / cosd
    dec = gaia['dec'] + dt * pmdec / 3.6e6

    x, y = wcs.skyToPixelArray(ra, dec, degrees=True)
    return x - bbox.x.start, y - bbox.y.start


def fetch_gaia_or_none(wcs, bbox, gmax=GMAX, require=False):
    """
    Fetch the Gaia stars of a patch, or report the failure.

    Parameters
    ----------
    wcs, bbox, gmax:
        As fetch_gaia
    require: bool, optional
        Raise on failure (never proceed silently without stars when
        the caller asked for star handling); otherwise the error is
        printed and None returned so the caller can degrade
        gracefully

    Returns
    -------
    gaia: structured array or None
    """
    try:
        return fetch_gaia(wcs, bbox, gmax=gmax)
    except RuntimeError as err:
        # the only expected failure: all mirrors exhausted
        # (per-mirror errors are consumed inside fetch_gaia)
        if require:
            raise RuntimeError(
                'gaia download failed and star subtraction '
                'was requested'
            ) from err
        print('    gaia download failed:', err)
        return None
