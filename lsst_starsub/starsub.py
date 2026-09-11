"""
The star subtraction of a patch coadd as a library call.

lsst_mdet's pipeline calls handle_stars_joint in place of its own
lsst_mdet.starsub.handle_stars (option --starsub-method joint); the
contract is the same, so the two routes can be compared.  The census,
star masks and output table are lsst_mdet's; the sky and the stars
are the joint fit of lsst_starsub.joint.

The fit is also returned in a form that make_fit_tables turns into
four small tables for the output file, from which render_fit
rebuilds the sky and star images that were subtracted.
"""
import numpy as np

# the output extensions of the fit tables
META_EXT = 'starsub_meta'
STARS_EXT = 'starsub_stars'
SKY_EXT = 'starsub_sky'
WING_EXT = 'starsub_wing'
FIT_EXTS = (META_EXT, STARS_EXT, SKY_EXT, WING_EXT)

# the census columns carried into the stars table
CENSUS_COLUMNS = ('ra', 'dec', 'x', 'y', 'G', 'ruwe', 'is_sat', 'on_image')


def load_wing(fname):
    """
    Load the per-band wing model.

    Parameters
    ----------
    fname: str
        The wing file, calibrated per band at S3DF with
        lsst-starsub-visit-template and checked across tracts
        by the broad calibration

    Returns
    -------
    WingModel, with the file name in its fname attribute
    """
    from .wing import read_wing_model

    wing = read_wing_model(fname)
    wing.fname = fname
    return wing


def handle_stars_joint(deep_coadd, wcs, gaia, wing, gsub=None,
                       spacing=None, prior=None, verbose=True):
    """
    Subtract the stars and the sky of a patch coadd with the joint fit.

    The stored object background is restored first, so the image holds
    the whole sky and the mesh replaces the background model; then the
    sky mesh and the star model are subtracted in place.  Returns what
    lsst_mdet.starsub.handle_stars returns, plus the fit.

    Parameters
    ----------
    deep_coadd: deep_coadd
        The coadd as lsst_mdet loads it (the 'object' background
        applied); its image is modified in place
    wcs: ButlerWcs or FileWcs
        For the gaia pixel positions
    gaia: array with fields
        The gaia extract (lsst_mdet.gaia)
    wing: WingModel
        From load_wing
    gsub: float, optional
        Census depth; default lsst_mdet.starsub.GSUB
    spacing: float, optional
        The sky mesh node spacing in pixels; default
        lsst_starsub.joint.SPACING
    prior: float, optional
        The amplitude prior width about the prediction; default
        lsst_starsub.joint.PRIOR_SIGMA
    verbose: bool, optional
        Print the census and fit summaries

    Returns
    -------
    starmask, star_table, dstar, fit:
        The bool star mask, the census table with the fitted
        amplitudes in 'A' (1 = the prediction), the distance
        transform off the mask, and the fit dict: the census
        (stars), A, free, nodes, node_values, spacing, prior, gfit,
        gsub, chi2, ncell, sky_sigma, shape, bg_restored and the
        wing, for make_fit_tables
    """
    from scipy import ndimage
    from lsst_mdet.defaults import DM_NO_DATA
    from lsst_mdet.gaia import gaia_pixel_positions
    from lsst_mdet.starsub import (
        GSUB, build_star_mask, make_star_table, select_stars,
    )
    from .joint import GFIT, PRIOR_SIGMA, SPACING, joint_fit

    if gsub is None:
        gsub = GSUB
    if spacing is None:
        spacing = SPACING
    if prior is None:
        prior = PRIOR_SIGMA

    mask0 = deep_coadd.mask.array[:, :, 0]
    x, y = gaia_pixel_positions(gaia, wcs, deep_coadd.bbox)
    stars = select_stars(gaia, x, y, mask0, gsub=gsub)
    starmask, _ = build_star_mask(stars, mask0, verbose=verbose,
                                  coadd=True)
    dstar = ndimage.distance_transform_edt(~starmask)

    apply = getattr(deep_coadd, 'apply_background', None)
    if apply is not None:
        apply(None)
        bg_restored = 'object'
    else:
        bg_restored = 'none'
        if verbose:
            print('    WARNING: no stored backgrounds to restore; the '
                  'mesh fits whatever sky the image carries')

    image = deep_coadd.image.array
    var = deep_coadd.variance.array
    good = (np.isfinite(var) & (var > 0) & ((mask0 & DM_NO_DATA) == 0)
            & ~starmask)
    sky_sigma = float(np.sqrt(np.median(var[good])))
    jf = joint_fit(
        image, good, stars, wing, sky_sigma,
        spacing=spacing, prior_sigma=prior, variance=var,
        verbose=verbose,
    )
    image -= jf['sky']
    image -= jf['star_model']
    star_table = make_star_table(stars, [])
    star_table['A'] = jf['A']
    if verbose:
        nfree = int(jf['free'].sum())
        print(f'    joint star model: {nfree} amplitudes fit, '
              f'{stars.size - nfree} pinned; sky mesh '
              f'{jf["node_values"].size} nodes; chi2/cell {jf["chi2"]:.2f}')

    fit = dict(
        stars=stars,
        A=jf['A'],
        free=jf['free'],
        nodes=jf['nodes'],
        node_values=jf['node_values'],
        spacing=spacing,
        prior=prior,
        gfit=GFIT,
        gsub=gsub,
        chi2=jf['chi2'],
        ncell=jf['ncell'],
        sky_sigma=sky_sigma,
        shape=image.shape,
        bg_restored=bg_restored,
        wing=wing,
    )
    return starmask, star_table, dstar, fit


def make_fit_tables(fits):
    """
    Turn the per-band fits into the four output tables.

    Parameters
    ----------
    fits: dict
        band -> the fit dict from handle_stars_joint; the census
        must be the same in every band (it is: lsst_mdet's
        census depends on the gaia extract and the patch alone)

    Returns
    -------
    dict of extname -> structured array, for the extensions
    META_EXT (one row per band: image shape, mesh geometry, the
    fit settings and quality, the background restored and the
    wing file), STARS_EXT (one row per census star with the
    per-band amplitude A_{band} and free_{band}), SKY_EXT (one
    row per mesh node per band) and WING_EXT (one row per radius
    per band of the wing profile, nJy per unit Gaia flux)
    """
    bands = list(fits.keys())
    first = fits[bands[0]]
    stars = first['stars']
    nstar = stars.size

    meta = np.zeros(len(bands), dtype=[
        ('band', 'U1'), ('nx', 'i4'), ('ny', 'i4'), ('spacing', 'f4'),
        ('nxnode', 'i2'), ('nynode', 'i2'), ('gsub', 'f4'), ('gfit', 'f4'),
        ('prior', 'f4'), ('nfree', 'i4'), ('npinned', 'i4'),
        ('ncell', 'i4'), ('chi2', 'f4'), ('sky_sigma', 'f4'),
        ('bg_restored', 'U16'), ('wing_file', 'U256'),
    ])

    star_dtype = [(name, stars.dtype[name]) for name in CENSUS_COLUMNS]
    for band in bands:
        star_dtype += [(f'A_{band}', 'f8'), (f'free_{band}', 'i2')]
    star_table = np.zeros(nstar, dtype=star_dtype)
    for name in CENSUS_COLUMNS:
        star_table[name] = stars[name]

    sky_rows = []
    wing_rows = []
    for i, band in enumerate(bands):
        fit = fits[band]
        if fit['stars'].size != nstar:
            raise ValueError(
                f'band {band} has {fit["stars"].size} census stars, '
                f'band {bands[0]} has {nstar}'
            )
        xn, yn = fit['nodes']
        ny, nx = fit['shape']
        meta[i] = (
            band, nx, ny, fit['spacing'], xn.size, yn.size, fit['gsub'],
            fit['gfit'], fit['prior'], int(fit['free'].sum()),
            int((~fit['free']).sum()), fit['ncell'], fit['chi2'],
            fit['sky_sigma'], fit['bg_restored'],
            getattr(fit['wing'], 'fname', ''),
        )
        star_table[f'A_{band}'] = fit['A']
        star_table[f'free_{band}'] = fit['free']

        iy, ix = np.mgrid[0:yn.size, 0:xn.size]
        sky = np.zeros(ix.size, dtype=[
            ('band', 'U1'), ('ix', 'i2'), ('iy', 'i2'), ('x', 'f4'),
            ('y', 'f4'), ('value', 'f4'),
        ])
        sky['band'] = band
        sky['ix'] = ix.ravel()
        sky['iy'] = iy.ravel()
        sky['x'] = xn[ix.ravel()]
        sky['y'] = yn[iy.ravel()]
        sky['value'] = np.asarray(fit['node_values']).reshape(
            yn.size, xn.size,
        ).ravel()
        sky_rows.append(sky)

        r, T = fit['wing']
        wing = np.zeros(r.size, dtype=[
            ('band', 'U1'), ('r', 'f4'), ('T', 'f4'),
        ])
        wing['band'] = band
        wing['r'] = r
        wing['T'] = T
        wing_rows.append(wing)

    return {
        META_EXT: meta,
        STARS_EXT: star_table,
        SKY_EXT: np.concatenate(sky_rows),
        WING_EXT: np.concatenate(wing_rows),
    }


def read_fit_tables(fname):
    """
    Read the fit tables from an output file.

    Returns
    -------
    dict of extname -> structured array, as make_fit_tables
    """
    import rustfits

    tables = {}
    with rustfits.FITS(fname) as fits:
        for ext in FIT_EXTS:
            tables[ext] = fits[ext].read()
    return tables


def render_fit(tables, band):
    """
    Rebuild the sky and star images subtracted from a band.

    The joint fit was made on the coadd with the background named
    in the meta table's bg_restored restored, so image_delivered
    + that background - sky - stars is the image the processing
    saw (before the background redo's noise calibration and the
    star taper).

    Parameters
    ----------
    tables: dict
        From make_fit_tables or read_fit_tables
    band: str

    Returns
    -------
    sky, stars: (ny, nx) f4 images in nJy
    """
    from .joint import render_mesh
    from .visit import render_canonical_stars
    from .wing import WingModel

    meta = tables[META_EXT]
    m = meta[meta['band'] == band]
    if m.size != 1:
        raise ValueError(f'band {band} not in the fit tables')
    m = m[0]
    shape = (int(m['ny']), int(m['nx']))
    spacing = float(m['spacing'])

    sky_rows = tables[SKY_EXT]
    sky_rows = sky_rows[sky_rows['band'] == band]
    values = np.zeros((int(m['nynode']), int(m['nxnode'])), dtype='f8')
    values[sky_rows['iy'], sky_rows['ix']] = sky_rows['value']
    xn = np.arange(0, shape[1] + spacing, spacing, dtype='f8')
    yn = np.arange(0, shape[0] + spacing, spacing, dtype='f8')
    sky = render_mesh((xn, yn), values, shape)

    wing_rows = tables[WING_EXT]
    wing_rows = wing_rows[wing_rows['band'] == band]
    wing = WingModel(wing_rows['r'], wing_rows['T'])

    stars = tables[STARS_EXT]
    star_model = render_canonical_stars(
        shape, stars, wing, gsub=99.0, amps=stars[f'A_{band}'],
        verbose=False,
    )
    return sky, star_model
