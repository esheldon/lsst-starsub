"""
The star subtraction of a patch coadd as a library call.

lsst_mdet's pipeline calls handle_stars_joint in place of the stamp
templates' handle_stars (lsst_starsub.stamps; option --starsub-method
joint); the contract is the same, so the two routes can be compared.
The census, star masks and output table are the shared ones
(lsst_starsub.census); the sky and the stars are the joint fit of
lsst_starsub.joint.

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

# stars below the census depth (gsub) down to WING_GMAX get only their
# predicted wing subtracted, tapered in over WING_RIN-WING_ROUT px from
# the star, and are not masked, so their cores stay in the image for the
# detection like any star; their measured fluxes and sizes are unchanged
# (NONPOS_SIZE 76 -> 79 percent of them).  Without it the joint route
# leaves the light of these wings on nearby galaxies: in the injection
# test near G 19-21 stars (run-dp2-test-nearstar-inject, 2026-09-13) the
# typical shear galaxy (i 22.5-23.5) within 60 px came out +2.4, +1.2,
# +1.7 percent bright in r, i, z, and +1.0, 0.0, +0.5 with it.  The
# 8-16 px taper leaves the least close in, in typical fields; in crowded
# ones it over-subtracts by ~1.5 percent at 12-25 px.  The caller must
# supply the Gaia stars to this depth.  None: off.  Read at call time
WING_GMAX = 21.0
WING_RIN = 8.0    # px
WING_ROUT = 16.0  # px
# px: with this set, each of those stars' wing amplitude comes from its
# core in the band (core_amplitudes), not the Gaia prediction, which
# scatters by ~30 percent with the star's color: the faint stars are
# redder than the calibration's, median core amplitudes r 0.87, i 1.2, z
# 1.6 (2026-09-13).  None: the prediction
WING_CORE_RAP = 5.0
# the core amplitudes outside this range (a neighbor in the aperture, a
# star moved off its Gaia position) fall back to the prediction
WING_AMP_RANGE = (0.25, 4.0)


def wing_taper(r, rin=None, rout=None):
    """
    Return the radial taper that keeps a star's wing and drops its core.

    A smooth step, 0 inside rin and 1 beyond rout (the cumulative
    triweight of lsst_starsub.census.taper_from_distance).

    Parameters
    ----------
    r: array
        Radii in px
    rin, rout: float, optional
        The inner and outer radii in px; default WING_RIN, WING_ROUT

    Returns
    -------
    taper: array
    """
    from ..census import taper_from_distance

    rin = WING_RIN if rin is None else rin
    rout = WING_ROUT if rout is None else rout
    return taper_from_distance(
        np.maximum(np.asarray(r) - rin, 0.0),
        rout - rin,
    )


def core_amplitudes(image, good, x, y, G, wing, rap):
    """
    Measure each star's wing amplitude from its core.

    lsst_starsub.wing.core_amplitudes with the WING_AMP_RANGE guard:
    the flux within rap px over the model's, 1 where the aperture
    leaves the image or holds a pixel that is not good, or where the
    ratio falls outside WING_AMP_RANGE.

    Parameters
    ----------
    image: array
        The image, the sky and the census stars subtracted
    good: bool array
        The usable pixels
    x, y, G: arrays
        The stars' patch-frame positions and Gaia G
    wing: WingModel
        The full wing model, core included
    rap: float
        The aperture radius in px

    Returns
    -------
    amps: array
        The amplitudes, 1 the prediction
    """
    from ..wing import core_amplitudes as measure

    amps, _, _ = measure(image, good, x, y, G, wing, rap,
                         amp_range=WING_AMP_RANGE)
    return amps


def subtract_faint_wings(
    image, gaia, x, y, wing, gsub, good=None,
    verbose=True,
):
    """
    Subtract the predicted wings of the stars below the census depth.

    The on-image stars with gsub <= G < WING_GMAX, with the core
    tapered away (wing_taper), in place; nothing when WING_GMAX is None.
    The amplitude is the prediction (1, as for the pinned census
    stars), or with WING_CORE_RAP set each star's own from its core
    (core_amplitudes).

    Parameters
    ----------
    image: array
        The image, modified in place
    gaia: array with fields
        The gaia extract (lsst_starsub.gaia), to WING_GMAX
    x, y: arrays
        The stars' patch-frame positions
    wing: WingModel
        The full wing model
    gsub: float
        The census depth
    good: bool array, optional
        The usable pixels, for the core amplitudes; default all
    verbose: bool, optional
        Print the number subtracted

    Returns
    -------
    nwing: int
        The number of stars whose wings were subtracted
    """
    from ..wing import render_canonical_stars
    from ..wing import WingModel

    if WING_GMAX is None or not WING_GMAX > gsub:
        return 0

    G = np.asarray(gaia['phot_g_mean_mag'], dtype='f8')
    ny, nx = image.shape

    sel = ((G >= gsub) & (G < WING_GMAX)
           & (x >= 0) & (x < nx) & (y >= 0) & (y < ny))

    if not sel.any():
        return 0

    faint = np.zeros(sel.sum(), dtype=[('x', 'f8'), ('y', 'f8'), ('G', 'f8')])
    faint['x'], faint['y'], faint['G'] = x[sel], y[sel], G[sel]

    amps = None

    if WING_CORE_RAP is not None:
        if good is None:
            good = np.ones(image.shape, dtype=bool)
        amps = core_amplitudes(
            image, good, faint['x'], faint['y'],
            faint['G'], wing, WING_CORE_RAP,
        )

    # rendered as the full model less its core: the renderer ends each
    # star's window where the profile first falls below its floor, so a
    # profile rising from zero, the wing alone, would render nothing

    core = WingModel(wing.r, wing.T * (1.0 - wing_taper(wing.r)))
    model = render_canonical_stars(image.shape, faint, wing,
                                   gsub=WING_GMAX, amps=amps, verbose=False)
    model -= render_canonical_stars(image.shape, faint, core,
                                    gsub=WING_GMAX, amps=amps, verbose=False)
    image -= model

    if verbose:
        amp = (
            'the prediction' if amps is None else
            f'from the cores, median {np.median(amps):.2f}, '
            f'{np.mean(amps == 1.0) * 100:.0f} percent fallback'
        )
        print(
            f'    wings of {faint.size} stars of G {gsub:g}-{WING_GMAX:g} '
            f'subtracted (cores kept, not masked; amplitude {amp}), max '
            f'{model.max():.2f} nJy'
        )

    return int(faint.size)


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
    wing: WingModel
        With the file name in its fname attribute
    """
    from ..wing import read_wing_model

    wing = read_wing_model(fname)
    wing.fname = fname
    return wing


def handle_stars_joint(
    deep_coadd, wcs, gaia, wing, gsub=None,
    spacing=None, prior=None, detect_settings=None,
    verbose=True,
):
    """
    Subtract the stars and the sky of a patch coadd with the joint fit.

    The stored 'object' background is undone first
    (apply_background(None)), leaving the image with only the initial
    background subtracted, the one determined without masking
    objects; the mesh then takes the place of the 'object' model.
    The sky mesh and the star model are subtracted in place, and with
    WING_GMAX set the wings of the fainter stars (subtract_faint_wings).
    Returns what lsst_starsub.stamps.handle_stars returns, plus the fit.

    Parameters
    ----------
    deep_coadd: deep_coadd
        The coadd as lsst_mdet loads it (the 'object' background
        applied); its image is modified in place
    wcs: ButlerWcs or FileWcs
        For the gaia pixel positions
    gaia: array with fields
        The gaia extract (lsst_starsub.gaia)
    wing: WingModel
        From load_wing
    gsub: float, optional
        Census depth; default lsst_starsub.census.GSUB
    spacing: float, optional
        The sky mesh node spacing in pixels; default
        lsst_starsub.joint.SPACING
    prior: float, optional
        The amplitude prior width about the prediction; default
        lsst_starsub.joint.PRIOR_SIGMA
    detect_settings: dict, optional
        The detection settings of the fit's source segmentation;
        lsst_mdet passes metadetection's.  Default
        lsst_starsub.joint.DETECT_SETTINGS
    verbose: bool, optional
        Print the census and fit summaries

    Returns
    -------
    starmask, star_table, dstar, fit:
        The bool star mask (the circles alone; the caller applies
        the taper and masks dstar < census.APOD_STARS, see the
        census module), the census table with the fitted
        amplitudes in 'A' (1 = the prediction), the distance
        transform off the mask, and the fit dict: the census
        (stars), A, free, nodes, node_values, spacing, prior, gfit,
        gsub, chi2, ncell, sky_sigma, shape, bg_restored and the
        wing, for make_fit_tables; diffuse, the bool mask of the
        large diffuse segments left to the sky fit (joint_fit), for
        the caller to mask; nwing, the number of fainter stars
        whose wings were subtracted, and the settings of that step
        (wing_gmax, wing_rin, wing_rout, wing_core_rap)
    """
    from ..census import GSUB, make_star_table, patch_census
    from ..joint import GFIT, PRIOR_SIGMA, SPACING, joint_fit
    from ..maskbits import DM_NO_DATA

    if gsub is None:
        gsub = GSUB
    if spacing is None:
        spacing = SPACING
    if prior is None:
        prior = PRIOR_SIGMA

    mask0 = deep_coadd.mask.array[:, :, 0]
    stars, starmask, _, dstar, x, y = patch_census(
        gaia, wcs, deep_coadd.bbox, mask0, gsub=gsub, coadd=True,
        verbose=verbose,
    )

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
        detect_settings=detect_settings, verbose=verbose,
    )

    image -= jf['sky']
    image -= jf['star_model']
    nwing = subtract_faint_wings(image, gaia, x, y, wing, gsub, good=good,
                                 verbose=verbose)
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
        A_err=jf['A_err'],
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
        diffuse=jf['diffuse'],
        nwing=nwing,
        wing_gmax=WING_GMAX,
        wing_rin=WING_RIN,
        wing_rout=WING_ROUT,
        wing_core_rap=WING_CORE_RAP,
    )
    return starmask, star_table, dstar, fit


def make_fit_tables(fits):
    """
    Turn the per-band fits into the four output tables.

    Parameters
    ----------
    fits: dict
        band -> the fit dict from handle_stars_joint; the census
        must be the same in every band (it is: the census
        depends on the gaia extract and the patch alone)

    Returns
    -------
    dict of extname -> structured array, for the extensions
    META_EXT (one row per band: image shape, mesh geometry, the
    fit settings and quality, the background restored, the wing
    file, and the faint-star wing step's settings and number of
    stars, NaN for a setting that was None), STARS_EXT (one row per
    census star with the per-band amplitude A_{band} and
    free_{band}), SKY_EXT (one
    row per mesh node per band) and WING_EXT (one row per radius
    per band of the wing profile, nJy per unit Gaia flux)
    """
    from ..wing import profile_of

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
        ('wing_gmax', 'f4'), ('wing_rin', 'f4'), ('wing_rout', 'f4'),
        ('wing_core_rap', 'f4'), ('nwing', 'i4'),
    ])

    def setting(fit, name):
        """A setting of the fit dict, NaN where missing or None."""
        v = fit.get(name)
        return np.nan if v is None else v

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
            setting(fit, 'wing_gmax'), setting(fit, 'wing_rin'),
            setting(fit, 'wing_rout'), setting(fit, 'wing_core_rap'),
            fit.get('nwing', 0),
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

        r, T = profile_of(fit['wing'])
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

    Parameters
    ----------
    fname: str
        The output file

    Returns
    -------
    tables: dict
        extname -> structured array, as make_fit_tables
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
    star taper), less the wings of the stars below the census
    depth (subtract_faint_wings, nwing of them in the meta table),
    which are not stored and so not rebuilt here.

    Parameters
    ----------
    tables: dict
        From make_fit_tables or read_fit_tables
    band: str

    Returns
    -------
    sky, stars: (ny, nx) f4 images in nJy
    """
    from ..joint import render_mesh
    from ..wing import render_canonical_stars
    from ..wing import WingModel

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
        shape, stars, wing, amps=stars[f'A_{band}'],
        verbose=False,
    )

    return sky, star_model
