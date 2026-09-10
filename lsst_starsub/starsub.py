"""
The star subtraction of a patch coadd as a library call.

lsst_mdet's pipeline calls handle_stars_joint in place of its own
lsst_mdet.starsub.handle_stars (option --starsub-method joint); the
contract is the same, so the two routes can be compared.  The census,
star masks and output table are lsst_mdet's; the sky and the stars
are the joint fit of lsst_starsub.joint.
"""
import numpy as np


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
    WingModel
    """
    from .wing import read_wing_model

    return read_wing_model(fname)


def handle_stars_joint(deep_coadd, wcs, gaia, wing, gsub=None,
                       spacing=None, prior=None, verbose=True):
    """
    Subtract the stars and the sky of a patch coadd with the joint fit.

    The stored object background is restored first, so the image holds
    the whole sky and the mesh replaces the background model; then the
    sky mesh and the star model are subtracted in place.  Returns what
    lsst_mdet.starsub.handle_stars returns.

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
    starmask, star_table, dstar:
        The bool star mask, the census table with the fitted
        amplitudes in 'A' (1 = the prediction), and the distance
        transform off the mask
    """
    from scipy import ndimage
    from lsst_mdet.defaults import DM_NO_DATA
    from lsst_mdet.gaia import gaia_pixel_positions
    from lsst_mdet.starsub import (
        GSUB, build_star_mask, make_star_table, select_stars,
    )
    from .joint import PRIOR_SIGMA, SPACING, joint_fit

    if gsub is None:
        gsub = GSUB
    mask0 = deep_coadd.mask.array[:, :, 0]
    x, y = gaia_pixel_positions(gaia, wcs, deep_coadd.bbox)
    stars = select_stars(gaia, x, y, mask0, gsub=gsub)
    starmask, _ = build_star_mask(stars, mask0, verbose=verbose)
    dstar = ndimage.distance_transform_edt(~starmask)

    apply = getattr(deep_coadd, 'apply_background', None)
    if apply is not None:
        apply(None)
    elif verbose:
        print('    WARNING: no stored backgrounds to restore; the '
              'mesh fits whatever sky the image carries')

    image = deep_coadd.image.array
    var = deep_coadd.variance.array
    good = (np.isfinite(var) & (var > 0) & ((mask0 & DM_NO_DATA) == 0)
            & ~starmask)
    sky_sigma = float(np.sqrt(np.median(var[good])))
    jf = joint_fit(
        image, good, stars, wing, sky_sigma,
        spacing=SPACING if spacing is None else spacing,
        prior_sigma=PRIOR_SIGMA if prior is None else prior,
        variance=var, verbose=verbose,
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
    return starmask, star_table, dstar
