"""
The cell coadd as delivered.

The data id, the loader that gives the MultipleCellCoadd with the
object background restored, and the smooth-surface fit the
forward-model check uses.
"""
import numpy as np

from ..site import SKYMAP


def coadd_data_id(tract, patch, band):
    """
    Build the butler data id of a patch coadd.

    Parameters
    ----------
    tract, patch: int
    band: str

    Returns
    -------
    data_id: dict
    """
    return dict(band=band, skymap=SKYMAP, tract=tract, patch=patch)


def load_cell_coadd(butler, did):
    """
    Load the None-state cell coadd of a patch.

    deep_coadd_cell_predetection where present (the weekly runs),
    else DP2's deep_coadd, a lsst.images CellCoadd with the object
    background subtracted, restored (apply_background(None)) and
    converted to the legacy class (per-cell inputs with weights,
    stitch, grid, wcs).

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    did: dict
        The data id (coadd_data_id)

    Returns
    -------
    coadd: MultipleCellCoadd
    """
    try:
        have = bool(butler.exists('deep_coadd_cell_predetection', did))
    except Exception:
        have = False
    if have:
        return butler.get('deep_coadd_cell_predetection', dataId=did)
    cc = butler.get('deep_coadd', dataId=did)
    if not hasattr(cc, 'to_legacy_cell_coadd'):
        raise RuntimeError(
            'deep_coadd is not a cell coadd and '
            'deep_coadd_cell_predetection is absent'
        )
    print('    deep_coadd is a CellCoadd: restoring its object '
          'background and converting to the legacy class')
    cc.apply_background(None)
    return cc.to_legacy_cell_coadd()


# the detector-scale part of a background surface (level and
# gradients) is removed by a surface fit of this order in the
# forward-model check; 0 removes the mean only
SMOOTH_ORDER = 2


def fit_smooth_surface(model, order):
    """
    Fit a polynomial surface to an image and evaluate it on the grid.

    Parameters
    ----------
    model: array
        The image
    order: int
        The total order; 0 fits the mean alone

    Returns
    -------
    surface: array
        The least-squares surface, the shape of model
    """
    ny, nx = model.shape
    y, x = np.mgrid[0:ny, 0:nx]
    # scaled to [-1, 1] for conditioning
    xs = 2.0 * x / max(nx - 1, 1) - 1.0
    ys = 2.0 * y / max(ny - 1, 1) - 1.0
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((xs ** i * ys ** j).ravel())
    basis = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(basis, model.ravel(), rcond=None)
    return (basis @ coef).reshape(ny, nx)
