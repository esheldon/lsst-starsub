"""
Gaia DR3 extracts for a visit.

A visit's detectors span ten or so tracts, so the pooled per-visit
template has its own extract, made from the same refcat shards and
conversion as the per-tract files (lsst_starsub.cli.make_gaia) over
the bounding circle of the visit's detectors.
"""
import os

import numpy as np


VISIT_GAIA_PATTERN = 'gaia-dr3-visit-{visit}.fits'
VISIT_MARGIN_DEG = 0.05
DEFAULT_GMAX = 21.0


def visit_gaia_path(gaia_dir, visit):
    """
    Get the path of a visit's Gaia file.

    Parameters
    ----------
    gaia_dir: str
    visit: int

    Returns
    -------
    path: str
    """
    return os.path.join(gaia_dir, VISIT_GAIA_PATTERN.format(visit=int(visit)))


def visit_circle(butler, visit):
    """
    Get the bounding circle of a visit's detectors.

    Over the detectors with a wcs in the visit summary, grown by
    VISIT_MARGIN_DEG.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    visit: int

    Returns
    -------
    circle: lsst.sphgeom.Circle
    """
    import lsst.sphgeom as sphgeom
    from ..site import INSTRUMENT

    cat = butler.get(
        'visit_summary', dataId=dict(instrument=INSTRUMENT, visit=int(visit)),
    )

    vecs = []
    for rec in cat:
        wcs = rec.getWcs()
        if wcs is None:
            continue
        bbox = rec.getBBox()
        for corner in bbox.getCorners():
            c = wcs.pixelToSky(float(corner.x), float(corner.y))
            v = c.getVector()
            vecs.append([v.x(), v.y(), v.z()])

    vecs = np.array(vecs)

    if vecs.size == 0:
        raise RuntimeError(f'visit {visit}: no detector has a wcs')

    mean = vecs.mean(axis=0)
    mean /= np.linalg.norm(mean)
    cosang = np.clip(vecs @ mean, -1, 1)
    radius = np.rad2deg(np.arccos(cosang).max()) + VISIT_MARGIN_DEG
    center = sphgeom.UnitVector3d(float(mean[0]), float(mean[1]),
                                  float(mean[2]))

    return sphgeom.Circle(center, sphgeom.Angle.fromDegrees(radius))


def make_visit_gaia_file(butler, visit, outfile, gmax=DEFAULT_GMAX):
    """
    Write a visit's Gaia file.

    The stars in the visit's bounding circle brighter than gmax, in
    the lsst-starsub-make-gaia layout.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    visit: int
    outfile: str
    gmax: float, optional
        Default DEFAULT_GMAX

    Returns
    -------
    outfile: str
    """
    import rustfits
    from ..cli.make_gaia import REFCAT, convert_shard, get_shard_ids, in_circle

    circle = visit_circle(butler, visit)
    shard_ids = get_shard_ids(circle)
    parts = [
        convert_shard(butler.get(REFCAT, htm7=shard_id))
        for shard_id in shard_ids
    ]
    stars = np.concatenate(parts)
    keep = (
        in_circle(circle, stars['ra'], stars['dec'])
        & np.isfinite(stars['phot_g_mean_mag'])
        & (stars['phot_g_mean_mag'] < gmax)
    )
    stars = stars[keep]
    stars = stars[np.argsort(stars['source_id'], kind='stable')]
    print(
        f'    visit {visit}: circle radius '
        f'{circle.getOpeningAngle().asDegrees():.2f} deg, '
        f'{len(shard_ids)} shards, {stars.size} stars to G < {gmax:g}, '
        f'writing {outfile}'
    )
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    tmpfile = outfile + '.tmp'
    rustfits.write(tmpfile, stars, extname='gaia')
    os.replace(tmpfile, outfile)

    return outfile


def ensure_visit_gaia_file(butler, visit, gaia_dir, gmax=DEFAULT_GMAX):
    """
    Get the path of a visit's Gaia file, making it if missing.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    visit: int
    gaia_dir: str
    gmax: float, optional
        Default DEFAULT_GMAX

    Returns
    -------
    path: str
    """
    path = visit_gaia_path(gaia_dir, visit)
    if not os.path.exists(path):
        make_visit_gaia_file(butler, visit, path, gmax=gmax)
    return path
