"""
the mask of the joint fit's diffuse regions: the union over the
bands, grown by the margin
"""
import numpy as np

from lsst_starsub.census import diffuse_mask


def test_diffuse_mask_union_and_margin():
    shape = (200, 300)
    r = np.zeros(shape, dtype=bool)
    r[50:60, 50:60] = True
    i = np.zeros(shape, dtype=bool)
    i[150:155, 250:255] = True
    fits = {
        'r': {'diffuse': r},
        'i': {'diffuse': i},
        'z': {'diffuse': np.zeros(shape, dtype=bool)},
    }
    m = diffuse_mask(fits, 20)
    assert m[r].all() and m[i].all()
    # grown by 20 px and no further
    assert m[55, 79] and not m[55, 80]
    assert m[30, 55] and not m[29, 55]
    assert np.array_equal(diffuse_mask(fits, 0), r | i)


def test_diffuse_mask_none():
    empty = {'r': {'diffuse': np.zeros((50, 50), dtype=bool)}}
    assert diffuse_mask(empty, 20) is None
    # fit dicts from an lsst_starsub without the diffuse region
    assert diffuse_mask({'r': {}}, 20) is None


def test_select_stars_intruders():
    from lsst_starsub.census import (
        GSAT, STAR_MARGIN, circle_radius, select_stars,
    )

    n = 400
    mask0 = np.zeros((n, n), dtype='i4')
    gaia = np.zeros(4, dtype=[
        ('ra', 'f8'), ('dec', 'f8'), ('phot_g_mean_mag', 'f8'), ('ruwe', 'f8'),
    ])
    gaia['ra'] = [1.0, 1.1, 1.2, 1.3]
    gaia['dec'] = 0.0
    gaia['ruwe'] = 1.0
    # on image; bright just off the edge; G 16 just off the edge;
    # G 16 far off
    gaia['phot_g_mean_mag'] = [17.0, 12.0, 16.0, 16.0]
    x = np.array([100.0, n + 50.0, n + 30.0, n + 150.0])
    y = np.array([100.0, 100.0, 200.0, 200.0])

    stars = select_stars(gaia, x, y, mask0, gsub=19.0, verbose=False)
    # the default rule: intruders brighter than GSAT within STAR_MARGIN
    # (the census is sorted brightest first)
    assert stars['on_image'].tolist() == [0, 1]
    assert 50.0 < STAR_MARGIN and 12.0 < GSAT

    stars = select_stars(
        gaia, x, y, mask0, gsub=19.0, verbose=False, intruder_gmax=17.0,
        intruder_margin=lambda g: 3 * circle_radius(g),
    )
    # 3 mask radii of a G 16 star is 103 px: the one at 30 px is in,
    # the one at 150 px is out
    assert stars['on_image'].tolist() == [0, 0, 1]
    assert sorted(stars['G'][stars['on_image'] == 0].tolist()) == [12.0, 16.0]


def test_select_stars_saturation_from_mask():
    from lsst_starsub.census import GSAT, select_stars
    from lsst_starsub.maskbits import DM_SAT

    # at good seeing stars a magnitude fainter than GSAT saturate: the
    # flag comes from the mask, with a tighter test for the fainter
    # stars so a neighbor's bleed trail does not flag them
    n = 400
    mask0 = np.zeros((n, n), dtype='i4')
    gaia = np.zeros(3, dtype=[
        ('ra', 'f8'), ('dec', 'f8'), ('phot_g_mean_mag', 'f8'), ('ruwe', 'f8'),
    ])
    gaia['ra'] = [1.0, 1.1, 1.2]
    gaia['dec'] = 0.0
    gaia['ruwe'] = 1.0
    gaia['phot_g_mean_mag'] = [GSAT + 1.0, GSAT + 1.0, GSAT + 1.0]
    x = np.array([100.0, 200.0, 300.0])
    y = np.array([100.0, 200.0, 300.0])
    # saturated pixels at the first star's center, 4 px from the
    # second's (a trail passing by), none at the third
    mask0[100, 100] |= DM_SAT
    mask0[204, 200] |= DM_SAT

    stars = select_stars(gaia, x, y, mask0, gsub=19.0, verbose=False)
    by_x = {int(s['x']): int(s['is_sat']) for s in stars}
    assert by_x == {100: 1, 200: 0, 300: 0}

    # the coadd rule: the saturation test only for the stars brighter
    # than sat_gmax, so the census does not depend on the band's
    # saturation bits; the faint saturated star is in the census by
    # its magnitude (gsub) but not flagged, and drops out when it is
    # fainter than gsub
    stars = select_stars(gaia, x, y, mask0, gsub=19.0, verbose=False,
                         sat_gmax=GSAT + 0.5)
    by_x = {int(s['x']): int(s['is_sat']) for s in stars}
    assert by_x == {100: 0, 200: 0, 300: 0}
    stars = select_stars(gaia, x, y, mask0, gsub=GSAT, verbose=False,
                         sat_gmax=GSAT + 0.5)
    assert stars.size == 0
    stars = select_stars(gaia, x, y, mask0, gsub=GSAT, verbose=False)
    assert stars.size == 1 and int(stars['x'][0]) == 100
