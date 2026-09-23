"""
the large-galaxy mask: the catalog read for a patch, the match of a
catalog galaxy to the joint fit's large sources, the mask ellipse
"""
import numpy as np
import pytest

from lsst_starsub import galaxies as gmod
from lsst_starsub.geom import SimpleBox


class TangentWcs:
    """
    a flat tangent-plane wcs about (ra0, dec0) at PIXEL_SCALE, ra
    increasing with x (enough for the reader and the positions)
    """
    def __init__(self, ra0=10.0, dec0=-20.0):
        self.ra0, self.dec0 = ra0, dec0
        self.scale = gmod.PIXEL_SCALE / 3600

    def pixelToSkyArray(self, x, y, degrees=True):
        x, y = np.asarray(x, dtype='f8'), np.asarray(y, dtype='f8')
        dec = self.dec0 + y * self.scale
        ra = self.ra0 + x * self.scale / np.cos(np.deg2rad(self.dec0))
        return ra, dec

    def skyToPixelArray(self, ra, dec, degrees=True):
        ra, dec = np.asarray(ra, dtype='f8'), np.asarray(dec, dtype='f8')
        y = (dec - self.dec0) / self.scale
        x = (ra - self.ra0) * np.cos(np.deg2rad(self.dec0)) / self.scale
        return x, y


def big_table(rows):
    return np.array(rows, dtype=[
        ('x', 'f8'), ('y', 'f8'), ('a', 'f8'), ('b', 'f8'),
        ('theta', 'f8'), ('npix', 'i8'),
    ])


def test_read_galaxy_file(tmp_path):
    import rustfits

    wcs = TangentWcs()
    bbox = SimpleBox(1000, 5000, 1000, 5000)
    # the patch center and a corner on the sky
    ra_c, dec_c = wcs.pixelToSkyArray([3000.0], [3000.0])
    ra_far, dec_far = wcs.pixelToSkyArray([9000.0], [3000.0])
    cat = np.zeros(4, dtype=[('pgc', 'i8'), ('ra', 'f8'), ('dec', 'f8'),
                             ('logd25', 'f4'), ('logr25', 'f4'), ('pa', 'f4'),
                             ('name', 'U24')])
    cat['pgc'] = [1, 2, 3, 4]
    cat['ra'] = [ra_c[0], ra_c[0], ra_far[0], ra_far[0]]
    cat['dec'] = [dec_c[0], dec_c[0], dec_far[0], dec_far[0]]
    # 1: in the patch, big; 2: in the patch, too small; 3: 20' off a
    # 13.3' box, D25 1': out; 4: the same but D25 30': its light
    # reaches the patch
    cat['logd25'] = np.log10([10.0, 3.0, 10.0, 300.0])
    cat['name'] = ['a', 'b', 'c', 'd']
    fname = str(tmp_path / 'gals.fits')
    rustfits.write(fname, cat, extname='leda')

    gals = gmod.read_galaxy_file(fname, wcs, bbox, d25min=0.5)
    assert sorted(gals['pgc']) == [1, 4]
    assert np.allclose(gals['d25_arcmin'][gals['pgc'] == 1], 1.0)
    assert gals['name'][gals['pgc'] == 4][0] == 'd'

    x, y = gmod.galaxy_pixel_positions(gals, wcs, bbox)
    assert np.allclose(x[gals['pgc'] == 1], 2000.0)
    assert np.allclose(y[gals['pgc'] == 1], 2000.0)


def test_match_and_ellipse():
    big = big_table([(100.0, 100.0, 20.0, 10.0, 0.0, 3000),
                     (400.0, 400.0, 30.0, 30.0, 0.0, 6000)])
    # within the floor radius
    assert gmod.match_big_source(big, 110.0, 95.0, 1.0) == 0
    # too far for a 1' galaxy (match radius max(25, 75) = 75 px)
    assert gmod.match_big_source(big, 250.0, 100.0, 1.0) is None
    # a larger catalog size widens the radius
    assert gmod.match_big_source(big, 250.0, 100.0, 5.0) == 0
    assert gmod.match_big_source(None, 100.0, 100.0, 1.0) is None
    assert gmod.match_big_source(big[:0], 100.0, 100.0, 1.0) is None

    # a merged or clipped segment: the centroid is far from the
    # catalog position but the isophotal ellipse (a_iso 226 px along
    # x, 56 along y) contains it
    merged = big_table([(500.0, 500.0, 80.0, 20.0, 0.0, 40000)])
    assert gmod.in_source_ellipse(merged[0], 700.0, 500.0)
    assert not gmod.in_source_ellipse(merged[0], 500.0, 700.0)
    assert gmod.match_big_source(merged, 700.0, 500.0, 1.0) == 0
    assert gmod.match_big_source(merged, 500.0, 700.0, 1.0) is None

    a, b, theta = gmod.source_ellipse(big[0], scale=2.0, rmax=1e9)
    riso = np.sqrt(3000 / np.pi)
    assert np.isclose(a, 2.0 * riso / np.sqrt(0.5))
    assert np.isclose(b, 0.5 * a)
    assert theta == 0.0
    # the cap
    a, b, _ = gmod.source_ellipse(big[0], scale=2.0, rmax=30.0)
    assert a == 30.0 and np.isclose(b, 15.0)


def test_galaxy_mask():
    shape = (500, 500)
    fits = {
        'r': {'big_sources': big_table(
            [(100.0, 100.0, 20.0, 10.0, 0.0, 3000)])},
        'i': {'big_sources': big_table(
            [(102.0, 99.0, 20.0, 10.0, 0.0, 3200),
             (400.0, 400.0, 30.0, 30.0, 0.0, 6000)])},
        'z': {'big_sources': None},
    }
    gals = np.zeros(2, dtype=gmod.GALAXY_DTYPE)
    gals['pgc'] = [7, 8]
    # a small catalog size, so the D25 floor (75 px) is inside the
    # data ellipse
    gals['d25_arcmin'] = [0.5, 1.0]
    gals['logr25'] = np.nan
    gals['pa'] = np.nan
    x = np.array([101.0, 250.0])
    y = np.array([100.0, 250.0])

    mask, table = gmod.galaxy_mask(fits, gals, x, y, shape, scale=2.0)
    assert mask is not None and mask.dtype == bool
    # galaxy 7 matched in r and i, galaxy 8 (no source there) in none
    assert sorted(table['band']) == ['i', 'r']
    assert set(table['pgc']) == {7}
    # elongated along x (theta 0, q 0.5): the mask reaches farther in x
    a = 2.0 * np.sqrt(3200 / np.pi) / np.sqrt(0.5)
    assert mask[100, int(100 + 0.9 * a)]
    assert not mask[int(100 + 0.9 * a), 100]
    assert not mask[400, 400]
    # the unmatched galaxy gets neither a data ellipse nor the floor
    assert not mask[250, 250]

    # nothing matches: no mask
    mask, table = gmod.galaxy_mask(fits, gals[1:], x[1:], y[1:], shape)
    assert mask is None and table.size == 0


def test_galaxy_mask_floor():
    """
    a chopped segment: the data ellipse is small, the catalog D25
    ellipse takes over, with the catalog axis ratio and orientation
    """
    shape = (800, 800)
    # a tiny matched source: riso 31 px, round
    fits = {'i': {'big_sources': big_table(
        [(400.0, 400.0, 5.0, 5.0, 0.0, 3000)])}}
    gals = np.zeros(1, dtype=gmod.GALAXY_DTYPE)
    gals['pgc'] = 9
    gals['d25_arcmin'] = 2.0        # 300 px semi-major
    gals['logr25'] = np.log10(4.0)  # axis ratio 0.25
    gals['pa'] = 0.0                # major axis north-south, along y
    x, y = np.array([400.0]), np.array([400.0])

    a, b, theta = gmod.catalog_ellipse(gals[0])
    assert np.isclose(a, 300.0) and np.isclose(b, 75.0)
    assert np.isclose(theta, np.pi / 2)
    # the patch frame has east along -x: a galaxy at pa 45 (northeast)
    # runs from lower right to upper left, 135 degrees from +x; the
    # mirror image (45 degrees) was the bug through v0.3.0
    g45 = gals.copy()
    g45['pa'] = 45.0
    _, _, theta45 = gmod.catalog_ellipse(g45[0])
    assert np.isclose(theta45, np.deg2rad(135.0))
    m45 = gmod.catalog_exclusion(g45, x, y, shape, scale=1.0)
    assert m45[400 + 200, 400 - 200] and not m45[400 + 200, 400 + 200]

    mask, table = gmod.galaxy_mask(fits, gals, x, y, shape, scale=1.5)
    assert table.size == 1
    # along y the floor reaches 300 px, along x only 75
    assert mask[400 + 280, 400] and mask[400 - 280, 400]
    assert mask[400, 400 + 60]
    assert not mask[400, 400 + 120]
    assert not mask[400 + 320, 400]

    # pa 90: the major axis along x
    gals['pa'] = 90.0
    mask, _ = gmod.galaxy_mask(fits, gals, x, y, shape, scale=1.5)
    assert mask[400, 400 + 280] and not mask[400 + 120, 400]


def test_catalog_exclusion():
    """
    the sky-fit exclusion: every catalog galaxy's D25 ellipse scaled
    by GAL_SKY_SCALE, matched or not
    """
    gals = np.zeros(2, dtype=gmod.GALAXY_DTYPE)
    gals['pgc'] = [1, 2]
    gals['d25_arcmin'] = [1.0, 0.5]      # 150 and 75 px semi-major
    gals['logr25'] = [0.0, np.log10(2.0)]
    gals['pa'] = [np.nan, 90.0]          # round; major axis along x
    x, y = np.array([200.0, 600.0]), np.array([200.0, 600.0])

    excl = gmod.catalog_exclusion(gals, x, y, (800, 800), scale=2.0)
    assert excl is not None and excl.dtype == bool
    # round, 300 px radius
    assert excl[200, 200 + 290] and excl[200 + 290, 200]
    assert not excl[200, 200 + 310]
    # q 0.5 along y: 150 px along x, 75 along y
    assert excl[600, 600 + 140] and not excl[600, 600 + 160]
    assert excl[600 + 70, 600] and not excl[600 + 85, 600]

    assert gmod.catalog_exclusion(gals[:0], x[:0], y[:0], (10, 10)) is None
    assert gmod.catalog_exclusion(None, None, None, (10, 10)) is None


def test_galaxy_mask_needs_positions():
    with pytest.raises(TypeError):
        gmod.galaxy_mask({}, None, None, None)
