"""
the joint fit's sky mesh: the neighbor-difference operator and the
smoothness prior filling a node that has no data
"""
import numpy as np
import pytest

import lsst_starsub.joint as jmod
from lsst_starsub.wing import render_canonical_stars


def test_mesh_difference_matrix():
    nodes = (np.arange(4) * 10.0, np.arange(3) * 10.0)
    D = jmod.mesh_difference_matrix(nodes)
    # 3 rows of 3 horizontal pairs, 2 rows of 4 vertical pairs
    assert D.shape == (3 * 3 + 2 * 4, 12)
    assert np.allclose(D @ np.ones(12), 0.0)
    L = (D.T @ D).toarray()
    # the degrees: corners 2, edges 3, interior 4
    deg = np.diag(L).reshape(3, 4)
    assert deg[0, 0] == 2 and deg[0, 1] == 3 and deg[1, 1] == 4


def _plane_fit(delta, monkeypatch):
    n, spacing = 768, 128
    rng = np.random.RandomState(31)
    yy, xx = np.mgrid[0:n, 0:n]
    sky = 0.2 + 0.3 * xx / n
    r = np.arange(60.0)
    canonical = (r, 3e7 * np.exp(-r / 4.0))
    stars = np.zeros(1, dtype=[('x', 'f8'), ('y', 'f8'), ('G', 'f4')])
    stars['x'], stars['y'], stars['G'] = 600.0, 150.0, 16.0
    image = (sky + rng.normal(size=(n, n))
             + render_canonical_stars((n, n), stars, canonical, gsub=99.0,
                                      verbose=False))
    # the whole hat of the node at (384, 384) masked
    good = np.ones((n, n), dtype=bool)
    good[250:518, 250:518] = False
    monkeypatch.setattr(jmod, 'MESH_SMOOTH_DELTA', delta)
    jf = jmod.joint_fit(image, good, stars, canonical, 1.0,
                        spacing=spacing, verbose=False)
    xn, yn = jf['nodes']
    k = 3 * xn.size + 3
    assert xn[3] == 384 and yn[3] == 384
    return jf['node_values'][k], jf['node_err'][k]


def test_smooth_fills_empty_node(monkeypatch):
    value, err = _plane_fit(0.1, monkeypatch)
    # the neighbors' mean, which for a plane is the plane
    assert abs(value - 0.35) < 0.05
    ridge_value, ridge_err = _plane_fit(None, monkeypatch)
    # the ridge alone leaves the node unconstrained
    assert ridge_err > 10 * err


def _filament_image():
    """noise, a faint wide filament (diffuse) with a bright galaxy on
    it, and a bright galaxy away from it"""
    rng = np.random.RandomState(7)
    n = 900
    yy, xx = np.mgrid[:n, :n]
    img = rng.normal(size=(n, n))
    img += 1.6 * np.exp(-0.5 * ((yy - 0.3 * xx - 400) / 40.0) ** 2)
    img += 200 * np.exp(-np.hypot(xx - 450, yy - 535) / 5.0)   # on it
    img += 200 * np.exp(-np.hypot(xx - 200, yy - 150) / 5.0)   # off it
    return img


def test_diffuse_segments_left_to_sky(monkeypatch):
    img = _filament_image()
    good = np.ones(img.shape, dtype=bool)
    ridge = (int(0.3 * 750 + 400), 750)     # (row, col) on the filament

    monkeypatch.setattr(jmod, 'SEG_DIFFUSE_MEDIAN', None)
    det = jmod.deep_segmentation(img, good, 1.0)
    assert det[ridge]

    monkeypatch.setattr(jmod, 'SEG_DIFFUSE_MEDIAN', 1.6)
    det, region = jmod.deep_segmentation(img, good, 1.0, return_diffuse=True)
    # the filament is left to the sky fit, both galaxies stay masked
    assert not det[ridge]
    assert det[535, 450] and det[150, 200]
    # and returned as the diffuse region, the galaxy on it included
    assert region[ridge] and region[535, 450] and not region[150, 200]


def test_joint_fit_returns_diffuse(monkeypatch):
    img = _filament_image()
    r = np.arange(60.0)
    canonical = (r, 3e7 * np.exp(-r / 4.0))
    stars = np.zeros(1, dtype=[('x', 'f8'), ('y', 'f8'), ('G', 'f4')])
    stars['x'], stars['y'], stars['G'] = 800.0, 100.0, 16.0
    img += render_canonical_stars(img.shape, stars, canonical, gsub=99.0,
                                  verbose=False)
    good = np.ones(img.shape, dtype=bool)
    ridge = (int(0.3 * 750 + 400), 750)

    monkeypatch.setattr(jmod, 'SEG_DIFFUSE_MEDIAN', 1.6)
    jf = jmod.joint_fit(img, good, stars, canonical, 1.0, spacing=128,
                        verbose=False)
    assert jf['diffuse'][ridge] and not jf['diffuse'][150, 200]

    monkeypatch.setattr(jmod, 'SEG_DIFFUSE_MEDIAN', None)
    jf = jmod.joint_fit(img, good, stars, canonical, 1.0, spacing=128,
                        verbose=False)
    assert not jf['diffuse'].any()


def _edge_setup():
    """a star just off the right edge with its inner wing on the image"""
    n, spacing = 512, 128
    rng = np.random.RandomState(7)
    r = np.arange(200.0)
    canonical = (r, 5e6 * (1.0 + r / 3.0) ** -2.5)
    stars = np.zeros(2, dtype=[('x', 'f8'), ('y', 'f8'), ('G', 'f4')])
    stars['x'] = [200.0, n + 20.0]
    stars['y'] = [250.0, 300.0]
    stars['G'] = [15.0, 11.0]
    truth = np.array([1.0, 1.4])
    image = (0.5 + rng.normal(size=(n, n))
             + render_canonical_stars((n, n), stars, canonical, gsub=99.0,
                                      amps=truth, verbose=False))
    good = np.ones((n, n), dtype=bool)
    # the stars' cores are masked
    yy, xx = np.mgrid[0:n, 0:n]
    for st in stars:
        good &= np.hypot(yy - st['y'], xx - st['x']) > 12
    return image, good, stars, canonical, spacing, truth


def test_edge_star_free_margin():
    image, good, stars, canonical, spacing, truth = _edge_setup()
    # default: the off-image star is pinned to the prediction
    jf = jmod.joint_fit(image, good, stars, canonical, 1.0,
                        spacing=spacing, verbose=False)
    assert jf['free'].tolist() == [True, False]
    assert jf['A'][1] == 1.0 and not np.isfinite(jf['A_err'][1])
    assert np.isfinite(jf['A_err'][0]) and jf['A_err'][0] > 0
    # within the margin the edge star is fit and recovers its amplitude
    jf = jmod.joint_fit(image, good, stars, canonical, 1.0,
                        spacing=spacing, free_margin=50.0, verbose=False)
    assert jf['free'].tolist() == [True, True]
    assert abs(jf['A'][1] - truth[1]) < 5 * jf['A_err'][1]
    assert abs(jf['A'][1] - truth[1]) < 0.1


def test_fixed_amplitudes():
    image, good, stars, canonical, spacing, truth = _edge_setup()
    # pass 2: every star pinned to a given amplitude, the sky alone fit
    jf = jmod.joint_fit(image, good, stars, canonical, 1.0,
                        spacing=spacing, gfit=-np.inf, amps=truth,
                        verbose=False)
    assert not jf['free'].any()
    assert np.array_equal(jf['A'], truth)
    resid = image - jf['sky'] - jf['star_model']
    # the wing of the edge star is gone: the band 20-60 px inside the
    # edge along its row is flat to the noise
    band = resid[280:320, 452:492]
    assert abs(band.mean()) < 0.1
    # with amps the free star's prior is centered there; the faint
    # star is weakly constrained and lands within its error
    jf2 = jmod.joint_fit(image, good, stars, canonical, 1.0,
                         spacing=spacing, amps=truth, verbose=False)
    assert jf2['free'].tolist() == [True, False]
    assert abs(jf2['A'][0] - truth[0]) < 3 * jf2['A_err'][0]
    assert jf2['A'][1] == truth[1]


def test_free_override():
    image, good, stars, canonical, spacing, truth = _edge_setup()
    # the on-image star pinned at its amplitude, the edge star free
    jf = jmod.joint_fit(image, good, stars, canonical, 1.0,
                        spacing=spacing, amps=truth,
                        free=np.array([False, True]), verbose=False)
    assert jf['free'].tolist() == [False, True]
    assert jf['A'][0] == truth[0] and not np.isfinite(jf['A_err'][0])
    assert abs(jf['A'][1] - truth[1]) < 0.1


def test_ghost_disk_recovered():
    # a bright star with its wing and a ghost disk of a known
    # amplitude on a sky plane; the star's inner region masked as the
    # census would: the fit recovers the wing amplitude and the disk
    n, spacing = 2048, 256
    rng = np.random.RandomState(5)
    yy, xx = np.mgrid[0:n, 0:n]
    sky = 0.5 + 0.4 * xx / n - 0.2 * yy / n
    r = np.arange(0.0, 3000.0, 0.5)
    canonical = (r, 4e8 * (1.0 + r / 3.0) ** -2.5)
    stars = np.zeros(1, dtype=[('x', 'f8'), ('y', 'f8'), ('G', 'f4')])
    stars['x'], stars['y'], stars['G'] = 1000.0, 1050.0, 6.5
    d_true = 1.4
    image = (sky + rng.normal(size=(n, n))
             + 0.9 * render_canonical_stars((n, n), stars, canonical,
                                            gsub=99.0, verbose=False)
             + jmod.render_disks((n, n), stars, np.array([d_true]), 'i'))
    good = np.hypot(xx - 1000.0, yy - 1050.0) > 300.0
    with pytest.raises(ValueError):
        jmod.joint_fit(image, good, stars, canonical, 1.0,
                       spacing=spacing, gfit=99.0, verbose=False)
    jf = jmod.joint_fit(image, good, stars, canonical, 1.0,
                        spacing=spacing, gfit=99.0, band='i', verbose=False)
    assert jf['disk_free'][0] and jf['free'][0]
    assert abs(jf['A'][0] - 0.9) < 0.05
    assert abs(jf['D'][0] - d_true) < 0.05
    assert jf['D_err'][0] < 0.05
    # the disk pinned: a sky-only fit at the given amplitudes
    jf2 = jmod.joint_fit(image, good, stars, canonical, 1.0,
                         spacing=spacing, gfit=-np.inf,
                         amps=np.array([0.9]), disks=np.array([d_true]),
                         fit_disks=False, band='i', verbose=False)
    assert not jf2['disk_free'][0] and jf2['D'][0] == d_true
    assert np.isnan(jf2['D_err'][0])
    resid = image - jf2['sky'] - jf2['star_model']
    assert abs(np.median(resid[good])) < 0.05
