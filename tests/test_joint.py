"""
the joint fit's sky mesh: the neighbour-difference operator and the
smoothness prior filling a node that has no data
"""
import numpy as np

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
    # the neighbours' mean, which for a plane is the plane
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
