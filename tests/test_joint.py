"""
the joint fit's sky mesh: the neighbour-difference operator and the
smoothness prior filling a node that has no data
"""
import numpy as np

import lsst_starsub.joint as jmod
from lsst_starsub.visit import render_canonical_stars


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
