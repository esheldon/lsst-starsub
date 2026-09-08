"""
stack-free tests: the mask conversion, the restoration
accounting, the wide exclusion, the sky pass and the profile
machinery on a synthetic detector
"""
import numpy as np
import pytest

from lsst_mdet.defaults import DM_INTRP, DM_NO_DATA, DM_SAT
from lsst_starsub.visit import (
    VisitExposure, build_wide_star_mask, convert_mask, iq_tier,
    restore_background, sky_background, star_model_image,
)
from lsst_starsub.profiles import (
    measure_profiles, radial_edges, stack_profiles,
)

PLANES = {
    'BAD': 0, 'SAT': 1, 'INTRP': 2, 'CR': 3, 'EDGE': 4,
    'DETECTED': 5, 'SUSPECT': 7, 'NO_DATA': 8, 'SPIKE': 15,
}


def test_convert_mask():
    m = np.zeros((4, 4), dtype='i4')
    m[0, 0] = 1 << PLANES['SAT']
    m[0, 1] = 1 << PLANES['CR']
    m[0, 2] = 1 << PLANES['NO_DATA']
    m[0, 3] = (1 << PLANES['DETECTED']) | (1 << PLANES['SPIKE'])
    m[1, 0] = (1 << PLANES['SAT']) | (1 << PLANES['INTRP'])
    out = convert_mask(m, PLANES)
    assert out.shape == (4, 4, 1)
    assert out[0, 0, 0] == DM_SAT
    assert out[0, 1, 0] == DM_INTRP
    assert out[0, 2, 0] == DM_NO_DATA
    assert out[0, 3, 0] == 0
    assert out[1, 0, 0] == DM_SAT | DM_INTRP
    assert out[2:, :, 0].sum() == 0


def test_iq_tier():
    assert iq_tier(0.001) == 'low'
    assert iq_tier(0.003) == 'medium'
    assert iq_tier(0.01) == 'high'
    assert iq_tier(0.5) == 'very_high'
    assert iq_tier(np.nan) == 'unknown'


def make_vexp(dim=512, sky=1000.0, sigma=20.0, seed=3):
    rng = np.random.default_rng(seed)
    coarse = np.full((dim, dim), sky, dtype='f4')
    fine = (rng.normal(size=(dim, dim)) * 0.0).astype('f4')
    fine += 3.0
    image = rng.normal(scale=sigma, size=(dim, dim)).astype('f4')
    var = np.full((dim, dim), sigma ** 2, dtype='f4')
    mask = np.zeros((dim, dim, 1), dtype='i4')
    return VisitExposure(
        image=image, variance=var, mask=mask, band='i',
        backgrounds=dict(
            initial_coarse=coarse, initial_fine=fine,
            skycorr=np.zeros((dim, dim), dtype='f4'),
        ),
        rng=rng,
    )


def test_restore_accounting():
    vexp = make_vexp()
    before = vexp.image.array.copy()
    add = restore_background(vexp, which='fine')
    assert np.allclose(add, 3.0)
    assert np.allclose(vexp.image.array - before, 3.0)
    assert np.allclose(vexp.restored, 3.0)
    add2 = restore_background(vexp, which='initial')
    assert np.allclose(vexp.restored, 3.0 + 1003.0)
    assert np.allclose(add2, 1003.0)
    with pytest.raises(ValueError):
        restore_background(vexp, which='bogus')
    assert vexp.sky_level == pytest.approx(1003.0)
    assert vexp.sky_sigma == pytest.approx(20.0)


def test_wide_mask():
    stars = np.zeros(2, dtype=[
        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'), ('on_image', 'i2'),
    ])
    stars['x'] = [100.0, 400.0]
    stars['y'] = [100.0, 400.0]
    stars['G'] = [12.0, 18.0]
    wide = build_wide_star_mask(stars, (512, 512))
    # the bright star covers its template extent (>= 250 px)
    assert wide[100, 340]
    # the faint star: circle floor 20 plus the margin
    assert wide[400, 400 + 30]
    assert not wide[400, 400 + 60]


def test_sky_pass_and_profiles():
    vexp = make_vexp()
    restore_background(vexp, which='initial')
    excl = np.zeros(vexp.image.array.shape, dtype=bool)
    back = sky_background(vexp, exclude=excl, bw=128)
    assert np.median(back) == pytest.approx(1003.0, abs=2.0)
    assert abs(np.median(vexp.image.array)) < 2.0

    # a fake star model and the profile machinery
    stars = np.zeros(1, dtype=[
        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'), ('on_image', 'i2'),
    ])
    stars['x'] = 256.0
    stars['y'] = 256.0
    stars['G'] = 14.0
    stars['on_image'] = 1
    gy, gx = np.mgrid[0:512, 0:512]
    rr = np.maximum(np.hypot(gy - 256.0, gx - 256.0), 1.0)
    wing = (1.0e5 * rr ** -2.0).astype('f4')
    states = dict(
        flat=vexp.image.array + wing,
        residual=vexp.image.array,
    )
    seg = np.zeros((512, 512), dtype='i4')
    edges, table = measure_profiles(states, vexp, stars, seg)
    assert edges.size == radial_edges().size
    assert set(table['state']) == {'flat', 'residual'}
    med_f, cnt = stack_profiles(table, 'flat', 13.0, 15.0, min_stars=1)
    med_r, _ = stack_profiles(table, 'residual', 13.0, 15.0, min_stars=1)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    w = np.isfinite(med_f) & (rmid < 150)
    assert w.sum() >= 3
    # the wing is recovered in flux-normalized sigma units
    # (the annulus median of r^-2 sits a few percent below the
    # bin-center value; pixel noise adds a little more)
    expect = 1.0e5 * rmid ** -2.0 / 20.0 / 10 ** (-0.4 * 14.0)
    assert np.allclose(med_f[w], expect[w], rtol=0.2)
    assert np.all(np.abs(med_r[w]) < 0.2 * np.abs(expect[w]))


def test_star_model_image():
    T = np.ones((5, 5), dtype='f4')
    slist = [
        dict(sl=np.s_[0:5, 0:5], T=T, A=2.0),
        dict(sl=np.s_[2:7, 2:7], T=T, A=0.0),
    ]
    model = star_model_image((10, 10), slist)
    assert model[0, 0] == 2.0
    assert model[6, 6] == 0.0
