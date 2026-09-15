"""
stack-free tests: the mask conversion, the restoration
accounting, the wide exclusion, the sky pass and the profile
machinery on a synthetic detector
"""
import numpy as np
import pytest

from lsst_starsub.maskbits import DM_INTRP, DM_NO_DATA, DM_SAT
from lsst_starsub.visit.exposure import (
    VisitExposure,
    build_wide_star_mask,
    convert_mask,
    iq_tier,
    restore_background,
    sky_background,
    star_model_image,
)
from lsst_starsub.visit.profiles import (
    measure_profiles,
    radial_edges,
    stack_profiles,
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


def test_core_amplitudes_and_visit_wing():
    from lsst_starsub.wing import WingModel, core_amplitudes
    from lsst_starsub.visit.trough import R_BLEND, R_JOIN, visit_wing

    # a star with a known amplitude on a flat noisy image
    rng = np.random.default_rng(5)
    n = 128
    r = np.arange(0.0, 3000.0, 0.5)
    T = 2.0e8 * (1.0 + r / 2.0) ** -3.0
    wing = WingModel(r, T)
    G = 16.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(yy - 64.3, xx - 63.6)
    image = 1.7 * 10 ** (-0.4 * G) * np.interp(rr, r, T)
    image += rng.normal(size=(n, n))
    good = np.ones((n, n), dtype=bool)
    amps, errs, ok = core_amplitudes(
        image, good, np.array([63.6, 5.0]), np.array([64.3, 64.0]),
        np.array([G, G]), wing, 5.0, amp_range=(0.25, 4.0), sky_sigma=1.0,
    )
    assert ok.tolist() == [True, False]
    assert abs(amps[0] - 1.7) < 5 * errs[0] and abs(amps[0] - 1.7) < 0.05
    assert amps[1] == 1.0 and not np.isfinite(errs[1])

    # the visit wing: the stack inside the junction, the canonical
    # beyond, blended between
    prof = np.exp(-np.arange(72.0) / 5.0)
    tmpl = dict(
        prof=prof,
        params=dict(k_in=3.0, slope=-3.0, ln_a=0.0, aur_slope=-2.0,
                    aur_amp=0.0),
    )
    canonical = WingModel(r, T)
    vw = visit_wing(tmpl, canonical)
    assert np.allclose(vw.r, r)
    inside = r < R_BLEND
    assert np.allclose(vw.T[inside], 3.0 * np.interp(r[inside],
                                                     np.arange(72.0), prof))
    beyond = r >= R_JOIN
    assert np.allclose(vw.T[beyond], T[beyond])


def test_flag_dead_pixels():
    from lsst_starsub.visit.exposure import flag_dead_pixels

    var = np.full((64, 64), 100.0, dtype='f4')
    var[:32, :16] = 3.0          # a dead block
    var[5, 40] = np.nan          # a bad pixel stays as it is
    mask = np.zeros((64, 64, 1), dtype='i4')
    n = flag_dead_pixels(mask, var)
    assert n == 32 * 16
    assert (mask[:32, :16, 0] & DM_NO_DATA).all()
    assert not (mask[32:, :, 0] & DM_NO_DATA).any()
    assert mask[5, 40, 0] == 0


def test_detector_core_stack():
    from lsst_starsub.visit.exposure import (
        CORE_STACK_MIN, detector_core_stack,
    )
    from lsst_starsub.wing import core_amplitudes

    # 30 stars of one gaussian core, per unit Gaia flux 1e9 at the
    # peak, with amplitudes 0.8-1.2 about 1, on a flat noisy image
    rng = np.random.default_rng(11)
    vexp = make_vexp(dim=512, sky=0.0, sigma=2.0)
    vexp.backgrounds['initial_coarse'][:] = 0.0
    vexp.backgrounds['initial_fine'][:] = 0.0
    n = 30
    # positions at least 30 px apart, so no aperture holds two stars
    xs, ys = [], []
    while len(xs) < n:
        x, y = rng.uniform(40, 470, 2)
        if all(np.hypot(x - a, y - b) > 30 for a, b in zip(xs, ys)):
            xs.append(x)
            ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    G = rng.uniform(16.0, 18.5, n)
    truth = rng.uniform(0.8, 1.2, n)
    yy, xx = np.mgrid[0:512, 0:512]
    sig = 2.5
    for x, y, g, a in zip(xs, ys, G, truth):
        rr2 = (yy - y) ** 2 + (xx - x) ** 2
        vexp.image.array[:] += (
            a * 1e9 * 10 ** (-0.4 * g) * np.exp(-0.5 * rr2 / sig ** 2)
        ).astype('f4')
    stars = np.zeros(n, dtype=[
        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'), ('on_image', 'i2'),
        ('is_sat', 'i2'),
    ])
    stars['x'], stars['y'], stars['G'], stars['on_image'] = xs, ys, G, 1
    assert n >= CORE_STACK_MIN
    stack, nstar = detector_core_stack(vexp, stars)
    assert nstar == n
    half = stack.shape[0] // 2
    # the stack peaks at the median amplitude, a little below 1e9
    # for the pixel phases
    peak = stack[half, half] / 1e9
    assert 0.9 * np.median(truth) < peak <= np.median(truth) * 1.02
    amps, errs, ok = core_amplitudes(
        vexp.image.array, vexp.good, xs, ys, G, stack, 5.0, sky_sigma=2.0,
    )
    assert ok.all()
    # each star's amplitude relative to the median star, to the
    # pixel-phase scatter of a 5 px aperture
    assert np.allclose(amps / np.median(truth), truth, rtol=0.03)
