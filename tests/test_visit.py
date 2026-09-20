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
    vw, scale = visit_wing(tmpl, canonical)
    assert np.allclose(vw.r, r)
    # the stack scaled onto the canonical over the match range
    stack = 3.0 * np.interp(r, np.arange(72.0), prof)
    m = (r >= 30.0) & (r <= 40.0)
    assert np.isclose(scale, np.median(T[m] / stack[m]))
    inside = r < R_BLEND
    assert np.allclose(vw.T[inside], scale * stack[inside])
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
    # the wing the amplitudes refer to: the same core at amplitude 1
    r = np.linspace(0, 20, 401)
    wing = (r, 1e9 * np.exp(-0.5 * r ** 2 / sig ** 2))
    stack, nstar = detector_core_stack(vexp, stars, wing)
    assert nstar == n
    half = stack.shape[0] // 2
    # the stack has the wing's zero point (its flux within 12 px), so
    # it peaks at 1e9, a little below for the pixel phases
    peak = stack[half, half] / 1e9
    assert 0.9 < peak <= 1.02
    amps, errs, ok = core_amplitudes(
        vexp.image.array, vexp.good, xs, ys, G, stack, 5.0, sky_sigma=2.0,
    )
    assert ok.all()
    # each star's amplitude is relative to the wing, not to the median
    # star: the stack supplies only the core's shape
    assert abs(np.median(amps) - np.median(truth)) < 0.02
    # the faintest stars have 40 nJy in the aperture against a noise
    # of 20, so 5 percent
    assert np.allclose(amps, truth, rtol=0.05)
    with pytest.raises(ValueError):
        detector_core_stack(vexp, stars, wing, half=8)

    # a bright blob 7 px from one stack star, a galaxy or an unlisted
    # neighbor: the clipped mean is not thrown by it
    ratio = stack[half, half] / stack[half, half + 4]
    k = int(np.argmin(G))
    rr2 = (yy - ys[k]) ** 2 + (xx - xs[k] - 7) ** 2
    vexp.image.array[:] += (
        50 * 1e9 * 10 ** (-0.4 * G[k]) * np.exp(-0.5 * rr2 / sig ** 2)
    ).astype('f4')
    stack2, nstar2 = detector_core_stack(vexp, stars, wing)
    assert nstar2 == n
    assert abs(stack2[half, half] / stack2[half, half + 4] / ratio - 1) < 0.02
    assert abs(stack2[half, half] / stack[half, half] - 1) < 0.02
    # and exclude leaves stars out
    exclude = np.zeros(n, dtype=bool)
    exclude[k] = True
    stack3, nstar3 = detector_core_stack(vexp, stars, wing, exclude=exclude)
    assert nstar3 == n - 1


def test_box_medians():
    from lsst_starsub.visit.profiles import box_medians, state_maps

    rng = np.random.default_rng(3)
    image = rng.normal(0, 1, (70, 100)) + 5.0
    usable = np.ones(image.shape, dtype=bool)
    # a bright source the mask hides, and a box mostly masked
    image[10:14, 10:14] = 1000.0
    usable[10:14, 10:14] = False
    usable[32:64, 0:30] = False
    med = box_medians(image, usable, box=32)
    assert med.shape == (2, 3)
    assert np.isnan(med[1, 0])
    assert np.allclose(med[np.isfinite(med)], 5.0, atol=0.5)
    maps = state_maps({'a': image, 'b': 2 * image}, usable, box=32)
    assert set(maps) == {'box_a', 'box_b'}
    m, hdr = maps['box_b']
    assert hdr['BOX'] == 32 and hdr['STATE'] == 'b'
    assert np.allclose(m[0, 1], 2 * med[0, 1])


def test_box_background_no_overshoot():
    from lsst_starsub.visit.exposure import box_background

    # a sky with a gradient, a large excluded corner: the background
    # in the corner comes from the nearest boxes that have pixels and
    # stays within the sky's range (a spline over the boxes overshot
    # by 60 nJy on a real detector)
    rng = np.random.default_rng(9)
    n = 1024
    yy, xx = np.mgrid[0:n, 0:n]
    sky = 1000.0 + 0.02 * xx + 0.01 * yy
    image = (sky + rng.normal(0, 20.0, (n, n))).astype('f4')
    usable = np.ones((n, n), dtype=bool)
    usable[:600, 500:] = False
    back = box_background(image, usable, 128)
    assert back.shape == image.shape
    assert back.min() >= sky.min() - 5 and back.max() <= sky.max() + 5
    # where there are pixels the background follows the sky to the
    # box noise (20 / sqrt(128^2) ~ 0.2, the bilinear lag a little more)
    have = usable & (xx > 64) & (xx < n - 64) & (yy > 64) & (yy < n - 64)
    assert np.abs(back - sky)[have].max() < 3.0
    # the far-edge partial boxes are filled, not zero
    assert np.all(np.isfinite(back)) and back[-1, -1] > 900


def test_product_evaluations_match_renderers():
    from lsst_starsub.joint import render_mesh
    from lsst_starsub.visit.exposure import box_background, boxes_at
    from lsst_starsub.visit.product import Product, mesh_at, sky_at

    rng = np.random.default_rng(21)
    shape = (300, 340)
    # the mesh: evenly spaced nodes past the far edge, random values
    xn = np.arange(0, shape[1] + 64, 64, dtype='f8')
    yn = np.arange(0, shape[0] + 64, 64, dtype='f8')
    values = rng.normal(0, 3.0, (yn.size, xn.size))
    ref = render_mesh((xn, yn), values.ravel(), shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    got = mesh_at(xn, yn, values, xx, yy)
    assert np.allclose(got, ref, atol=1e-4)
    # the boxes: box_background's image equals boxes_at on the grid
    image = (1000.0 + rng.normal(0, 5.0, shape)).astype('f4')
    usable = np.ones(shape, dtype=bool)
    back, boxes = box_background(image, usable, 32, return_boxes=True)
    assert np.allclose(boxes_at(boxes, 32, xx, yy), back, atol=1e-3)
    # the product's sky is their sum, at any position
    p = Product()
    p.xn, p.yn, p.node_values, p.boxes, p.bw = xn, yn, values, boxes, 32
    assert np.allclose(sky_at(p, xx, yy), back + ref, atol=1e-3)
    assert np.isfinite(sky_at(p, np.array([10.3, 339.9]),
                              np.array([0.2, 299.7]))).all()


def test_wing_fit_disk_term():
    """the ghost ring in the wing profiles comes off at the band's level and
    the aureole is recovered; left in, the aureole flattens"""
    from lsst_starsub.joint import disk_level
    from lsst_starsub.visit.template import (
        binned_wings, disk_annuli, fit_wing_model, wing_edges, wing_law,
    )

    rng = np.random.default_rng(3)
    slope, ln_a, aur_slope, aur_amp, k_in = -4.0, 1.0, -2.2, 1e-3, 4e12
    disk_amp = disk_level('i')
    edges = wing_edges()
    rmid = 0.5 * (edges[1:] + edges[:-1])

    # the stack: the law in template units, 1 percent errors
    r = np.arange(60.0)
    prof = wing_law(r, slope, ln_a, aur_slope, aur_amp)
    prof_err = 0.01 * prof + 1e-9
    prof = prof + rng.normal(size=r.size) * prof_err

    # the wing profiles: 400 stars G 9-15, the wing plus the disk per unit
    # flux, noise from the sky scaled by the flux
    n = 400
    G = rng.uniform(9.0, 15.0, n)
    flux = 10.0 ** (-0.4 * G)
    truth = k_in * wing_law(rmid, slope, ln_a, aur_slope, aur_amp) \
        + disk_amp * disk_annuli(edges, rmid)
    sig = 3e-9 / flux[:, None] / np.sqrt(rmid)[None, :]
    wing = np.zeros(n, dtype=[('G', 'f4'), ('prof', 'f8', rmid.size)])
    wing['G'] = G
    wing['prof'] = truth[None, :] + rng.normal(size=(n, rmid.size)) * sig

    binned = binned_wings(wing, edges)
    fit = fit_wing_model(prof, prof_err, binned, edges, disk_amp)
    assert fit['disk_amp'] == disk_amp
    assert abs(fit['aur_slope'] - aur_slope) < 0.11
    assert abs(fit['k_in'] / k_in - 1) < 0.1

    nodisk = fit_wing_model(prof, prof_err, binned)
    assert nodisk['disk_amp'] == 0.0
    assert nodisk['aur_slope'] >= fit['aur_slope']
    assert nodisk['chi2'] > 10 * fit['chi2']


def test_product_band_from_meta_table():
    """the band comes out of the one-row meta table as a plain string"""
    from lsst_starsub.visit.product import Product, product_band

    p = Product()
    p.meta = np.zeros(1, dtype=[('band', 'U1'), ('visit', 'i8')])
    p.meta['band'] = 'z'
    assert product_band(p) == 'z'
    p.meta = np.zeros(1, dtype=[('band', 'S2')])
    p.meta['band'] = b'r '
    assert product_band(p) == 'r'


def test_core_amplitudes_saturated_center():
    """a star with its center unusable is measured over the rest of
    the aperture when the fraction left is allowed"""
    from lsst_starsub.wing import WingModel, core_amplitudes

    rng = np.random.default_rng(11)
    n = 96
    r = np.arange(0.0, 3000.0, 0.5)
    T = 2.0e8 * (1.0 + r / 2.0) ** -3.0
    wing = WingModel(r, T)
    G, amp = 15.0, 1.6
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(yy - 48.3, xx - 47.6)
    image = amp * 10 ** (-0.4 * G) * np.interp(rr, r, T)
    image += rng.normal(size=(n, n))
    good = rr > 4.0
    args = (np.array([47.6]), np.array([48.3]), np.array([G]), wing, 11.0)
    amps, errs, ok = core_amplitudes(image, good, *args, sky_sigma=1.0)
    assert not ok[0] and amps[0] == 1.0
    amps, errs, ok = core_amplitudes(image, good, *args, sky_sigma=1.0,
                                     min_frac=0.05)
    assert ok[0]
    assert abs(amps[0] - amp) < 3 * errs[0] and abs(amps[0] - amp) < 0.05
    # too little of the aperture left: not measured
    amps, errs, ok = core_amplitudes(image, rr > 10.0, *args, sky_sigma=1.0,
                                     min_frac=0.05)
    assert not ok[0]
