"""
the wings of the stars below the census depth: the core taper and the
subtraction, cores left in place
"""
import numpy as np

import lsst_starsub.starsub as smod
from lsst_starsub.wing import WingModel


def _wing():
    r = np.arange(0.0, 200.0, 0.5)
    return WingModel(r, 1e9 * np.exp(-r / 6.0) + 1e6 / (1 + r) ** 3)


def test_wing_taper():
    r = np.arange(0.0, 30.0, 0.5)
    t = smod.wing_taper(r, rin=8.0, rout=16.0)
    assert np.all(t[r <= 8.0] == 0)
    assert np.allclose(t[r >= 16.0], 1.0)
    mid = (r > 8.0) & (r < 16.0)
    assert np.all((t[mid] > 0) & (t[mid] < 1))


def _gaia():
    gaia = np.zeros(2, dtype=[('phot_g_mean_mag', 'f8')])
    gaia['phot_g_mean_mag'] = [18.0, 20.0]   # a census star, a faint one
    x = np.array([30.0, 100.0])
    y = np.array([30.0, 100.0])
    return gaia, x, y


def test_faint_wings_off(monkeypatch):
    monkeypatch.setattr(smod, 'WING_GMAX', None)
    gaia, x, y = _gaia()
    image = np.zeros((200, 200), dtype='f4')
    assert smod.subtract_faint_wings(image, gaia, x, y, _wing(), 19.0,
                                     verbose=False) == 0
    assert not image.any()


def test_faint_wings_subtracted(monkeypatch):
    # the amplitude from the Gaia prediction
    monkeypatch.setattr(smod, 'WING_GMAX', 21.0)
    monkeypatch.setattr(smod, 'WING_CORE_RAP', None)
    monkeypatch.setattr(smod, 'WING_RIN', 8.0)
    monkeypatch.setattr(smod, 'WING_ROUT', 16.0)
    gaia, x, y = _gaia()
    wing = _wing()
    image = np.zeros((200, 200), dtype='f4')
    n = smod.subtract_faint_wings(image, gaia, x, y, wing, 19.0,
                                  verbose=False)
    assert n == 1
    yy, xx = np.mgrid[:200, :200]
    rf = np.hypot(xx - 100.0, yy - 100.0)
    # the faint star's core untouched, its wing subtracted beyond rout
    assert np.all(image[rf < 7.5] == 0)
    ring = (rf > 20) & (rf < 40)
    pred = 10 ** (-0.4 * 20.0) * np.interp(rf[ring], wing.r, wing.T)
    assert np.allclose(image[ring], -pred, rtol=1e-3)
    # the census star (G 18) is not the job of this step
    rc = np.hypot(xx - 30.0, yy - 30.0)
    assert np.all(image[rc < 3] == 0)


def _star_image(amp):
    """a G 20 star at (100.3, 99.6) drawn at amp times its prediction"""
    from lsst_starsub.visit import render_canonical_stars

    star = np.zeros(1, dtype=[('x', 'f8'), ('y', 'f8'), ('G', 'f8')])
    star['x'], star['y'], star['G'] = 100.3, 99.6, 20.0
    return star, render_canonical_stars((200, 200), star, _wing(), gsub=21,
                                        amps=[amp], verbose=False)


def test_core_amplitudes(monkeypatch):
    monkeypatch.setattr(smod, 'WING_AMP_RANGE', (0.25, 4.0))
    star, image = _star_image(2.0)
    good = np.ones(image.shape, dtype=bool)
    args = (star['x'], star['y'], star['G'], _wing(), 5.0)
    assert np.isclose(smod.core_amplitudes(image, good, *args)[0], 2.0,
                      rtol=1e-3)
    # a bad pixel in the aperture: the prediction
    good[100, 101] = False
    assert smod.core_amplitudes(image, good, *args)[0] == 1.0
    # out of the plausible range: the prediction
    star, image = _star_image(10.0)
    good = np.ones(image.shape, dtype=bool)
    assert smod.core_amplitudes(image, good, *args)[0] == 1.0


def _fit(**extra):
    stars = np.zeros(2, dtype=[(c, 'f8') for c in smod.CENSUS_COLUMNS])
    wing = _wing()
    wing.fname = 'wing.fits'
    fit = dict(
        stars=stars, A=np.ones(2), free=np.array([True, False]),
        nodes=(np.arange(3.0), np.arange(2.0)), node_values=np.zeros(6),
        spacing=256, prior=0.3, gfit=17.0, gsub=19.0, chi2=1.0, ncell=10,
        sky_sigma=1.0, shape=(100, 120), bg_restored='object', wing=wing,
    )
    fit.update(extra)
    return fit


def test_fit_tables_wing_meta():
    fits = {'r': _fit(nwing=7, wing_gmax=21.0, wing_rin=8.0, wing_rout=16.0,
                      wing_core_rap=None)}
    meta = smod.make_fit_tables(fits)[smod.META_EXT]
    assert meta['nwing'][0] == 7 and meta['wing_gmax'][0] == 21.0
    assert np.isnan(meta['wing_core_rap'][0])
    # fit dicts from before the faint-star step
    meta = smod.make_fit_tables({'r': _fit()})[smod.META_EXT]
    assert meta['nwing'][0] == 0 and np.isnan(meta['wing_gmax'][0])


def test_faint_wings_core_matched(monkeypatch):
    monkeypatch.setattr(smod, 'WING_GMAX', 21.0)
    monkeypatch.setattr(smod, 'WING_CORE_RAP', 5.0)
    monkeypatch.setattr(smod, 'WING_RIN', 8.0)
    monkeypatch.setattr(smod, 'WING_ROUT', 16.0)
    star, image = _star_image(2.0)
    gaia = np.zeros(1, dtype=[('phot_g_mean_mag', 'f8')])
    gaia['phot_g_mean_mag'] = star['G']
    smod.subtract_faint_wings(image, gaia, star['x'], star['y'], _wing(),
                              19.0, verbose=False)
    yy, xx = np.mgrid[:200, :200]
    rf = np.hypot(xx - 100.3, yy - 99.6)
    # the wing at twice the prediction is removed beyond the taper
    ring = (rf > 20) & (rf < 40)
    assert np.allclose(image[ring], 0, atol=1e-6)
