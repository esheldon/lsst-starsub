"""
per-visit wing characterization pooled over the focal plane

Per-detector templates (20-25 stamps) are not usable: their halo
slopes scatter from -2.6 to -4.8 with no seeing trend.  Here one
template and aureole are built per visit from all its detectors:

- the inner template is the median stack of every detector's
  template-star stamps (G 15.5-17.5, core-normalized, neighbors
  NaNed, stamps inside a brighter star's wide zone excluded),
  from the restored image with a wide-box sky pass
- the wing cloud is the flux-normalized (10^(-0.4 G)) azimuthal
  profile of every census star brighter than AUR_GMAX, on the
  warp state (raw minus the focal-plane skyCorr model, no
  detector-scale sky fit) so no local sky pass can have absorbed
  the wing; each detector's ambient level is subtracted first.
  Bright stars (G < FAR_GMAX) are measured to FAR_RMAX px, the
  mid-bright ones to MID_RMAX.  Detector-scale sky structure is
  uncorrelated with the star positions and averages out of the
  pooled median
- the aureole is one power law fit jointly with the inner law's
  scale on the pooled cloud beyond AUR_RMIN, in template units;
  the fitted scale k_in converts template units to nJy per unit
  Gaia flux, so a star's wing image is k_in 10^(-0.4 G) T(r)
  without a per-star amplitude fit

The products (write_template_file) are the extended template
array (to TMPL_OUT_MAX), the inner stack, the parameters, the
cloud with its model, and the per-detector and per-star records
"""
import numpy as np

from .census import build_star_mask, field_segmentation, select_stars
from .profiles import R_MIN, measure_profiles
from .stamps import (
    AUR_GMAX, AUR_SLOPE_SEP, CANON, HALO_SLOPE, TMPL_HALF, TMPL_OUT_MAX,
    denoise_template, extend_template_halo, fit_halo_slope,
    measure_coadd_fwhm, select_template_stars,
)
from .visit import (
    GSUB, WIDE_BW, WIDE_GMAX, build_wide_star_mask, restore_background,
    sky_background,
)

# the wing cloud: bright stars to FAR_RMAX, the rest to MID_RMAX
FAR_GMAX = 13.5
FAR_RMAX = 2500.0
MID_RMAX = 900.0
NBIN_WING = 30
# the aureole fit range in the pooled cloud.  Beyond AUR_RMAX a
# single visit cannot measure the wing: the per-star local sky
# offsets of the warp state (a few nJy, detector-scale structure)
# exceed the wing there and only pooling over many visits
# averages them out; the far cloud is kept for that
AUR_RMIN = 40.0
AUR_MIN_COUNT = 10
# G bins of the cloud diagnostics
CLOUD_GBINS = [(6.0, 10.0), (10.0, 12.0), (12.0, 13.5), (13.5, 15.5)]


def wing_edges():
    return np.unique(np.round(np.logspace(
        np.log10(R_MIN), np.log10(FAR_RMAX), NBIN_WING + 1,
    )))


def star_stamps(image, good, seg, x, y, sel, halo):
    """
    the individual core-normalized, sub-pixel-aligned template
    stamps (as lsst_starsub.stamps.stack_star_stamps builds before
    its median), skipping stars inside the halo zone

    Returns
    -------
    stamps (n, 2 TMPL_HALF + 1, 2 TMPL_HALF + 1) with NaN where
    unusable, the indices into sel that were used, and the core
    amplitudes the stamps were normalized by (image units; with
    the stars' Gaia fluxes these give the zero point k_in of
    template units directly)
    """
    from scipy import ndimage

    half = TMPL_HALF
    m = half + 2
    gy, gx = np.mgrid[-half:half + 1, -half:half + 1]
    core = np.hypot(gy, gx) < 6
    stamps = []
    used = []
    amps = []
    for k in sel:
        cx, cy = float(x[k]), float(y[k])
        icx, icy = int(round(cx)), int(round(cy))
        if halo[icy, icx]:
            continue
        cut = np.s_[icy - m:icy + m + 1, icx - m:icx + m + 1]
        stamp = image[cut].astype('f8')
        stamp[~good[cut]] = np.nan
        segcut = seg[cut]
        central = segcut[m, m]
        stamp[(segcut != 0) & (segcut != central)] = np.nan
        stamp = ndimage.shift(
            stamp, (icy - cy, icx - cx), order=1, cval=np.nan,
        )[2:-2, 2:-2]
        amp = np.nansum(stamp[core])
        if not amp > 0:
            continue
        stamps.append(stamp / amp)
        used.append(k)
        amps.append(float(amp))
    return (np.array(stamps), np.array(used, dtype=int),
            np.array(amps, dtype='f8'))


def extract_detector(vexp, gaia, gsub=GSUB):
    """
    the per-detector inputs of the pooled template

    Parameters
    ----------
    vexp: VisitExposure
        As loaded (delivered state); modified in place
    gaia: array with fields
        The gaia extract for the detector

    Returns
    -------
    dict with visit, detector, band, fwhm, sky_sigma, calib,
    ambient, stamps (n, 101, 101), stamp_G, wing (structured
    array per star: G, x, y, prof in nJy per unit flux, npix),
    edges
    """
    from .gaia import gaia_pixel_positions

    mask0 = vexp.mask.array[:, :, 0]
    x, y = gaia_pixel_positions(gaia, vexp.wcs, vexp.bbox)
    stars = select_stars(gaia, x, y, mask0, gsub=gsub)
    build_star_mask(stars, mask0, verbose=False)
    wide = build_wide_star_mask(stars, mask0.shape)
    halo = build_wide_star_mask(stars[stars['G'] < WIDE_GMAX], mask0.shape)
    fwhm = measure_coadd_fwhm(vexp)
    sig = vexp.sky_sigma
    good = vexp.good

    delivered = vexp.image.array.copy()
    # the warp state for the wings: no detector-scale sky fit
    warp = delivered - vexp.backgrounds['skycorr']

    # the sky-flattened restored image for the stamps
    restore_background(vexp, which='initial')
    sky_background(vexp, exclude=wide, bw=WIDE_BW)
    flat = vexp.image.array
    seg = field_segmentation(flat, good, sig)

    ok = good & (seg == 0) & ~wide
    amb_flat = float(np.median(flat[ok]))
    amb_warp = float(np.median(warp[ok]))

    sel = select_template_stars(gaia, x, y, flat.shape)
    stamps, used, amps = star_stamps(
        flat - amb_flat, good, seg, x, y, sel, halo,
    )

    edges = wing_edges()
    work = {'warp': warp - amb_warp}
    far_edges = edges
    mid_edges = edges[edges <= MID_RMAX]
    _, tfar = measure_profiles(
        work, vexp, stars, seg, gmax=FAR_GMAX, edges=far_edges,
    )
    _, tmid = measure_profiles(
        work, vexp, stars, seg, gmin=FAR_GMAX, gmax=AUR_GMAX,
        edges=mid_edges,
    )
    nb = edges.size - 1
    wing = np.zeros(tfar.size + tmid.size, dtype=[
        ('G', 'f4'), ('x', 'f8'), ('y', 'f8'),
        ('prof', 'f4', nb), ('npix', 'i4', nb),
    ])
    wing['prof'] = np.nan
    for i, row in enumerate(tfar):
        wing[i] = (row['G'], row['x'], row['y'],
                   row['prof'] * sig, row['npix'])
    nmid = mid_edges.size - 1
    for j, row in enumerate(tmid):
        i = tfar.size + j
        wing['G'][i], wing['x'][i], wing['y'][i] = row['G'], row['x'], row['y']
        wing['prof'][i, :nmid] = row['prof'] * sig
        wing['npix'][i, :nmid] = row['npix']
    # nJy per unit gaia flux
    wing['prof'] /= (10.0 ** (-0.4 * wing['G']))[:, np.newaxis]

    print(
        f'    fwhm {fwhm if fwhm is None else round(fwhm, 2)} arcsec, '
        f'sigma {sig:.1f} nJy, {stamps.shape[0]} stamps '
        f'({sel.size} candidates), {tfar.size} far + {tmid.size} mid '
        f'wing stars'
    )
    return dict(
        visit=vexp.visit, detector=vexp.detector, band=vexp.band,
        fwhm=np.nan if fwhm is None else float(fwhm), sky_sigma=sig,
        calib=vexp.calib, ambient_flat=amb_flat, ambient_warp=amb_warp,
        stamps=stamps.astype('f4'),
        stamp_G=gaia['phot_g_mean_mag'][used].astype('f4'),
        stamp_amp=amps.astype('f4'),
        wing=wing, edges=edges,
    )


def cloud_median(wing, edges, glo=None, ghi=None, min_stars=3):
    """
    the median flux-normalized profile over the stars in [glo,
    ghi), NaN where fewer than min_stars contribute; its error
    (1.253 x the MAD-scaled scatter over sqrt(n)); and the counts
    """
    sel = np.ones(wing.size, dtype=bool)
    if glo is not None:
        sel &= wing['G'] >= glo
    if ghi is not None:
        sel &= wing['G'] < ghi
    profs = wing['prof'][sel]
    nb = edges.size - 1
    med = np.full(nb, np.nan)
    err = np.full(nb, np.nan)
    if profs.shape[0] == 0:
        return med, err, np.zeros(nb, dtype=int)
    count = np.sum(np.isfinite(profs), axis=0)
    for k in np.flatnonzero(count >= min_stars):
        v = profs[:, k][np.isfinite(profs[:, k])]
        med[k] = np.median(v)
        mad = np.median(np.abs(v - med[k]))
        err[k] = 1.253 * 1.4826 * mad / np.sqrt(v.size)
    return med, err, count


def cloud_table(wing, edges, gbins=CLOUD_GBINS):
    """
    the per-G-bin cloud: one row per (G bin, radius) with the
    median, its error and the count; glo = -1 marks the all-star
    rows (diagnostic only, the fit uses the G bins: the faint
    stars outnumber the bright ones and their flux-normalized
    noise dominates the all-star median beyond ~300 px)
    """
    rmid = 0.5 * (edges[1:] + edges[:-1])
    rows = []
    for glo, ghi in [(None, None)] + list(gbins):
        m, e, c = cloud_median(wing, edges, glo, ghi)
        for k in range(rmid.size):
            rows.append((
                -1.0 if glo is None else glo, 99.0 if ghi is None else ghi,
                rmid[k], m[k], e[k], c[k], np.nan,
            ))
    return np.array(rows, dtype=[
        ('glo', 'f4'), ('ghi', 'f4'), ('rmid', 'f4'), ('med', 'f4'),
        ('err', 'f4'), ('count', 'i4'), ('model', 'f4'),
    ])


def stack_profile_errors(stamps, prof_size):
    """
    the error of the stack's azimuthal profile at each integer
    radius: the scatter of the per-stamp azimuthal means over
    sqrt(n), MAD-scaled and median-corrected
    """
    half = TMPL_HALF
    gy, gx = np.mgrid[-half:half + 1, -half:half + 1]
    rbin = np.round(np.hypot(gy, gx)).astype(int).ravel()
    err = np.full(prof_size, np.nan)
    means = []
    for st in stamps:
        v = st.ravel().astype('f8')
        ok = np.isfinite(v) & (rbin < prof_size)
        num = np.bincount(rbin[ok], weights=v[ok], minlength=prof_size)
        den = np.bincount(rbin[ok], minlength=prof_size)
        with np.errstate(invalid='ignore', divide='ignore'):
            means.append(num / den)
    means = np.array(means)
    for k in range(prof_size):
        v = means[:, k][np.isfinite(means[:, k])]
        if v.size >= 3:
            med = np.median(v)
            err[k] = 1.253 * 1.4826 * np.median(np.abs(v - med)) / np.sqrt(
                v.size,
            )
    return err


# the joint fit: stack radii, slope grids, the k_in scan about
# the continuity estimate
STACK_RFIT = (10, 50)
INNER_SLOPES = np.arange(-5.0, -2.5, 0.05)
AUR_SLOPES = np.arange(-3.5, -1.0, 0.05)
KIN_SCAN = np.exp(np.linspace(np.log(0.5), np.log(2.0), 29))


def fit_wing_model(prof, prof_err, cloud, rmin=AUR_RMIN, rmax=FAR_RMAX,
                   min_count=AUR_MIN_COUNT):
    """
    the two-power-law wing fit jointly to the stack profile
    (template units, STACK_RFIT radii, with a sky pedestal) and
    the per-G-bin cloud (nJy per unit gaia flux, rmin to rmax):

        stack(r) = a r^s1 + b r^s2 + c
        cloud(r) = k_in (a r^s1 + b r^s2)

    a, b, c linear at each (s1, s2, k_in) of the grids; k_in is
    scanned about its continuity estimate (cloud over stack where
    they overlap).  The single-law stack fit alone cannot give
    the inner law: the cloud is already flatter than the stack's
    22-50 px slope by 50 px, so the two components overlap there

    Returns
    -------
    dict with slope (s1), ln_a, aur_slope (s2), aur_amp (b),
    pedestal (c), k_in, chi2, npt, chi2_stack, chi2_cloud
    """
    r0, r1 = STACK_RFIT
    rs = np.arange(r0, min(r1, prof.size - 1) + 1).astype('f8')
    ys = prof[rs.astype(int)].astype('f8')
    es = prof_err[rs.astype(int)].astype('f8')
    oks = np.isfinite(ys) & np.isfinite(es) & (es > 0)
    rs, ys, es = rs[oks], ys[oks], es[oks]

    usable = (
        (cloud['glo'] >= 0) & np.isfinite(cloud['med'])
        & np.isfinite(cloud['err']) & (cloud['err'] > 0)
        & (cloud['count'] >= min_count)
        & (cloud['rmid'] >= rmin) & (cloud['rmid'] <= rmax)
    )
    rc = cloud['rmid'][usable].astype('f8')
    yc = cloud['med'][usable].astype('f8')
    ec = cloud['err'][usable].astype('f8')
    if rc.size < 3 or rs.size < 5:
        raise RuntimeError(
            f'wing fit: {rs.size} stack and {rc.size} cloud points'
        )

    # continuity estimate of k_in: cloud over stack profile where
    # both exist (the stack to prof.size - 1)
    over = rc < prof.size - 1
    if over.sum() == 0:
        raise RuntimeError('wing fit: no cloud point overlaps the stack')
    k0 = float(np.median(yc[over] / np.interp(rc[over], np.arange(
        prof.size), prof)))

    ws, wc = 1.0 / es, 1.0 / ec
    best = None
    for k_in in k0 * KIN_SCAN:
        for s1 in INNER_SLOPES:
            for s2 in AUR_SLOPES:
                if s2 < s1 + AUR_SLOPE_SEP:
                    continue
                a_s = np.vstack([
                    rs ** s1, rs ** s2, np.ones(rs.size),
                ]).T * ws[:, None]
                a_c = np.vstack([
                    k_in * rc ** s1, k_in * rc ** s2, np.zeros(rc.size),
                ]).T * wc[:, None]
                basis = np.vstack([a_s, a_c])
                y = np.concatenate([ys * ws, yc * wc])
                coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
                if not (coef[0] > 0 and coef[1] > 0):
                    continue
                resid = basis @ coef - y
                chi2 = float(np.sum(resid ** 2))
                if best is None or chi2 < best[0]:
                    best = (chi2, s1, s2, k_in, coef,
                            float(np.sum(resid[:rs.size] ** 2)))
    if best is None:
        raise RuntimeError('wing fit found no positive solution')
    chi2, s1, s2, k_in, coef, chi2_stack = best
    return dict(
        slope=float(s1), ln_a=float(np.log(coef[0])),
        aur_slope=float(s2), aur_amp=float(coef[1]),
        pedestal=float(coef[2]), k_in=float(k_in), chi2=chi2,
        npt=int(rs.size + rc.size), chi2_stack=chi2_stack,
        chi2_cloud=chi2 - chi2_stack, k_in0=k0,
    )


def wing_law(r, slope, ln_a, aur_slope, aur_amp):
    """the analytic halo in template units at radius r"""
    rc = np.maximum(np.asarray(r, dtype='f8'), 1.0)
    return np.exp(ln_a) * rc ** slope + aur_amp * rc ** aur_slope


def pool_visit(extracts):
    """
    the pooled template and wing model of one visit

    Parameters
    ----------
    extracts: list of dict
        From extract_detector

    Returns
    -------
    dict with template (extended array), stack (inner, denoised,
    pedestal removed), prof (its azimuthal profile), params
    dict, cloud (structured array per G bin and radius: glo,
    ghi, rmid, med, count, model), wing (all stars, with a
    detector column), detectors (structured array), edges
    """
    band = extracts[0]['band']
    canon = CANON.get(band)
    stamps = np.concatenate([e['stamps'] for e in extracts if
                             e['stamps'].size > 0])
    nstamp = stamps.shape[0]
    tmpl = np.nanmedian(stamps, axis=0)
    tmpl[~np.isfinite(tmpl)] = 0.0
    tmpl = 0.5 * (tmpl + tmpl[::-1, ::-1])
    tmpl, prof = denoise_template(tmpl)
    prof_err = stack_profile_errors(stamps, prof.size)
    # the single-law stack fit, for the record and as a fallback
    slope1, ln_a1, ped1 = fit_halo_slope(
        prof, fallback_slope=canon['slope'] if canon else HALO_SLOPE,
    )

    edges = extracts[0]['edges']
    wings = []
    for e in extracts:
        w = np.zeros(e['wing'].size, dtype=[('detector', 'i4')]
                     + e['wing'].dtype.descr)
        w['detector'] = e['detector']
        for name in e['wing'].dtype.names:
            w[name] = e['wing'][name]
        wings.append(w)
    wing = np.concatenate(wings)

    # the direct zero point from the stamps' core amplitudes
    # (nJy per unit gaia flux, in core-normalized template units),
    # when the extracts carry them; the joint fit's k_in is the
    # same quantity from the cloud-stack continuity
    k_stamp = np.nan
    amps = [e['stamp_amp'] for e in extracts if 'stamp_amp' in e]
    if len(amps) > 0:
        amps = np.concatenate(amps)
        gs = np.concatenate([e['stamp_G'] for e in extracts
                             if 'stamp_amp' in e])
        k_stamp = float(np.median(amps / 10.0 ** (-0.4 * gs)))

    cloud = cloud_table(wing, edges)
    fit = fit_wing_model(prof, prof_err, cloud)
    slope, ln_a, ped = fit['slope'], fit['ln_a'], fit['pedestal']
    aur = fit
    tmpl = tmpl - ped
    prof = prof - ped
    cloud['model'] = aur['k_in'] * wing_law(
        cloud['rmid'], slope, ln_a, aur['aur_slope'], aur['aur_amp'],
    )

    detectors = np.array([
        (e['detector'], e['fwhm'], e['sky_sigma'], e['calib'],
         e['ambient_flat'], e['ambient_warp'], e['stamps'].shape[0],
         e['wing'].size)
        for e in extracts
    ], dtype=[
        ('detector', 'i4'), ('fwhm', 'f4'), ('sky_sigma', 'f4'),
        ('calib', 'f4'), ('ambient_flat', 'f4'), ('ambient_warp', 'f4'),
        ('nstamp', 'i4'), ('nwing', 'i4'),
    ])
    fwhm = float(np.nanmedian(detectors['fwhm']))
    template = extend_template_halo(
        tmpl, slope, ln_a, aur['aur_slope'], aur['aur_amp'],
        out_half=TMPL_OUT_MAX + 2,
    )
    params = dict(
        visit=int(extracts[0]['visit']), band=band, ndet=len(extracts),
        nstamp=int(nstamp), nwing=int(wing.size), fwhm=fwhm,
        slope=float(slope), ln_a=float(ln_a), pedestal=float(ped),
        aur_slope=aur['aur_slope'], aur_amp=aur['aur_amp'],
        k_in=aur['k_in'], k_in0=aur['k_in0'], k_stamp=k_stamp,
        chi2=aur['chi2'],
        npt=aur['npt'], chi2_stack=aur['chi2_stack'],
        chi2_cloud=aur['chi2_cloud'],
        stack_slope=float(slope1), stack_ln_a=float(ln_a1),
        stack_pedestal=float(ped1),
    )
    print(
        f'    pooled: {nstamp} stamps from {len(extracts)} detectors, '
        f'fwhm {fwhm:.2f}; joint fit: inner slope {slope:.2f} '
        f'ln_a {ln_a:.3f} pedestal {ped:.1e}, aureole slope '
        f'{aur["aur_slope"]:.2f} amp {aur["aur_amp"]:.2e}, k_in '
        f'{aur["k_in"]:.3e} (continuity {aur["k_in0"]:.3e}, stamps '
        f'{k_stamp:.3e}); chi2 '
        f'{aur["chi2_stack"]:.1f} stack + {aur["chi2_cloud"]:.1f} cloud '
        f'for {aur["npt"]} points, {wing.size} stars; single-law stack '
        f'slope {slope1:.2f}'
    )
    return dict(
        template=template.astype('f4'), stack=tmpl.astype('f4'),
        prof=prof, prof_err=prof_err, params=params, cloud=cloud,
        wing=wing, detectors=detectors, edges=edges,
    )


def write_template_file(fname, pooled):
    """
    extensions: template, stack, params (one row), cloud, wing,
    detectors, edges, prof
    """
    import rustfits

    p = pooled['params']
    params = np.zeros(1, dtype=[
        (k, 'S8' if isinstance(v, str) else ('i8' if isinstance(v, int)
                                             else 'f8'))
        for k, v in p.items()
    ])
    for k, v in p.items():
        params[k] = v
    edges_t = np.zeros(1, dtype=[('edges', 'f8', pooled['edges'].size)])
    edges_t['edges'][0] = pooled['edges']
    prof_t = np.zeros(1, dtype=[('prof', 'f8', pooled['prof'].size),
                                ('err', 'f8', pooled['prof'].size)])
    prof_t['prof'][0] = pooled['prof']
    prof_t['err'][0] = pooled['prof_err']
    print('writing:', fname)
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_image(pooled['template'], extname='template')
        fits.write_image(pooled['stack'], extname='stack')
        fits.write_table(params, extname='params')
        fits.write_table(pooled['cloud'], extname='cloud')
        fits.write_table(pooled['wing'], extname='wing')
        fits.write_table(pooled['detectors'], extname='detectors')
        fits.write_table(edges_t, extname='edges')
        fits.write_table(prof_t, extname='prof')


def read_template_file(fname):
    import rustfits

    out = {}
    with rustfits.FITS(fname) as fits:
        for name in ('template', 'stack', 'cloud', 'wing', 'detectors'):
            out[name] = fits[name].read()
        params = fits['params'].read()
        out['params'] = {}
        for k in params.dtype.names:
            v = params[k][0]
            if params[k].dtype.kind in 'SU':
                v = v.decode() if isinstance(v, bytes) else str(v)
                v = v.strip()
            else:
                v = v.item()
            out['params'][k] = v
        out['edges'] = fits['edges'].read()['edges'][0]
        pt = fits['prof'].read()
        out['prof'] = pt['prof'][0]
        out['prof_err'] = pt['err'][0]
    return out


def plot_template(png, pooled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = pooled['params']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    prof = pooled['prof']
    r = np.arange(prof.size)
    ax.errorbar(r[1:], np.abs(prof[1:]), yerr=pooled['prof_err'][1:],
                fmt='k.', ms=3, label='stack profile')
    ax.set_yscale('log')
    rr = np.logspace(np.log10(3), np.log10(TMPL_OUT_MAX), 200)
    ax.semilogy(rr, np.exp(p['ln_a']) * rr ** p['slope'], 'b-',
                label=f'inner r^{p["slope"]:.2f}')
    ax.semilogy(rr, p['aur_amp'] * rr ** p['aur_slope'], 'g-',
                label=f'aureole r^{p["aur_slope"]:.2f}')
    ax.semilogy(rr, wing_law(rr, p['slope'], p['ln_a'], p['aur_slope'],
                             p['aur_amp']), 'r-', lw=0.8, label='halo')
    ax.set_xscale('log')
    ax.set_xlabel('r [px]')
    ax.set_ylabel('template units')
    ax.set_title(f'{p["nstamp"]} stamps, fwhm {p["fwhm"]:.2f}"')
    ax.legend(fontsize=7)

    ax = axes[1]
    cloud = pooled['cloud']
    for glo, ghi in CLOUD_GBINS:
        w = (cloud['glo'] == glo) & (cloud['ghi'] == ghi)
        m, e = cloud['med'][w], cloud['err'][w]
        ok = np.isfinite(m) & (m > 0)
        ax.errorbar(cloud['rmid'][w][ok], m[ok], yerr=e[ok], fmt='.',
                    ms=4, capsize=2, label=f'G {glo:g}-{ghi:g}')
        neg = np.isfinite(m) & (m <= 0)
        if neg.any():
            ax.plot(cloud['rmid'][w][neg], -m[neg], 'x', ms=4,
                    color=ax.lines[-1].get_color())
    w = cloud['glo'] == -1.0
    ax.plot(cloud['rmid'][w], cloud['model'][w], 'r-', lw=1.2,
            label='k_in x halo')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r [px]')
    ax.set_ylabel('nJy per unit gaia flux')
    ax.set_title(f'wing cloud, {p["nwing"]} stars (x: negative)')
    ax.legend(fontsize=7)

    ax = axes[2]
    d = pooled['detectors']
    ax.plot(d['detector'], d['fwhm'], 'k.', label='fwhm ["]')
    ax.set_xlabel('detector')
    ax.set_ylabel('fwhm [arcsec]')
    ax2 = ax.twinx()
    ax2.plot(d['detector'], d['nstamp'], 'b+', label='stamps')
    ax2.set_ylabel('stamps', color='b')
    ax.set_title(f'{p["ndet"]} detectors')
    fig.suptitle(f'visit {p["visit"]} {p["band"]}')
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print('wrote', png)


def canonical_wing(files, rmax=3000.0):
    """
    the canonical physical wing of a band: the median over visit
    template files of k_in T_v(r), the wing in nJy per unit Gaia
    flux, on a common radial grid.  No per-visit information
    remains; the test of whether one wing per band suffices

    Returns
    -------
    r, T (nJy per unit flux), and the per-visit curves (nvisit, nr)
    """
    from .trough import radial_template

    curves = []
    r0 = None
    for f in files:
        t = read_template_file(f)
        r, T = radial_template(t)
        if r0 is None:
            r0 = r
        curves.append(t['params']['k_in'] * np.interp(r0, r, T))
    curves = np.array(curves)
    return r0, np.median(curves, axis=0), curves


def write_canonical_wing(fname, r, T, band, nvisit):
    import rustfits

    tab = np.zeros(r.size, dtype=[('r', 'f8'), ('T', 'f8')])
    tab['r'], tab['T'] = r, T
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(tab, extname='wing',
                         header={'band': band, 'nvisit': int(nvisit)})


def read_canonical_wing(fname):
    import rustfits

    with rustfits.FITS(fname) as fits:
        tab = fits['wing'].read()
    return tab['r'].astype('f8'), tab['T'].astype('f8')
