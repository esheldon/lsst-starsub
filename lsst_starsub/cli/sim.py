"""
cli/sim: the ideal-conditions simulation of one patch (see
lsst_starsub.visit.sim), followed by the cleaning of the chosen coadd
state with every star's truth known.

Writes {outdir}/sim-{tract}-{patch}-{band}-s{seed}.fits (the
coadd states, the response, the truth star and sky coadds, the
visit table) and {outdir}/clean-sim{seed}-{tag}-{tract}-{patch}-
{band}.fits in the lsst-starsub-cell-clean layout (read by
scripts/compare_inject.py with tag sim{seed}-{state}...), plus a
'trough_dmask' table: the d - r_mask profiles of none - (raw -
sky) (the trough: the delivered coadd minus the ideal one) and
of forward - (raw - sky) (what the response coadd leaves)
"""
import os
import sys

import numpy as np


def get_args():
    """
    Parse the command line.
    """
    import argparse
    from ..visit.sim import DEFAULTS
    from . import add_butler_arguments
    from ..census import GSUB

    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, default=7275)
    parser.add_argument('--patch', type=int, default=55)
    parser.add_argument('--band', default='i')
    parser.add_argument('--gaia-file', required=True)
    parser.add_argument('--canonical', required=True)
    parser.add_argument('--outdir', required=True)
    add_butler_arguments(parser)
    parser.add_argument('--seed', type=int, default=None,
                        help='default: the patch number')
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--sim-file', help='reuse this simulation')
    parser.add_argument('--sim-only', action='store_true')
    for k, v in DEFAULTS.items():
        parser.add_argument(f'--{k.replace("_", "-")}', type=type(v),
                            default=v)
    parser.add_argument('--state', default='forward',
                        choices=['forward', 'none', 'restored'])
    parser.add_argument('--gsub', type=float, default=GSUB)
    parser.add_argument('--nround', type=int, default=2)
    parser.add_argument('--star-model', default='joint',
                        choices=['joint', 'template', 'canonical'])
    parser.add_argument('--bright-grow', type=float, default=None)
    parser.add_argument('--joint-spacing', type=float, default=None,
                        help='joint model: sky mesh node spacing (px)')
    parser.add_argument('--joint-prior', type=float, default=None,
                        help='joint model: amplitude prior width about the '
                             'prediction (default lsst_starsub.joint.'
                             'PRIOR_SIGMA; 0 disables)')
    parser.add_argument('--no-images', action='store_true')
    return parser.parse_args()


def sim_geometry(butler, tract, patch, margin=150):
    """
    Get the tract wcs and the cell-coadd-sized patch box.

    The tract wcs and the cell-coadd-sized patch box
    """
    from ..geom import ButlerWcs, SimpleBox
    from ..site import SKYMAP

    skymap = butler.get('skyMap', skymap=SKYMAP)
    tr = skymap[tract]
    bb = tr[patch].getInnerBBox()
    tbox = SimpleBox(bb.getBeginX() - margin, bb.getEndX() + margin,
                     bb.getBeginY() - margin, bb.getEndY() + margin)
    return ButlerWcs(tr.getWcs()), tbox


def write_sim_file(fname, sim, truth_table, cfg, meta):
    """
    Write the simulation file.
    """
    import rustfits
    from ..coadd.io import _meta_table

    with rustfits.FITS(fname, 'w+') as fits:
        for name in ('none', 'response', 'stars', 'sky', 'raw', 'var'):
            fits.write_image(np.ascontiguousarray(sim[name], dtype='f4'),
                             extname=name, compress='gzip_2')
        fits.write_image(sim['satmask'].astype('u1'), extname='satmask',
                         compress='gzip_2')
        fits.write_table(sim['visits'], extname='visits')
        fits.write_table(truth_table, extname='truth_stars')
        fits.write_table(_meta_table({**cfg, **meta}), extname='meta')


def read_sim_file(fname):
    """
    Read a simulation file.
    """
    import rustfits

    sim = {}
    with rustfits.FITS(fname) as fits:
        for name in ('none', 'response', 'stars', 'sky', 'raw', 'var'):
            sim[name] = fits[name].read()
        sim['satmask'] = fits['satmask'].read().astype(bool)
        sim['visits'] = fits['visits'].read()
        truth_table = fits['truth_stars'].read()
    return sim, truth_table


def main():
    """
    Simulate a patch and clean it.
    """
    from ..gaia import GMAX, gaia_pixel_positions, read_gaia_file
    from ..maskbits import DM_SAT
    from ..coadd.clean import (
        clean_stem,
        clean_tag,
        run_clean,
        write_clean_file,
    )
    from ..visit.profiles import ambient_levels, measure_profiles
    from ..visit.sim import DEFAULTS, simulate_coadd
    from ..wing import read_wing_model
    from ..visit.exposure import VisitExposure, make_visit_butler

    sys.stdout.reconfigure(line_buffering=True)
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    seed = args.patch if args.seed is None else args.seed
    cfg = {k: getattr(args, k) for k in DEFAULTS}

    butler = make_visit_butler(args.repo, args.collection)
    wcs, tbox = sim_geometry(butler, args.tract, args.patch)
    shape = (tbox.y.stop - tbox.y.start, tbox.x.stop - tbox.x.start)
    gaia = read_gaia_file(args.gaia_file, wcs, tbox,
                          gmax=max(args.gsub, GMAX))
    x, y = gaia_pixel_positions(gaia, wcs, tbox)
    G = gaia['phot_g_mean_mag'].astype('f8')
    canonical = read_wing_model(args.canonical)

    sim_name = os.path.join(
        args.outdir,
        f'sim-{args.tract:05d}-{args.patch:02d}-{args.band}-s{seed}.fits',
    )
    if args.sim_file:
        print('reading', args.sim_file)
        sim, truth_table = read_sim_file(args.sim_file)
        amp = truth_table['wing_scale'].astype('f8')
    else:
        rng = np.random.default_rng(seed)
        amp = np.ones(gaia.size)
        if cfg['amp_scatter'] > 0:
            amp = np.exp(rng.normal(size=gaia.size) * cfg['amp_scatter'])
        print(f'simulating {cfg["nvisit"]} visits of {shape} with '
              f'{gaia.size} stars (seed {seed})')
        sim = simulate_coadd(shape, x, y, G, amp, canonical, cfg, seed=seed,
                             nproc=args.nproc)
        truth_table = np.zeros(gaia.size, dtype=[
            ('x', 'f8'), ('y', 'f8'), ('G', 'f4'), ('ra', 'f8'),
            ('dec', 'f8'), ('core_flux', 'f8'), ('wing_scale', 'f8'),
        ])
        truth_table['x'], truth_table['y'], truth_table['G'] = x, y, G
        truth_table['ra'], truth_table['dec'] = gaia['ra'], gaia['dec']
        truth_table['core_flux'] = 10.0 ** (-0.4 * G)
        truth_table['wing_scale'] = amp
        meta = dict(tract=args.tract, patch=args.patch, band=args.band,
                    seed=seed, nstar=int(gaia.size))
        print('writing', sim_name)
        write_sim_file(sim_name, sim, truth_table, cfg, meta)
    v = sim['visits']
    print(f'    fwhm {v["fwhm"].min():.2f}-{v["fwhm"].max():.2f}", '
          f'thresholds {v["threshold"].min():.0f}-{v["threshold"].max():.0f}, '
          f'detected fractions {v["detected_fraction"].min():.2f}-'
          f'{v["detected_fraction"].max():.2f}')
    if args.sim_only:
        return

    none = sim['none']
    if args.state == 'forward':
        image = none + sim['response']
    elif args.state == 'restored':
        image = sim['raw'] - sim['sky']
    else:
        image = none.copy()
    mask = np.zeros(shape + (1,), dtype='i4')
    mask[:, :, 0][sim['satmask']] = DM_SAT
    vexp = VisitExposure(
        image=image, variance=sim['var'], mask=mask, band=args.band,
        backgrounds={}, wcs=wcs, visit=args.tract, detector=args.patch,
        noise=np.zeros(shape, dtype='f4'),
    )
    vexp.bbox = tbox
    print(f'    state {args.state}: sky sigma {vexp.sky_sigma:.2f} nJy')
    truth = sim['stars']
    out = run_clean(
        vexp, gaia, tbox, args.state, args.gsub, args.nround,
        args.bright_grow, args.star_model, canonical,
        truth=truth, inj=truth_table,
        joint_spacing=args.joint_spacing,
        joint_prior=None if args.joint_prior is None
        else (None if args.joint_prior <= 0 else args.joint_prior),
    )

    # the trough (delivered minus the ideal delivered, raw - sky)
    # and what the response coadd leaves of it
    ideal = sim['raw'] - sim['sky']
    tstates = {
        'trough_none': none - ideal,
        'trough_forward': none + sim['response'] - ideal,
    }
    del ideal
    tamb = ambient_levels(tstates, vexp, out['seg'], np.zeros(shape, bool))
    _, ttable = measure_profiles(
        tstates, vexp, out['res']['stars'], out['seg'], ambient=tamb,
        mode='dmask',
    )
    res = out['res']
    meta = dict(
        tract=args.tract, patch=args.patch, band=args.band,
        state=args.state, collection='sim', gsub=args.gsub,
        nround=args.nround, sky_sigma=out['sky_sigma'],
        fwhm=res['fwhm'] if res['fwhm'] is not None else -1.0,
        bright_grow=-1.0 if args.bright_grow is None else args.bright_grow,
        star_model=args.star_model, inject='sim', inject_wing_scale=1.0,
        seed=seed, sim_file=sim_name,
        **cfg,
    )
    tag = f'sim{seed}-' + clean_tag(args.state, args.star_model,
                                    joint_spacing=args.joint_spacing)
    stem = clean_stem(args.outdir, tag, args.tract, args.patch, args.band)
    write_clean_file(stem, out, meta, no_images=args.no_images,
                     extra_tables={'trough_dmask': ttable})


if __name__ == '__main__':
    main()
