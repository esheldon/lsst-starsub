"""
The per-detector amplitude scale of a visit's pass-1 fits against the
seeing and the field radius.

For every detector: the median core amplitude (the unsaturated stars,
free == 2), the median wing-fit amplitude of the well-constrained bright
stars (free == 1, A_err below --max-err), the fit chi2 and the detector
fwhm from the profiles meta, and the field radius from a previous
per_detector_amplitudes.npy in the run directory (or the butler when
absent).  Prints the slope of the core scale against the fwhm, the
number that says whether the core amplitudes still track the local
seeing, and writes the table.

usage: python core_scale_by_detector.py INDIR VISIT OUT.npy [--radius FILE]
"""
import argparse

import numpy as np

from lsst_starsub.visit.gather import profiles_files, read_detector_stars


def main():
    p = argparse.ArgumentParser()
    p.add_argument('indir')
    p.add_argument('visit', type=int)
    p.add_argument('outfile')
    p.add_argument('--radius', default=None,
                   help='a per_detector_amplitudes.npy with det and r_fp; '
                        'default the butler')
    p.add_argument('--max-err', type=float, default=0.2,
                   help='wing-fit stars with A_err below this count as '
                        'well constrained (a single visit constrains the '
                        'G < 13 stars to 0.08-0.2)')
    p.add_argument('--repo', default='dp2_prep_future')
    p.add_argument('--collection', default='LSSTCam/runs/DRP/DP2')
    args = p.parse_args()

    radius = {}
    if args.radius is not None:
        prev = np.load(args.radius)
        radius = dict(zip(prev['det'], prev['r_fp']))
    else:
        from lsst.daf.butler import Butler
        from lsst_starsub.visit.exposure import detector_field_radius
        butler = Butler(args.repo, collections=args.collection)

    dt = [('det', 'i4'), ('r_fp', 'f8'), ('ncore', 'i4'), ('A_core', 'f8'),
          ('nwing', 'i4'), ('A_wing', 'f8'), ('chi2', 'f8'), ('fwhm', 'f8')]
    rows = []
    for fname in profiles_files(args.indir, args.visit):
        stars, meta = read_detector_stars(fname)
        det = int(meta['detector'][0])
        if det not in radius:
            radius[det] = detector_field_radius(butler, det)
        core = stars['free'] == 2
        wing = (stars['free'] == 1) & (stars['A_err'] < args.max_err)
        rows.append((
            det, radius[det],
            core.sum(), np.median(stars['A'][core]) if core.any() else np.nan,
            wing.sum(), np.median(stars['A'][wing]) if wing.any() else np.nan,
            float(meta['chi2'][0]), float(meta['fwhm'][0]),
        ))
    t = np.array(rows, dtype=dt)
    np.save(args.outfile, t)

    ok = np.isfinite(t['A_core']) & np.isfinite(t['fwhm'])
    fwhm, ac = t['fwhm'][ok], t['A_core'][ok]
    slope, icpt = np.polyfit(fwhm, ac, 1)
    corr = np.corrcoef(fwhm, ac)[0, 1]
    resid = ac - (slope * fwhm + icpt)
    print(f'visit {args.visit}: {t.size} detectors, fwhm '
          f'{fwhm.min():.2f}-{fwhm.max():.2f} arcsec')
    print(f'core scale: median {np.median(ac):.3f}, rms over detectors '
          f'{ac.std():.3f}; vs fwhm slope {slope:+.3f} per arcsec, '
          f'corr {corr:+.2f}, rms about the line {resid.std():.3f}')
    inner = ok & (t['r_fp'] < 200)
    outer = ok & (t['r_fp'] >= 200)
    for name, sel in [('r < 200 mm', inner), ('r >= 200 mm', outer)]:
        if sel.any():
            print(f'  {name}: {sel.sum()} detectors, core median '
                  f'{np.median(t["A_core"][sel]):.3f}, wing median '
                  f'{np.nanmedian(t["A_wing"][sel]):.3f}')
    print(f'wrote {args.outfile}')


if __name__ == '__main__':
    main()
