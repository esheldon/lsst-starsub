"""
Two lsst-mdet catalogs of the same cells side by side: the objects
matched per metacal step, the shape and flux differences against
their errors, and the metacal shear response and mean shear of each.

usage: python compare_mdet.py LABEL_A=CAT_A.fits LABEL_B=CAT_B.fits
           [--s2n 10] [--radius 1.0]
"""
import argparse

import numpy as np
import rustfits


def match(a, b, radius):
    """indices into a and b of the pairs within radius px, per step"""
    from scipy.spatial import cKDTree

    ia, ib = [], []
    for step in np.unique(a['mcal_step']):
        sa = np.flatnonzero(a['mcal_step'] == step)
        sb = np.flatnonzero(b['mcal_step'] == step)
        if sa.size == 0 or sb.size == 0:
            continue
        tree = cKDTree(np.c_[b['x'][sb], b['y'][sb]])
        d, j = tree.query(np.c_[a['x'][sa], a['y'][sa]])
        ok = d < radius
        ia.extend(sa[ok])
        ib.extend(sb[j[ok]])
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def response(cat, sel):
    """the metacal response R11 (the 1p/1m steps) and the mean shear
    g1, g2 over R11 of a catalog; the steps are ns, 1p, 1m"""
    out = {}
    p = sel & (cat['mcal_step'] == '1p')
    m = sel & (cat['mcal_step'] == '1m')
    ns = sel & (cat['mcal_step'] == 'ns')
    r = (cat['g1'][p].mean() - cat['g1'][m].mean()) / 0.02 \
        if p.any() and m.any() else np.nan
    out['R11'] = r
    out['n'] = int(ns.sum())
    for c in ('1', '2'):
        g = cat[f'g{c}'][ns]
        out[f'g{c}'] = g.mean() / r if np.isfinite(r) else np.nan
        out[f'g{c}_err'] = g.std() / np.sqrt(max(1, g.size)) / r \
            if np.isfinite(r) else np.nan
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs=2)
    p.add_argument('--s2n', type=float, default=10.0)
    p.add_argument('--radius', type=float, default=1.0)
    args = p.parse_args()
    (la, fa), (lb, fb) = [r.split('=', 1) for r in args.runs]
    a = rustfits.read(fa, ext='cat')
    b = rustfits.read(fb, ext='cat')
    print(f'{la}: {a.size} rows; {lb}: {b.size} rows')
    for label, cat in ((la, a), (lb, b)):
        sel = ((cat['flags'] == 0) & (cat['g_flags'] == 0)
               & (cat['s2n'] > args.s2n))
        r = response(cat, sel)
        print(f'  {label:20s} flags 0, s2n > {args.s2n:.0f}: {r["n"]} '
              f'noshear objects; R11 {r["R11"]:.3f}; '
              f'<g1> {r["g1"]:+.4f} +- {r["g1_err"]:.4f}, '
              f'<g2> {r["g2"]:+.4f} +- {r["g2_err"]:.4f}')
    ia, ib = match(a, b, args.radius)
    print(f'matched {ia.size} rows within {args.radius} px')
    if ia.size == 0:
        return
    ma, mb = a[ia], b[ib]
    ok = ((ma['flags'] == 0) & (mb['flags'] == 0) & (ma['g_flags'] == 0)
          & (mb['g_flags'] == 0) & (ma['s2n'] > args.s2n))
    ma, mb = ma[ok], mb[ok]
    print(f'  {ok.sum()} matched with flags 0 and s2n > {args.s2n:.0f} in '
          f'{la}')
    for col in ('g1', 'g2', 'T'):
        err = np.hypot(ma[f'{col}_err'], mb[f'{col}_err'])
        d = (mb[col] - ma[col])
        rel = np.median(np.abs(d) / err)
        print(f'  {col:2s}: {lb} - {la} median {np.median(d):+.4f}, rms '
              f'{d.std():.4f}, median |diff| / err {rel:.3f}')
    for col in [c for c in ma.dtype.names if c.startswith('flux_')
                and not c.startswith('flux_err')]:
        band = col.split('_')[1]
        err = np.hypot(ma[f'flux_err_{band}'], mb[f'flux_err_{band}'])
        d = mb[col] - ma[col]
        print(f'  {col}: {lb} - {la} median {np.median(d):+.2f} nJy, '
              f'median |diff| / err {np.median(np.abs(d) / err):.3f}, '
              f'median relative {np.median(d / ma[col]):+.4f}')


if __name__ == '__main__':
    main()
