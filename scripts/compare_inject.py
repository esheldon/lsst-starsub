"""
the injection test: around the injected stars (truth known) the
stacked d - r_mask profiles of the final residual, of the residual
under a perfect star model ('perfect' = flat - truth, the sky
interpolation alone) and of the star model's error ('model_error'
= model - truth); next to them the real stars' residual from the
same runs.  Profiles in units of the coadd sky sigma; 3-sigma
clipped means with errors

usage: python compare_inject.py CLEANDIR TAG PATCH [PATCH ...]

reads {CLEANDIR}/clean-{TAG}-07275-{patch:02d}-i.fits; the
INJECT_TAGS env var gives several tags (comma separated) to
overlay, e.g. forward-inj,forward-inj13,forward-inj10
"""
import os
import sys
import numpy as np
import rustfits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

cdir = sys.argv[1]
tags = os.environ.get('INJECT_TAGS', sys.argv[2]).split(',')
patches = [int(p) for p in sys.argv[3:]]
tract = int(os.environ.get('TRACT', 7275))
band = os.environ.get('BAND', 'i')
gbins = [(6, 13), (13, 14), (14, 15.2), (15.2, 16), (16, 17)]
picks = [5, 10, 20, 30, 45]
curves = [
    ('residual', 1, 'injected residual'),
    ('perfect', 1, 'perfect-model residual (sky)'),
    ('model_error', 1, 'model - truth'),
    ('residual', 0, 'real residual'),
]


def clipped_mean(a, nsig=3.0, niter=3):
    a = np.asarray(a, dtype='f8')
    out_m = np.full(a.shape[1], np.nan)
    out_e = np.full(a.shape[1], np.nan)
    for k in range(a.shape[1]):
        v = a[:, k][np.isfinite(a[:, k])]
        if v.size < 2:
            continue
        for _ in range(niter):
            m, s = np.mean(v), np.std(v)
            keep = np.abs(v - m) <= nsig * s
            if keep.all() or keep.sum() < 2:
                break
            v = v[keep]
        out_m[k] = np.mean(v)
        out_e[k] = np.std(v) / np.sqrt(v.size)
    return out_m, out_e


from lsst_mdet.starsub import circle_radius  # noqa
from lsst_starsub.template import read_canonical_wing  # noqa

canonical = os.environ.get(
    'CANONICAL',
    os.path.expanduser('~/oh/starsub-visits/templates/canonical-wing-07275-i.fits'),
)
rc, Tc = read_canonical_wing(canonical)

tables = {}
truths = {}
edges = None
for tag in tags:
    rows = []
    trows = []
    for patch in patches:
        # simulation tags carry the seed, the patch number by default
        t = tag.replace('simSEED', f'sim{patch}')
        f = f'{cdir}/clean-{t}-{tract:05d}-{patch:02d}-{band}.fits'
        if not os.path.exists(f):
            continue
        with rustfits.FITS(f) as fits:
            names = [h.extname for h in fits]
            prof = fits[os.environ.get('PROFILE_TABLE', 'profiles_dmask')].read()
            edges = fits['dmask_edges'].read()['edges'][0]
            inj = fits['injected'].read() if 'injected' in names else []
            sig = float(fits['meta'].read()['sky_sigma'][0])
        if 'injected' not in prof.dtype.names:
            from numpy.lib import recfunctions as rfn
            prof = rfn.append_fields(prof, 'injected',
                                     np.zeros(prof.size, 'i2'), usemask=False)
        rows.append(prof)
        dm = 0.5 * (edges[1:] + edges[:-1])
        for st in inj:
            r = circle_radius(float(st['G'])) + dm
            t = (10.0 ** (-0.4 * float(st['G'])) * float(st['wing_scale'])
                 * np.interp(r, rc, Tc, right=0.0) / sig)
            trows.append((float(st['G']), t))
    if rows:
        tables[tag] = np.concatenate(rows)
        truths[tag] = trows
dmid = 0.5 * (edges[1:] + edges[:-1])
print('3-sigma clipped mean [10^-3 sigma] at d - r_mask =',
      [int(dmid[p]) for p in picks])

ntag = len(tables)
fig, axes = plt.subplots(ntag, len(gbins), figsize=(4.2 * len(gbins),
                                                    4 * ntag),
                         squeeze=False)
for row, (tag, t) in enumerate(tables.items()):
    print(f'{tag}:')
    for col, (glo, ghi) in enumerate(gbins):
        ax = axes[row, col]
        for i, (state, injected, label) in enumerate(curves):
            w = ((t['state'] == state) & (t['injected'] == injected)
                 & (t['G'] >= glo) & (t['G'] < ghi))
            if w.sum() == 0:
                continue
            m, e = clipped_mean(t['prof'][w] * 1e3)
            ls = '--' if injected == 0 else '-'
            ax.errorbar(dmid, m, yerr=e, fmt='.' + ls, ms=3, color=f'C{i}',
                        label=f'{label} ({w.sum()})')
            print(f'  G {glo:4.1f}-{ghi:4.1f} {label:28s} n={w.sum():4d}: '
                  + ' '.join(f'{m[p]:6.1f}+-{e[p]:4.1f}' for p in picks))
        tw = np.array([t for g, t in truths[tag] if glo <= g < ghi])
        if tw.size:
            m = np.mean(tw * 1e3, axis=0)
            ax.plot(dmid, m, 'k:', lw=1, label='truth wing (mean)')
            print(f'  G {glo:4.1f}-{ghi:4.1f} {"truth wing (mean)":28s} '
                  f'n={tw.shape[0]:4d}: '
                  + ' '.join(f'{m[p]:6.1f}     ' for p in picks))
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylim(-40, 60)
        ax.set_title(f'{tag}: G {glo}-{ghi}')
        ax.set_xlabel('d - r_mask [px]')
        ax.set_ylabel('10^-3 sigma')
        if col == 0:
            ax.legend(fontsize=7)
pstr = '-'.join(str(p) for p in patches) if len(patches) <= 4 \
    else f'{len(patches)}patches'
fig.suptitle(f'injection test, tract {tract} patches {pstr}')
fig.tight_layout()
out = f'{cdir}/inject-compare-{"-".join(tags)}-{pstr}.png'
fig.savefig(out, dpi=110)
print('wrote', out)
