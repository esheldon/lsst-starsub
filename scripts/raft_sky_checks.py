"""
Sky checks on the full-image outputs of one raft: flatness away from
stars and continuity across the detector edges, the pipeline's sky
against ours.

Reads the lsst-starsub-visit image files of a 3 x 3 raft (detectors
in row-major order, same orientation, the LSSTCam science rafts) and
measures, on the star-subtracted images:

- the sky flatness: medians of 64 px boxes against the distance to the
  nearest bright star (G < GBRIGHT, anywhere on the raft), for the
  pipeline's delivered sky, its warp-state sky, and ours;
- the edge continuity: for each pair of adjacent detectors, the total
  sky model on both sides of the shared edge in strips along it, the
  model difference across the gap (the true sky is continuous there,
  up to the per-amplifier offsets), and the residual data on both
  sides.

usage: python raft_sky_checks.py IMGDIR VISIT OUT.png DET0 [DET0 the
       lower-left detector; the raft is DET0 .. DET0 + 8]
"""
import glob
import os
import sys

import numpy as np
import rustfits

GBRIGHT = 10.0
BOX = 64
# LSSTCam science rafts in detector order: detector = 9 x raft index +
# 3 x sensor row + sensor column
RAFTS = ['R01', 'R02', 'R03', 'R10', 'R11', 'R12', 'R13', 'R14', 'R20',
         'R21', 'R22', 'R23', 'R24', 'R30', 'R31', 'R32', 'R33', 'R34',
         'R41', 'R42', 'R43']


def raft_name(det):
    """The raft of a science detector, e.g. R23."""
    return RAFTS[int(det) // 9]


def detector_name(det):
    """The full name of a science detector, e.g. R23_S11."""
    s = int(det) % 9
    return f'{raft_name(det)}_S{s // 3}{s % 3}'


STRIP = 128          # px on each side of an edge
ALONG = 256          # bins along the edge
PITCH = 4225         # px between detector origins in a raft (42.25 mm)
DBINS = [0, 150, 300, 600, 1200, 2400, 1e9]


def read_detector(fname):
    with rustfits.FITS(fname) as fits:
        d = {name: fits[name].read() for name in (
            'delivered', 'sky', 'star_model', 'restored', 'initial_coarse',
            'initial_fine', 'skycorr', 'starmask', 'mask', 'var',
        )}
        d['stars'] = fits['gaia_stars'].read()
        d['meta'] = fits['meta'].read()
    dm_initial = d['initial_coarse'] + d['initial_fine']
    # the three sky-subtracted images with the stars in: the pipeline's
    # delivered image, that with skyCorr applied, and ours
    d['img_delivered'] = d['delivered']
    d['img_skycorr'] = d['delivered'] - d['skycorr']
    d['img_starsub'] = d['delivered'] + d['restored'] - d['sky']
    # and with our star model removed
    d['res_delivered'] = d['img_delivered'] - d['star_model']
    d['res_skycorr'] = d['img_skycorr'] - d['star_model']
    d['res_starsub'] = d['img_starsub'] - d['star_model']
    # the total sky each removed, from the same calibrated image
    d['sky_delivered'] = dm_initial
    d['sky_skycorr'] = dm_initial + d['skycorr']
    d['sky_starsub'] = dm_initial - d['restored'] + d['sky']
    d['usable_stars'] = ((d['mask'] & 1) == 0) & np.isfinite(d['var']) \
        & (d['var'] > 0)
    d['usable'] = d['usable_stars'] & (d['starmask'] == 0)
    return d


def box_medians(image, usable, box):
    ny, nx = image.shape
    my, mx = ny // box, nx // box
    im = np.where(usable, image, np.nan)[:my * box, :mx * box]
    im = im.reshape(my, box, mx, box).transpose(0, 2, 1, 3).reshape(my, mx, -1)
    ok = np.isfinite(im).sum(axis=2) > 0.5 * box * box
    med = np.nanmedian(im, axis=2)
    med[~ok] = np.nan
    yc = (np.arange(my) + 0.5) * box
    xc = (np.arange(mx) + 0.5) * box
    return med, xc, yc


def main():
    imgdir, out = sys.argv[1], sys.argv[3]
    visit = int(sys.argv[2])
    det0 = int(sys.argv[4])
    dets = {}
    for k in range(9):
        det = det0 + k
        files = glob.glob(os.path.join(imgdir, f'*-{visit}-{det:03d}.fits'))
        if not files:
            print(f'no image file for detector {det}')
            continue
        d = read_detector(files[0])
        d['row'], d['col'] = k // 3, k % 3
        dets[det] = d
    sigma = np.median([np.sqrt(np.nanmedian(d['var'])) for d in dets.values()])
    print(f'{len(dets)} detectors, sky sigma {sigma:.1f} nJy')

    # bright stars in the raft frame
    bx, by, bg = [], [], []
    for d in dets.values():
        s = d['stars']
        s = s[(s['G'] < GBRIGHT) & (s['on_image'] == 1)]
        bx.extend(s['x'] + d['col'] * PITCH)
        by.extend(s['y'] + d['row'] * PITCH)
        bg.extend(s['G'])
    bx, by = np.array(bx), np.array(by)
    print(f'{bx.size} stars brighter than G {GBRIGHT}: '
          + ' '.join(f'{g:.1f}' for g in sorted(bg)))

    # flatness: box medians vs distance to the nearest bright star
    states = ['res_delivered', 'res_skycorr', 'res_starsub']
    dist, vals = [], {s: [] for s in states}
    for d in dets.values():
        first = True
        for s in states:
            med, xc, yc = box_medians(d[s], d['usable'], BOX)
            if first:
                xx, yy = np.meshgrid(xc + d['col'] * PITCH,
                                     yc + d['row'] * PITCH)
                if bx.size:
                    dd = np.min(np.hypot(xx[..., None] - bx,
                                         yy[..., None] - by), axis=2)
                else:
                    dd = np.full(xx.shape, 1e9)
                dist.append(dd.ravel())
                first = False
            vals[s].append(med.ravel())
    dist = np.concatenate(dist)
    vals = {s: np.concatenate(v) for s, v in vals.items()}
    # each state referenced to its own far field: the states differ by
    # a pedestal (what each sky estimator does with the unmasked faint
    # sources), which a coadd's own zero point absorbs; the structure
    # toward the stars is what matters
    far = dist >= DBINS[-2]
    pedestal = {s: np.nanmedian(vals[s][far]) for s in states}
    print('\nfar-field pedestal (nJy, beyond '
          f'{DBINS[-2]:.0f} px of any G < {GBRIGHT} star): '
          + ', '.join(f'{s} {pedestal[s]:+.2f}' for s in states))
    print('box medians less the pedestal (10^-3 sky sigma) by distance '
          f'to the nearest G < {GBRIGHT} star:')
    print('  distance (px)   nbox ' + ' '.join(f'{s:>14s}' for s in states))
    flat = {s: [] for s in states}
    for lo, hi in zip(DBINS[:-1], DBINS[1:]):
        sel = (dist >= lo) & (dist < hi) & np.isfinite(vals[states[0]])
        row = []
        for s in states:
            v = np.nan
            if sel.any():
                v = (np.nanmedian(vals[s][sel]) - pedestal[s]) / sigma * 1e3
            flat[s].append(v)
            row.append(v)
        print(f'  {lo:5.0f}-{min(hi, 9999):5.0f} {sel.sum():7d} '
              + ' '.join(f'{v:+14.0f}' for v in row))

    # edge continuity
    pairs = []
    for det, d in dets.items():
        for other, e in dets.items():
            if e['row'] == d['row'] and e['col'] == d['col'] + 1:
                pairs.append((det, other, 'x'))
            if e['col'] == d['col'] and e['row'] == d['row'] + 1:
                pairs.append((det, other, 'y'))
    models = ['sky_delivered', 'sky_skycorr', 'sky_starsub']
    print(f'\nedge continuity over {len(pairs)} adjacent pairs, strips '
          f'{STRIP} px each side, {ALONG} px bins along the edge:')
    jumps = {m: [] for m in models + ['data']}
    resid = {s: [] for s in states}
    for a, b, axis in pairs:
        A, B = dets[a], dets[b]
        # the calibrated data themselves, sources and stars masked
        for d in (A, B):
            d['data'] = d['delivered'] + d['sky_delivered']
        for m in models + ['data'] + states:
            ia, ib = A[m], B[m]
            ua, ub = A['usable'], B['usable']
            if axis == 'x':
                sa, sb = ia[:, -STRIP:], ib[:, :STRIP]
                ma, mb = ua[:, -STRIP:], ub[:, :STRIP]
            else:
                sa, sb = ia[-STRIP:, :].T, ib[:STRIP, :].T
                ma, mb = ua[-STRIP:, :].T, ub[:STRIP, :].T
            n = sa.shape[0] // ALONG
            for k in range(n):
                sl = np.s_[k * ALONG:(k + 1) * ALONG]
                if m in models:
                    va, vb = sa[sl].mean(), sb[sl].mean()
                    jumps[m].append(va - vb)
                    continue
                wa = np.where(ma[sl], sa[sl], np.nan)
                wb = np.where(mb[sl], sb[sl], np.nan)
                va, vb = np.nanmedian(wa), np.nanmedian(wb)
                if m == 'data':
                    jumps[m].append(va - vb)
                else:
                    resid[m].append((va, vb))
    print('  jump across the gap (A - B, nJy) of the data (sources '
          'masked) and of each sky model: median, rms over bins')
    for m in ['data'] + models:
        j = np.array(jumps[m])
        j = j[np.isfinite(j)]
        print(f'    {m:14s} {np.median(j):+7.2f} {j.std():7.2f}')
    print('  residual data on the two sides (median in the strips, '
          '10^-3 sky sigma): side A, side B, and their difference rms')
    for s in states:
        r = np.array(resid[s]) / sigma * 1e3
        print(f'    {s:14s} {np.nanmedian(r[:, 0]):+7.0f} '
              f'{np.nanmedian(r[:, 1]):+7.0f}   diff rms '
              f'{np.nanstd(r[:, 0] - r[:, 1]):6.0f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mids = [0.5 * (lo + min(hi, 4800))
            for lo, hi in zip(DBINS[:-1], DBINS[1:])]
    for s in states:
        axes[0].plot(mids, flat[s], 'o-', label=s)
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(f'distance to the nearest G < {GBRIGHT:.0f} star (px)')
    axes[0].set_ylabel('box median (10^-3 sky sigma)')
    axes[0].legend(fontsize=8)
    axes[0].set_title('sky flatness, stars removed')
    for m in models:
        axes[1].hist(jumps[m], bins=30, histtype='step', label=m)
    axes[1].set_xlabel('sky model jump across the gap (nJy)')
    axes[1].legend(fontsize=8)
    axes[1].set_title('edge continuity')
    fig.suptitle(f'visit {visit}, raft {raft_name(det0)}')
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print('wrote', out)


if __name__ == '__main__':
    main()
