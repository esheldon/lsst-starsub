"""
The visit-level consolidation of the per-detector joint fits.

Pass 1 fits every detector on its own (lsst-starsub-visit with the
joint model and --edge-factor), so a star across a detector edge is
fit on each detector that holds part of its inner wing.  This gather
reads those per-detector results and gives each star one amplitude
for the visit: the one from the detector that constrains it best, by
the amplitude error.  Stars no detector constrains keep the prediction.
The stars two or more detectors constrain give a check of the wing
model across detectors, reported here.  Pass 2 (lsst-starsub-visit
--amplitudes) pins every star to its consolidated amplitude and refits
the sky alone.
"""
import glob
import os

import numpy as np


def profiles_files(indir, visit):
    """
    Find the per-detector profile files of a visit.

    Parameters
    ----------
    indir: str
        The directory lsst-starsub-visit --profiles-only wrote to
    visit: int

    Returns
    -------
    files: list of str
        Sorted, the lsst-starsub-visit naming
        profiles-{tract}-{patch}-{band}-{visit}-{detector}.fits
    """
    pattern = os.path.join(indir, f'profiles-*-{int(visit)}-*.fits')
    return sorted(glob.glob(pattern))


def read_detector_stars(fname):
    """
    Read one detector's census with fitted amplitudes and its meta.

    Parameters
    ----------
    fname: str
        A profiles file with the gaia_stars extension of the joint
        model (A, A_err, free)

    Returns
    -------
    stars: structured array
    meta: structured array
        One row
    """
    import rustfits

    with rustfits.FITS(fname) as fits:
        stars = fits['gaia_stars'].read()
        meta = fits['meta'].read()
    for name in ('A_err', 'free'):
        if name not in stars.dtype.names:
            raise ValueError(
                f'{fname}: no {name} column; the profiles file must come '
                'from the joint model'
            )
    return stars, meta


def consolidate(tables, detectors):
    """
    Give each star one amplitude from the detector that constrains it best.

    Parameters
    ----------
    tables: list of structured array
        The per-detector census tables (ra, dec, G, A, A_err, free,
        on_image)
    detectors: list of int
        The detector of each table

    Returns
    -------
    amplitudes: structured array
        One row per star seen on any detector: ra, dec, G, A, A_err,
        D, D_err, disk_detector (the ghost disk from the detector that
        constrains it best; D 1, the prediction, where none does),
        detector (the one chosen, -1 when no detector constrained
        the star and A is the prediction), ndet (the number of
        detectors that constrained it), on_image (of the chosen
        detector)
    pairs: structured array
        One row per star constrained on two or more detectors, the
        two best: ra, dec, G, A1, A1_err, det1, A2, A2_err, det2
    """
    from .exposure import star_key

    rows = {}
    for table, det in zip(tables, detectors):
        keys = star_key(table['ra'], table['dec'])
        for key, st in zip(keys, table):
            constrained = bool(st['free']) and np.isfinite(st['A_err'])
            entry = rows.setdefault(key, dict(
                ra=float(st['ra']), dec=float(st['dec']), G=float(st['G']),
                fits=[], disks=[],
            ))
            if constrained:
                entry['fits'].append((
                    float(st['A_err']), float(st['A']), int(det),
                    int(st['on_image']),
                ))
            # the ghost disk (tables written before the disks have no
            # D column: the prediction, unconstrained)
            if 'D' in table.dtype.names and np.isfinite(st['D_err']):
                entry['disks'].append((float(st['D_err']), float(st['D']),
                                       int(det)))

    amp_dtype = [
        ('ra', 'f8'), ('dec', 'f8'), ('G', 'f4'), ('A', 'f8'),
        ('A_err', 'f8'), ('detector', 'i4'), ('ndet', 'i4'),
        ('on_image', 'i2'), ('D', 'f8'), ('D_err', 'f8'),
        ('disk_detector', 'i4'),
    ]
    pair_dtype = [
        ('ra', 'f8'), ('dec', 'f8'), ('G', 'f4'),
        ('A1', 'f8'), ('A1_err', 'f8'), ('det1', 'i4'),
        ('A2', 'f8'), ('A2_err', 'f8'), ('det2', 'i4'),
    ]
    amps = []
    pairs = []
    for key in sorted(rows):
        e = rows[key]
        fits = sorted(e['fits'])
        disks = sorted(e['disks'])
        dsk = disks[0] if disks else (np.nan, 1.0, -1)
        if fits:
            err, a, det, on = fits[0]
            amps.append((e['ra'], e['dec'], e['G'], a, err, det,
                         len(fits), on, dsk[1], dsk[0], dsk[2]))
        else:
            amps.append((e['ra'], e['dec'], e['G'], 1.0, np.nan, -1, 0, 0,
                         dsk[1], dsk[0], dsk[2]))
        if len(fits) >= 2:
            (e1, a1, d1, _), (e2, a2, d2, _) = fits[:2]
            pairs.append((e['ra'], e['dec'], e['G'], a1, e1, d1, a2, e2, d2))

    amplitudes = np.array(amps, dtype=amp_dtype)
    pairs = np.array(pairs, dtype=pair_dtype)
    return amplitudes, pairs


def report(amplitudes, pairs, max_err=0.05):
    """
    Print the consolidation summary and the cross-detector check.

    Parameters
    ----------
    amplitudes, pairs: structured arrays
        From consolidate
    max_err: float, optional
        Stars with an amplitude error below this are the
        well-constrained set for the visit scale

    Returns
    -------
    scale: float
        The visit's amplitude scale, the median amplitude of the
        well-constrained stars (nan when there are none)
    """
    n = amplitudes.size
    ncon = int((amplitudes['ndet'] > 0).sum())
    nmulti = int((amplitudes['ndet'] > 1).sum())
    print(f'{n} stars; {ncon} constrained on at least one detector, '
          f'{nmulti} on two or more, {n - ncon} keep the prediction')

    well = (amplitudes['ndet'] > 0) & (amplitudes['A_err'] < max_err)
    scale = np.nan
    if well.any():
        scale = float(np.median(amplitudes['A'][well]))
        spread = float(np.percentile(amplitudes['A'][well], 84)
                       - np.percentile(amplitudes['A'][well], 16)) / 2
        print(f'visit scale: median A {scale:.3f} over {int(well.sum())} '
              f'stars with A_err < {max_err:g} (half the 16-84 range '
              f'{spread:.3f})')

    if pairs.size == 0:
        print('no star constrained on two detectors')
        return scale

    diff = pairs['A1'] - pairs['A2']
    err = np.sqrt(pairs['A1_err']**2 + pairs['A2_err']**2)
    chi = diff / err
    print(f'cross-detector check on {pairs.size} stars: A1 - A2 median '
          f'{np.median(diff):+.4f}, rms {diff.std():.4f}; chi median '
          f'{np.median(chi):+.2f}, rms {chi.std():.2f} (1 if the errors '
          f'are right and the wing is the same on both)')
    for lo, hi in ((0, 12), (12, 14), (14, 15.5), (15.5, 17)):
        w = (pairs['G'] >= lo) & (pairs['G'] < hi)
        if w.sum() < 2:
            continue
        print(f'    G {lo:4.1f}-{hi:<4.1f}: {int(w.sum()):4d} stars, '
              f'A1 - A2 median {np.median(diff[w]):+.4f} rms '
              f'{diff[w].std():.4f}, chi rms {chi[w].std():.2f}, '
              f'median err {np.median(err[w]):.4f}')
    return scale


def write_amplitudes(fname, amplitudes, pairs, visit, band, scale):
    """
    Write the consolidated amplitudes for --amplitudes of pass 2.

    Parameters
    ----------
    fname: str
    amplitudes, pairs: structured arrays
        From consolidate
    visit: int
    band: str
    scale: float
        From report
    """
    import rustfits

    meta = np.zeros(1, dtype=[
        ('visit', 'i8'), ('band', 'U1'), ('scale', 'f8'), ('nstar', 'i4'),
        ('npair', 'i4'),
    ])
    meta[0] = (int(visit), band, scale, amplitudes.size, pairs.size)
    print('writing:', fname)
    with rustfits.FITS(fname, 'w+') as fits:
        fits.write_table(amplitudes, extname='amplitudes')
        fits.write_table(meta, extname='meta')
        if pairs.size:
            fits.write_table(pairs, extname='pairs')


def read_amplitudes(fname):
    """
    Read a consolidated amplitudes file.

    Parameters
    ----------
    fname: str

    Returns
    -------
    amplitudes: structured array
        The amplitudes extension (ra, dec, G, A, ...)
    """
    import rustfits

    return rustfits.read(fname, ext='amplitudes')
