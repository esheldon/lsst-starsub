"""
the visit-level consolidation: the best-constrained detector wins, the
unconstrained keep the prediction, and the pairs table holds the stars
two detectors constrain
"""
import numpy as np

from lsst_starsub.visit.gather import consolidate, report


def make_table(rows):
    dtype = [
        ('ra', 'f8'), ('dec', 'f8'), ('G', 'f4'), ('A', 'f8'),
        ('A_err', 'f8'), ('free', 'i2'), ('on_image', 'i2'),
    ]
    return np.array(rows, dtype=dtype)


def test_consolidate():
    # star 1 on detector 10, also fit on 11 with a worse error;
    # star 2 only on 11; star 3 pinned everywhere; star 4 on 10 with
    # free set but a nan error (no cells) does not count
    d10 = make_table([
        (10.0, -5.0, 12.0, 1.10, 0.01, 1, 1),
        (10.2, -5.0, 16.0, 1.00, np.nan, 0, 0),
        (10.4, -5.0, 14.0, 0.90, np.nan, 1, 0),
    ])
    d11 = make_table([
        (10.0, -5.0, 12.0, 1.30, 0.05, 1, 0),
        (10.1, -5.0, 13.0, 0.80, 0.02, 1, 1),
        (10.2, -5.0, 16.0, 1.00, np.nan, 0, 1),
    ])
    amps, pairs = consolidate([d10, d11], [10, 11])

    assert amps.size == 4
    by_ra = {round(float(r['ra']), 3): r for r in amps}
    s1 = by_ra[10.0]
    assert s1['A'] == 1.10 and s1['detector'] == 10 and s1['ndet'] == 2
    assert s1['on_image'] == 1
    s2 = by_ra[10.1]
    assert s2['A'] == 0.80 and s2['detector'] == 11 and s2['ndet'] == 1
    s3 = by_ra[10.2]
    assert s3['A'] == 1.0 and s3['detector'] == -1 and s3['ndet'] == 0
    s4 = by_ra[10.4]
    assert s4['detector'] == -1 and s4['ndet'] == 0

    assert pairs.size == 1
    p = pairs[0]
    assert p['det1'] == 10 and p['det2'] == 11
    assert p['A1'] == 1.10 and p['A2'] == 1.30

    scale = report(amps, pairs, max_err=0.03)
    # the well-constrained stars: A 1.10 (err 0.01) and 0.80 (0.02)
    assert abs(scale - 0.95) < 1e-12
