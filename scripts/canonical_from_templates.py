"""
The canonical wing of a band from its per-visit template files.

The median over visits of k_in T_v(r), after dropping the visits
whose pooled fit failed: a zero point k_in <= 0, an inner slope at
the scan bound (<= -4.95), or a chi2 above CHI2_FACTOR times the
median chi2 of the band (the z visits of 2025-09-02 on tract 7275
did all three).  Prints the excluded visits and the visit-to-visit
scatter of the survivors.

usage: python canonical_from_templates.py TEMPLATEDIR BAND OUT.fits
"""

import glob
import os
import sys
import numpy as np

from lsst_starsub.visit.template import canonical_wing, read_template_file
from lsst_starsub.wing import write_canonical_wing

CHI2_FACTOR = 5.0
SLOPE_BOUND = -4.95

tdir, band, out = sys.argv[1:4]
files = sorted(glob.glob(os.path.join(tdir, f'template-*-{band}.fits')))
params = [read_template_file(f)['params'] for f in files]
chi2 = np.array([float(p['chi2']) for p in params])
chi2_med = np.median(chi2)
keep = []
for f, p in zip(files, params):
    k_in, slope, c2 = float(p['k_in']), float(p['slope']), float(p['chi2'])
    why = []
    if not k_in > 0:
        why.append(f'k_in {k_in:.2e}')
    if slope <= SLOPE_BOUND:
        why.append(f'slope {slope:.2f} at the bound')
    if c2 > CHI2_FACTOR * chi2_med:
        why.append(
            f'chi2 {c2:.0f} > {CHI2_FACTOR:.0f} x median {chi2_med:.0f}'
        )
    if why:
        print(f'  excluded {os.path.basename(f)}: ' + ', '.join(why))
    else:
        keep.append(f)
r, T, curves = canonical_wing(keep)
write_canonical_wing(out, r, T, band, len(keep))
print(f'{band}: {len(keep)} of {len(files)} visits kept')
for rr in (50, 100, 300, 1000):
    k = np.argmin(abs(r - rr))
    s = np.std(curves[:, k]) / np.median(curves[:, k])
    print(f'{band}: scatter at {rr} px {100 * s:.0f} percent')
print('wrote', out)
