#!/bin/bash
# The per-band wing calibration end to end: per-visit template
# extracts on slurm (one job per detector, preemption sweeps until
# every extract exists), the pooled template per visit, then the
# canonical wing of the band as the median over visits.
#
# usage: band_wing.sh BAND VISITFILE [MAXVISITS]
#   VISITFILE: one visit per line; MAXVISITS picks a random subset
set -e
band=$1
visitfile=$2
maxv=${3:-0}
base=/sdf/home/e/esheldon/oh/starsub-visits
S=/sdf/home/e/esheldon/git/lsst-starsub/scripts
tdir=$base/templates
jobdir=$tdir/jobs-07275-$band
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

visits=$(grep -v '^#' $visitfile | awk 'NF')
if [ $maxv -gt 0 ]; then
  visits=$(echo "$visits" | shuf --random-source=<(yes) | head -n $maxv | sort -n)
fi
echo "$band: $(echo "$visits" | wc -l) visits"
echo "$visits" > $tdir/visits-07275-$band.txt

lsst-starsub-make-slurm-template --visits $visits --gaia-dir $base/gaia \
    --outdir $tdir --jobdir $jobdir --mem 8G --time 00:30:00 \
    --repo dp2_prep_future --collection LSSTCam/runs/DRP/DP2 --skip-existing
cd $jobdir
slurm-incsub --pattern tmpl- -n 1500 -p 60 *.sl > incsub.log 2>&1
for round in 1 2 3 4 5 6; do
  sleep 180
  until [ $(squeue -u $USER -h -o "%j" | grep -c "^tmpl-") -eq 0 ]; do sleep 120; done
  n=$($S/resubmit_preempted.sh $jobdir tmpl- 1500 | head -1 | awk '{print $1}')
  echo "sweep $round: $n resubmitted"
  [ $n -eq 0 ] && break
done

for v in $visits; do
  lsst-starsub-visit-template --visit $v --from-extracts --gaia-dir $base/gaia \
      --outdir $tdir --repo dp2_prep_future --collection LSSTCam/runs/DRP/DP2 \
      > $tdir/pool-$v.log 2>&1 || echo "pooling failed for $v"
done
python - <<EOF
import glob
from lsst_starsub.template import canonical_wing, write_canonical_wing
files = sorted(glob.glob('$tdir/template-*-$band.fits'))
r, T, curves = canonical_wing(files)
write_canonical_wing('$tdir/canonical-wing-07275-$band.fits', r, T, '$band', len(files))
import numpy as np
for rr in (50, 100, 300, 1000):
    k = np.argmin(abs(r - rr)); s = np.std(curves[:, k]) / np.median(curves[:, k])
    print(f'$band: {len(files)} visits; scatter at {rr} px {100 * s:.0f} percent')
print('wrote', '$tdir/canonical-wing-07275-$band.fits')
EOF
