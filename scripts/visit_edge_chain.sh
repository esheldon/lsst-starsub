#!/bin/bash
# The visit scheme on one visit, end to end, on S3DF (TODO step 9):
# the inner detectors, the visit's own wing, pass 1 with the core
# amplitudes, the gather, pass 2 at the consolidated and at the predicted
# amplitudes, and the edge-star stacks.
#
#   visit_edge_chain.sh VISIT BAND
#
# Needs the pooled template template-VISIT-BAND.fits and the per-visit
# Gaia extract from the wing calibration (band_wing.sh).  Runs in
# ~/oh/starsub-visits/edge-VISIT/.
set -u
V=$1
B=$2
base=~/oh/starsub-visits
run=$base/edge-$V
S=$(dirname $(realpath $0))
mkdir -p $run && cd $run
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
repo="--repo dp2_prep_future --collection LSSTCam/runs/DRP/DP2"

source $S/slurm_wait.sh

lsst-starsub-visit-detectors --visit $V $repo > detectors-inner.txt 2> detectors-inner.log
cat detectors-inner.log
lsst-starsub-visit-wing --template $base/templates/template-$V-$B.fits \
    --canonical $base/templates/canonical-wing-07275-$B.fits --outfile wing-$V-$B.fits
W=$run/wing-$V-$B.fits
gen="python $S/visit_pass_jobs.py $V detectors-inner.txt"

$gen pass1c jobs-p1c $W --band $B --core-rap 5 --tag p1c --diagnostics all
(cd jobs-p1c && slurm-incsub --pattern p1c- -n 1500 -p 30 *.sl > incsub.log 2>&1)
wait_complete $run/jobs-p1c p1c pass1c $V || exit 1
echo "pass1c outputs: $(ls pass1c/profiles-*.fits | wc -l) of $(wc -l < detectors-inner.txt)"
lsst-starsub-visit-gather --visit $V --indir pass1c --detectors detectors-inner.txt > gather-p1c.log 2>&1
cat gather-p1c.log
python - <<PY
import rustfits, numpy as np
f = 'pass1c/amplitudes-$V-$B.fits'
a = rustfits.read(f, ext='amplitudes'); m = rustfits.read(f, ext='meta')
a['A'] = 1.0; a['A_err'] = np.nan; a['detector'] = -1; a['ndet'] = 0
out = 'pass1c/amplitudes-prediction-$V-$B.fits'
rustfits.write(out, a, extname='amplitudes', mode='w+')
rustfits.write(out, m, extname='meta', mode='r+')
print('wrote', out)
PY
$gen pass2c-consolidated jobs-p2c $W --band $B --amplitudes pass1c/amplitudes-$V-$B.fits --tag p2c --diagnostics all
$gen pass2c-prediction jobs-p2cp $W --band $B --amplitudes pass1c/amplitudes-prediction-$V-$B.fits --tag p2cp --diagnostics all
(cd jobs-p2c && slurm-incsub --pattern p2c- -n 1500 -p 30 *.sl > incsub.log 2>&1)
(cd jobs-p2cp && slurm-incsub --pattern p2cp- -n 1500 -p 30 *.sl > incsub.log 2>&1)
wait_complete $run/jobs-p2c p2c pass2c-consolidated $V || exit 1
wait_complete $run/jobs-p2cp p2cp pass2c-prediction $V || exit 1
echo "outputs: consolidated $(ls pass2c-consolidated/profiles-*.fits | wc -l), prediction $(ls pass2c-prediction/profiles-*.fits | wc -l)"
python $S/edge_stack.py pass2c-prediction pass2c-consolidated $V edge-stack-c-inner.png detectors-inner.txt 2>&1 | grep -v Warning
