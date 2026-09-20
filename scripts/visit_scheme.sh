#!/bin/bash
# The visit scheme on one visit, production form: the inner detectors,
# the visit's own wing from its pooled template, pass 1 with the core
# amplitudes, the gather, pass 2 at the consolidated amplitudes.  No
# diagnostic runs.  Every batch waits until each detector's output
# exists (slurm_wait.sh).  Skips the steps whose outputs exist, so a
# rerun completes an interrupted visit.
#
#   visit_scheme.sh VISIT BAND [BASE]
#
# Runs in BASE/VISIT (default ~/oh/starsub-visits/tract-07275/VISIT):
# detectors-inner.txt, wing-VISIT-BAND.fits, pass1/, pass2/, the job
# dirs and scheme.log.  Needs templates/template-VISIT-BAND.fits, the
# band's canonical wing and gaia/gaia-dr3-visit-VISIT.fits under
# ~/oh/starsub-visits.
set -u
V=$1
B=$2
base=${3:-~/oh/starsub-visits/tract-07275}
data=~/oh/starsub-visits
run=$base/$V
S=$(dirname $(realpath $0))
source $S/slurm_wait.sh
mkdir -p $run && cd $run
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
repo="--repo dp2_prep_future --collection LSSTCam/runs/DRP/DP2"
log=$run/scheme.log
echo "== visit $V band $B $(date)" >> $log

if [ ! -s detectors-inner.txt ]; then
    lsst-starsub-visit-detectors --visit $V $repo > detectors-inner.txt 2> detectors-inner.log
fi
ndet=$(wc -l < detectors-inner.txt)
W=$run/wing-$V-$B.fits
if [ ! -f $W ]; then
    lsst-starsub-visit-wing --template $data/templates/template-$V-$B.fits \
        --canonical $data/templates/canonical-wing-07275-$B.fits --outfile $W >> $log 2>&1
fi
gen="python $S/visit_pass_jobs.py $V detectors-inner.txt"

# pass 1
if [ $(ls pass1/profiles-*.fits 2> /dev/null | wc -l) -lt $ndet ]; then
    $gen pass1 jobs-p1 $W --band $B --core-rap 5 --tag p1 >> $log
    (cd jobs-p1 && slurm-incsub --pattern p1-$V- -n 1500 -p 30 *.sl > incsub.log 2>&1)
    wait_complete $run/jobs-p1 p1 pass1 $V >> $log || { echo "pass 1 incomplete" >> $log; exit 1; }
fi
# the gather
A=pass1/amplitudes-$V-$B.fits
if [ ! -f $A ]; then
    lsst-starsub-visit-gather --visit $V --indir pass1 --detectors detectors-inner.txt > gather.log 2>&1 || exit 1
fi
# pass 2
if [ $(ls pass2/profiles-*.fits 2> /dev/null | wc -l) -lt $ndet ]; then
    $gen pass2 jobs-p2 $W --band $B --amplitudes $A --tag p2 >> $log
    (cd jobs-p2 && slurm-incsub --pattern p2-$V- -n 1500 -p 30 *.sl > incsub.log 2>&1)
    wait_complete $run/jobs-p2 p2 pass2 $V >> $log || { echo "pass 2 incomplete" >> $log; exit 1; }
fi
echo "done: $ndet detectors, pass1 $(ls pass1/profiles-*.fits | wc -l), pass2 $(ls pass2/profiles-*.fits | wc -l) $(date)" >> $log
