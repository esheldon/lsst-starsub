#!/bin/bash
# Rerun pass 1 of the visit scheme on one visit with the installed code
# and a new tag (e.g. after a change to the core amplitudes), then the
# gather and the per-detector core-scale check.  Assumes
# visit_edge_chain.sh has run: detectors-inner.txt, the visit wing and
# per_detector_amplitudes.npy (field radii) exist in the run directory.
# Waits until every detector's output exists (slurm_wait.sh), so the
# gather never runs on a partial pass.
#
#   visit_pass1_rerun.sh VISIT BAND TAG
set -u
V=$1
B=$2
T=$3
base=~/oh/starsub-visits
run=$base/edge-$V
S=$(dirname $(realpath $0))
source $S/slurm_wait.sh
cd $run
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

W=$run/wing-$V-$B.fits
python $S/visit_pass_jobs.py $V detectors-inner.txt pass$T jobs-$T $W \
    --band $B --core-rap 5 --tag $T --diagnostics all
(cd jobs-$T && slurm-incsub --pattern $T- -n 1500 -p 30 *.sl > incsub.log 2>&1)
wait_complete $run/jobs-$T $T pass$T $V || exit 1
echo "pass$T outputs: $(ls pass$T/profiles-*.fits | wc -l) of $(wc -l < detectors-inner.txt)"
lsst-starsub-visit-gather --visit $V --indir pass$T --detectors detectors-inner.txt > gather-$T.log 2>&1
cat gather-$T.log
python $S/core_scale_by_detector.py pass$T $V core-scale-$T.npy \
    --radius per_detector_amplitudes.npy 2>&1 | grep -v Warning
