#!/bin/bash
# Rerun pass 1 of the visit scheme on one visit with the installed code
# and a new tag (e.g. after a change to the core amplitudes), then the
# gather and the per-detector core-scale check.  Assumes
# visit_edge_chain.sh has run: detectors-inner.txt, the visit wing and
# per_detector_amplitudes.npy (field radii) exist in the run directory.
#
#   visit_pass1_rerun.sh VISIT BAND TAG
set -u
V=$1
B=$2
T=$3
base=~/oh/starsub-visits
run=$base/edge-$V
S=$(dirname $(realpath $0))
cd $run
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

wait_sweep() {  # $1 jobdir $2 pattern
    sleep 60
    until [ $(squeue -u $USER -h -o "%j" | grep -c "^$2") -eq 0 ]; do sleep 60; done
    for round in 1 2 3; do
        n=$($S/resubmit_preempted.sh $1 $2 1500 | head -1 | awk '{print $1}')
        echo "$2 sweep $round: $n resubmitted"
        [ "$n" = "0" ] && break
        sleep 120
        until [ $(squeue -u $USER -h -o "%j" | grep -c "^$2") -eq 0 ]; do sleep 60; done
    done
}

W=$run/wing-$V-$B.fits
python $S/visit_pass_jobs.py $V detectors-inner.txt pass$T jobs-$T $W \
    --band $B --core-rap 5 --tag $T
(cd jobs-$T && slurm-incsub --pattern $T- -n 1500 -p 30 *.sl > incsub.log 2>&1)
wait_sweep $run/jobs-$T $T-
echo "pass$T outputs: $(ls pass$T/profiles-*.fits | wc -l) of $(wc -l < detectors-inner.txt)"
lsst-starsub-visit-gather --visit $V --indir pass$T --detectors detectors-inner.txt > gather-$T.log 2>&1
cat gather-$T.log
python $S/core_scale_by_detector.py pass$T $V core-scale-$T.npy \
    --radius per_detector_amplitudes.npy 2>&1 | grep -v Warning
