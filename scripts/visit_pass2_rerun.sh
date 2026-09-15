#!/bin/bash
# Pass 2 of the visit scheme at the amplitudes of a pass-1 rerun
# (visit_pass1_rerun.sh), then the edge-star stacks against the
# existing pass-2 run at the predicted amplitudes (pass2c-prediction
# from visit_edge_chain.sh).
#
#   visit_pass2_rerun.sh VISIT BAND TAG      # TAG the pass-1 tag, e.g. p1d
set -u
V=$1
B=$2
T=$3
P2=${T/p1/p2}
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
python $S/visit_pass_jobs.py $V detectors-inner.txt pass$P2-consolidated jobs-$P2 $W \
    --band $B --amplitudes pass$T/amplitudes-$V-$B.fits --tag $P2
(cd jobs-$P2 && slurm-incsub --pattern $P2- -n 1500 -p 30 *.sl > incsub.log 2>&1)
wait_sweep $run/jobs-$P2 $P2-
echo "pass$P2 outputs: $(ls pass$P2-consolidated/profiles-*.fits | wc -l) of $(wc -l < detectors-inner.txt)"
python $S/edge_stack.py pass2c-prediction pass$P2-consolidated $V edge-stack-$P2-inner.png \
    detectors-inner.txt 2>&1 | grep -v Warning
