#!/bin/bash
# Pass 2 of the visit scheme at the amplitudes of a pass-1 rerun
# (visit_pass1_rerun.sh), then the edge-star stacks against the
# existing pass-2 run at the predicted amplitudes (pass2c-prediction
# from visit_edge_chain.sh).  Refuses to start on a partial pass 1, and
# waits until every detector's output exists (slurm_wait.sh).
#
#   visit_pass2_rerun.sh VISIT BAND TAG [P2TAG]
#       TAG the pass-1 tag, e.g. p1d; P2TAG the pass-2 tag, default the
#       pass-1 tag with p1 -> p2 (give one to redo pass 2 alone, e.g.
#       for a change in the outputs)
set -u
V=$1
B=$2
T=$3
P2=${4:-${T/p1/p2}}
base=~/oh/starsub-visits
run=$base/edge-$V
S=$(dirname $(realpath $0))
source $S/slurm_wait.sh
cd $run
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ndet=$(wc -l < detectors-inner.txt)
n1=$(ls pass$T/profiles-*.fits 2> /dev/null | wc -l)
if [ "$n1" -lt "$ndet" ]; then
    echo "pass $T has $n1 of $ndet outputs; complete it first"
    exit 1
fi
W=$run/wing-$V-$B.fits
python $S/visit_pass_jobs.py $V detectors-inner.txt pass$P2-consolidated jobs-$P2 $W \
    --band $B --amplitudes pass$T/amplitudes-$V-$B.fits --tag $P2
(cd jobs-$P2 && slurm-incsub --pattern $P2- -n 1500 -p 30 *.sl > incsub.log 2>&1)
wait_complete $run/jobs-$P2 $P2 pass$P2-consolidated $V || exit 1
echo "pass$P2 outputs: $(ls pass$P2-consolidated/profiles-*.fits | wc -l) of $ndet"
python $S/edge_stack.py pass2c-prediction pass$P2-consolidated $V edge-stack-$P2-inner.png \
    detectors-inner.txt 2>&1 | grep -v Warning
