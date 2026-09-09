#!/bin/bash
# resubmit the jobs of a slurm-incsub job directory that were
# preempted (the milano / rubin:default QOS is preemptable: the log
# ends with "CANCELLED") or never finished (no "Exit status" line
# and not in the queue): clear their .submitted markers and run
# slurm-incsub again with the pattern
#
# usage: resubmit_preempted.sh JOBDIR PATTERN [NJOBS]
set -e
jobdir=$1
pattern=$2
njobs=${3:-1000}
cd $jobdir
n=0
for sl in *.sl; do
  log=${sl%.sl}.log
  [ -f $sl.submitted ] || continue          # never submitted: incsub will
  if [ ! -f $log ]; then
    continue                                 # still queued, or no log yet
  fi
  if grep -q "CANCELLED" $log || ! grep -q "Exit status" $log; then
    # skip jobs still running
    name=${sl%.sl}
    if squeue -u $USER -h -o "%j" | grep -qx "$name"; then
      continue
    fi
    rm -f $sl.submitted
    n=$((n + 1))
  fi
done
echo "$n jobs to resubmit in $jobdir"
if [ $n -gt 0 ]; then
  slurm-incsub --pattern $pattern -n $njobs -p 60 *.sl > incsub-resubmit-$(date +%H%M%S).log 2>&1
  tail -n 1 incsub-resubmit-*.log | tail -n 1
fi
