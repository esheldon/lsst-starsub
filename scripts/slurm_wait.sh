#!/bin/bash
# Wait for a batch of per-detector jobs and resubmit until every output
# exists.  Sourced by the visit rerun scripts.
#
#   wait_complete JOBDIR PATTERN OUTDIR VISIT
#
# JOBDIR holds PATTERN-VISIT-DDD.sl jobs (slurm-incsub markers beside
# them), OUTDIR the profiles-*-VISIT-DDD.fits outputs.  Waits for the
# queue to drain, resubmits the jobs without an output (preempted
# without a log, lost, or failed) up to MAX_ROUNDS times, and reports
# what is missing at the end.  milano jobs are preemptable and a
# preempted job can vanish without a log, so the sweep goes by the
# outputs, not the logs.
MAX_ROUNDS=${MAX_ROUNDS:-10}

drain() {  # $1 pattern
    sleep 60
    until [ $(squeue -u $USER -h -o "%j" | grep -c "^$1") -eq 0 ]; do
        sleep 60
    done
}

missing_jobs() {  # $1 jobdir $2 pattern $3 outdir $4 visit -> job files
    for job in $1/$2-$4-*.sl; do
        det=$(basename $job .sl)
        det=${det##*-}
        ls $3/profiles-*-$4-$det.fits > /dev/null 2>&1 || echo $job
    done
}

wait_complete() {  # $1 jobdir $2 pattern $3 outdir $4 visit
    local jobdir=$1 pattern=$2 outdir=$3 visit=$4
    drain $pattern-
    for round in $(seq 1 $MAX_ROUNDS); do
        local miss=$(missing_jobs $jobdir $pattern $outdir $visit)
        local n=$(echo "$miss" | grep -c .)
        [ "$n" = "0" ] && { echo "$pattern: complete"; return 0; }
        echo "$pattern round $round: $n missing, resubmitting"
        for job in $miss; do
            rm -f $job.submitted ${job%.sl}.submitted
        done
        (cd $jobdir && slurm-incsub --pattern $pattern- -n 1500 -p 30 *.sl \
            > incsub-round$round.log 2>&1)
        drain $pattern-
    done
    local miss=$(missing_jobs $jobdir $pattern $outdir $visit)
    echo "$pattern: still missing after $MAX_ROUNDS rounds:"
    echo "$miss"
    return 1
}
