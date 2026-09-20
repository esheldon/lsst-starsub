#!/bin/bash
# The visit scheme on every visit of a list, a few at a time.
#
#   run_tract_visits.sh VISITS_FILE BAND [PARALLEL] [BASE]
#
# PARALLEL visits run side by side (default 6: ~900 queued jobs at a
# time under the account's cap); each visit's log is BASE/VISIT/scheme.log.
set -u
list=$1
B=$2
par=${3:-6}
base=${4:-~/oh/starsub-visits/tract-07275}
S=$(dirname $(realpath $0))
mkdir -p $base
grep -v '^\s*$' $list | xargs -P $par -I{} $S/visit_scheme.sh {} $B $base
for V in $(grep -v '^\s*$' $list); do
    echo "$V: $(tail -n 1 $base/$V/scheme.log 2>/dev/null)"
done
