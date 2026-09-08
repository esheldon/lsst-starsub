"""
cli/make_slurm: slurm jobs for lsst-starsub-visit over an input
list (visit detector [iq_score] per line, as --list-inputs
prints), for S3DF

Each job runs a chunk of the list sequentially in one process
per detector, single-threaded, with the memory and time set
from the measured cost (2.7 GB peak and ~75 s per detector on
the interactive node; the request carries a margin because the
default per-core share on milano is only about 3.7 GB, which is
where memory kills come from).  Chunking keeps the number of
simultaneous stack imports down
"""
import os

RUN_SCRIPT = r'''#!/usr/bin/bash
# runs the detectors listed in $1, one process each
if [ $# -lt 1 ]; then
    echo "run.sh listfile"
    exit 1
fi
listfile=$1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

status=0
while read -r visit detector rest; do
    [ -z "$visit" ] && continue
    log=%(outdir)s/log-${visit}-${detector}.log
    /usr/bin/time -v lsst-starsub-visit \
        --tract %(tract)d --patch %(patch)d --band %(band)s \
        --outdir %(outdir)s \
        --repo %(repo)s --collection %(collection)s \
        --visit ${visit} --detector ${detector} \
        %(extra)s > ${log} 2>&1
    rc=$?
    echo "done ${visit} ${detector} exit ${rc}"
    if [ ${rc} -ne 0 ]; then status=1; fi
done < ${listfile}
exit ${status}
'''

SLURM_TEMPLATE = r'''#!/bin/bash
#SBATCH --job-name=%(job_name)s
#SBATCH --output=%(logfile)s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=%(mem)s
#SBATCH --time=%(time)s
#SBATCH --partition=%(partition)s
#SBATCH --account=%(account)s

cd %(jobdir)s
./run.sh %(listfile)s
'''


def get_args():
    import argparse
    from ..visit import VISIT_COLLECTION, VISIT_REPO

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True,
                        help='input list from --list-inputs')
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--outdir', required=True,
                        help='where the detector outputs go')
    parser.add_argument('--jobdir', required=True,
                        help='where the job scripts and logs go')
    parser.add_argument('--repo', default=VISIT_REPO)
    parser.add_argument('--collection', default=VISIT_COLLECTION)
    parser.add_argument('--gaia-file', required=True)
    parser.add_argument('--per-job', type=int, default=8,
                        help='detectors per job')
    parser.add_argument('--mem', default='8G')
    parser.add_argument('--minutes-per-detector', type=float,
                        default=6.0,
                        help='walltime allowance per detector')
    parser.add_argument('--partition', default='milano')
    parser.add_argument('--account', default='rubin:default')
    parser.add_argument('--full-output', action='store_true',
                        help='write the image planes too '
                             '(default: --profiles-only)')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.jobdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    outdir = os.path.abspath(args.outdir)
    jobdir = os.path.abspath(args.jobdir)

    with open(args.inputs) as fobj:
        lines = [ln.strip() for ln in fobj if ln.strip()]

    extra = f'--gaia-file {os.path.abspath(args.gaia_file)}'
    if not args.full_output:
        extra += ' --profiles-only'

    run_path = os.path.join(jobdir, 'run.sh')
    with open(run_path, 'w') as fobj:
        fobj.write(RUN_SCRIPT % dict(
            outdir=outdir, tract=args.tract, patch=args.patch,
            band=args.band, repo=args.repo,
            collection=args.collection, extra=extra,
        ))
    os.chmod(run_path, 0o755)

    chunks = [
        lines[i:i + args.per_job]
        for i in range(0, len(lines), args.per_job)
    ]
    minutes = int(args.minutes_per_detector * args.per_job) + 10
    hours, mins = divmod(minutes, 60)
    tstr = f'{hours:02d}:{mins:02d}:00'

    submit = os.path.join(jobdir, 'submit.sh')
    with open(submit, 'w') as sub:
        sub.write('#!/bin/bash\n')
        for k, chunk in enumerate(chunks):
            listfile = os.path.join(jobdir, f'list-{k:03d}.txt')
            with open(listfile, 'w') as fobj:
                fobj.write('\n'.join(chunk) + '\n')
            job = os.path.join(jobdir, f'job-{k:03d}.sl')
            with open(job, 'w') as fobj:
                fobj.write(SLURM_TEMPLATE % dict(
                    job_name=f'ss-{args.tract}-{args.patch}-{k:03d}',
                    logfile=os.path.join(jobdir, f'job-{k:03d}.log'),
                    mem=args.mem, time=tstr,
                    partition=args.partition, account=args.account,
                    jobdir=jobdir, listfile=listfile,
                ))
            sub.write(f'sbatch {job}\n')
    os.chmod(submit, 0o755)
    print(
        f'{len(lines)} detectors in {len(chunks)} jobs of '
        f'{args.per_job}, {args.mem}, {tstr} each; submit with '
        f'{submit}'
    )


if __name__ == '__main__':
    main()
