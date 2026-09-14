"""
cli/make_slurm_template: slurm jobs for the pooled per-visit
template extraction, one job per (visit, detector), for S3DF

The per-visit Gaia files are made here, up front, so the jobs
never race to build one.  Each job runs lsst-starsub-visit-template
--extract-only on its detector; pool afterwards with

    lsst-starsub-visit-template --visit V --from-extracts \\
        --gaia-dir DIR --outdir OUTDIR

Measured on 2025071900593: 35-100 s and 2.5 GB per detector, so
the default request is one core, 4 GB and 15 minutes
"""
import os

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/usr/bin/time -v lsst-starsub-visit-template \
    --visit %(visit)d --detectors %(detector)d --extract-only \
    --gaia-dir %(gaia_dir)s --outdir %(outdir)s \
    --repo %(repo)s --collection %(collection)s
'''


def get_args():
    import argparse
    from ..site import VISIT_COLLECTION, VISIT_REPO

    parser = argparse.ArgumentParser()
    parser.add_argument('--visits', type=int, nargs='+', required=True)
    parser.add_argument(
        '--detectors', type=int, nargs='+',
        help='these detectors (default: every detector with a wcs and '
             'calibration in each visit summary)',
    )
    parser.add_argument('--gaia-dir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--jobdir', required=True)
    parser.add_argument('--repo', default=VISIT_REPO)
    parser.add_argument('--collection', default=VISIT_COLLECTION)
    parser.add_argument('--mem', default='4G')
    parser.add_argument('--time', default='00:15:00')
    parser.add_argument('--partition', default='milano')
    parser.add_argument('--account', default='rubin:default')
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='no job for a detector whose extract file exists',
    )
    return parser.parse_args()


def main():
    from ..visit.gaia import ensure_visit_gaia_file
    from ..visit.exposure import make_visit_butler
    from .visit_template import visit_detectors

    args = get_args()
    os.makedirs(args.jobdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    outdir = os.path.abspath(args.outdir)
    jobdir = os.path.abspath(args.jobdir)
    gaia_dir = os.path.abspath(args.gaia_dir)
    butler = make_visit_butler(args.repo, args.collection)

    submit = os.path.join(jobdir, 'submit.sh')
    njob = 0
    with open(submit, 'w') as sub:
        sub.write('#!/bin/bash\n')
        for visit in args.visits:
            ensure_visit_gaia_file(butler, visit, gaia_dir)
            dets = args.detectors or visit_detectors(butler, visit)
            edir = os.path.join(outdir, f'extracts-{visit}')
            for det in dets:
                if args.skip_existing and os.path.exists(os.path.join(
                        edir, f'extract-{visit}-{det:03d}.fits')):
                    continue
                name = f'tmpl-{visit}-{det:03d}'
                job = os.path.join(jobdir, name + '.sl')
                with open(job, 'w') as fobj:
                    fobj.write(SLURM_TEMPLATE % dict(
                        job_name=name,
                        logfile=os.path.join(jobdir, name + '.log'),
                        mem=args.mem, time=args.time,
                        partition=args.partition, account=args.account,
                        visit=visit, detector=det, gaia_dir=gaia_dir,
                        outdir=outdir, repo=args.repo,
                        collection=args.collection,
                    ))
                sub.write(f'sbatch {job}\n')
                njob += 1
    os.chmod(submit, 0o755)
    print(f'{njob} jobs, 1 core, {args.mem}, {args.time} each; '
          f'submit with {submit}')


if __name__ == '__main__':
    main()
