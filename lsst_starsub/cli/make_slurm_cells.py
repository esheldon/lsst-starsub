"""
cli/make_slurm_cells: slurm jobs for lsst-starsub-cell-restore
over the patches of a tract, one patch per job, for S3DF

Each job runs the restoration with --nproc workers and
--no-images (the profile tables and census only, ~1 MB per
patch).  Measured on patch 53 of tract 2395 in i: 1:54 wall and
2.3 GB parent peak with 4 workers (each worker holds a butler
and its share of the polynomials, well under 1 GB), so the
default request of 4 cpus, 12 GB and 20 minutes carries margin
"""
import os

SLURM_TEMPLATE = r'''#!/bin/bash
#SBATCH --job-name=%(job_name)s
#SBATCH --output=%(logfile)s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(nproc)d
#SBATCH --mem=%(mem)s
#SBATCH --time=%(time)s
#SBATCH --partition=%(partition)s
#SBATCH --account=%(account)s

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/usr/bin/time -v lsst-starsub-cell-restore \
    --tract %(tract)d --patch %(patch)d --band %(band)s \
    --outdir %(outdir)s \
    --repo %(repo)s --collection %(collection)s \
    --gaia-file %(gaia_file)s \
    --nproc %(nproc)d --no-images
'''


def get_args():
    import argparse
    from ..visit import VISIT_COLLECTION, VISIT_REPO

    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patches', type=int, nargs='+', required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--jobdir', required=True)
    parser.add_argument('--repo', default=VISIT_REPO)
    parser.add_argument('--collection', default=VISIT_COLLECTION)
    parser.add_argument('--gaia-file', required=True)
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--mem', default='12G')
    parser.add_argument('--time', default='00:20:00')
    parser.add_argument('--partition', default='milano')
    parser.add_argument('--account', default='rubin:default')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.jobdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    outdir = os.path.abspath(args.outdir)
    jobdir = os.path.abspath(args.jobdir)

    submit = os.path.join(jobdir, 'submit.sh')
    with open(submit, 'w') as sub:
        sub.write('#!/bin/bash\n')
        for patch in args.patches:
            name = f'cell-{args.tract:05d}-{patch:02d}-{args.band}'
            job = os.path.join(jobdir, name + '.sl')
            with open(job, 'w') as fobj:
                fobj.write(SLURM_TEMPLATE % dict(
                    job_name=name,
                    logfile=os.path.join(jobdir, name + '.log'),
                    nproc=args.nproc, mem=args.mem, time=args.time,
                    partition=args.partition, account=args.account,
                    tract=args.tract, patch=patch, band=args.band,
                    outdir=outdir, repo=args.repo,
                    collection=args.collection,
                    gaia_file=os.path.abspath(args.gaia_file),
                ))
            sub.write(f'sbatch {job}\n')
    os.chmod(submit, 0o755)
    print(
        f'{len(args.patches)} jobs, {args.nproc} cpus, {args.mem}, '
        f'{args.time} each; submit with {submit}'
    )


if __name__ == '__main__':
    main()
