"""
Slurm jobs for the visit scheme's passes over one visit, for S3DF.

One job per detector running lsst-starsub-visit with the joint model
and --profiles-only: pass 1 with --edge-factor, pass 2 with the
consolidated --amplitudes as well.  Submit with slurm-incsub, then
gather pass 1 with lsst-starsub-visit-gather.

usage: python visit_pass_jobs.py VISIT DETECTORS OUTDIR JOBDIR CANONICAL
           [--amplitudes FILE] [--edge-factor 3] [--mem 6G] [--time 00:20:00]

DETECTORS is a file with one detector per line.  The tract/patch the
tool requires are only labels in the output names here.
"""
import argparse
import os

SLURM_TEMPLATE = r'''#!/bin/bash
#SBATCH --job-name=%(job_name)s
#SBATCH --output=%(logfile)s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=%(mem)s
#SBATCH --time=%(time)s
#SBATCH --partition=milano
#SBATCH --account=rubin:default

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/usr/bin/time -v lsst-starsub-visit \
    --tract 7275 --patch 55 --band %(band)s \
    --repo dp2_prep_future --collection LSSTCam/runs/DRP/DP2 \
    --visit %(visit)d --detector %(detector)d \
    --gaia-file %(gaia_file)s \
    --star-model joint --canonical %(canonical)s \
    --edge-factor %(edge_factor)g%(amplitudes)s \
    %(profiles_only)s%(no_profiles)s%(diagnostics)s--outdir %(outdir)s
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('visit', type=int)
    p.add_argument('detectors')
    p.add_argument('outdir')
    p.add_argument('jobdir')
    p.add_argument('canonical')
    p.add_argument('--band', default='i')
    p.add_argument('--gaia-dir',
                   default='/sdf/home/e/esheldon/oh/starsub-visits/gaia')
    p.add_argument('--amplitudes', default=None)
    p.add_argument('--tag', default=None,
                   help='job name prefix; default p1, or p2 with --amplitudes')
    p.add_argument('--edge-factor', type=float, default=3.0)
    p.add_argument('--core-rap', type=float, default=None,
                   help='pass --core-rap; --canonical must be the visit wing')
    p.add_argument('--mem', default='6G')
    p.add_argument('--time', default='00:20:00')
    p.add_argument('--no-profiles', action='store_true',
                   help='(a no-op since the CLI writes the product alone '
                        'by default; kept for the running production '
                        'chain)')
    p.add_argument('--diagnostics', default=None,
                   help='pass --diagnostics: maps, profiles or all')
    p.add_argument('--images', action='store_true',
                   help='write the full image file (delivered, restored, '
                        'sky, star model) instead of --profiles-only')
    args = p.parse_args()

    outdir = os.path.abspath(args.outdir)
    jobdir = os.path.abspath(args.jobdir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(jobdir, exist_ok=True)
    gaia_file = os.path.join(args.gaia_dir,
                             f'gaia-dr3-visit-{args.visit}.fits')
    amplitudes = ''
    tag = 'p1'
    if args.amplitudes is not None:
        afile = os.path.abspath(args.amplitudes)
        amplitudes = f' \\\n    --amplitudes {afile}'
        tag = 'p2'
    if args.core_rap is not None:
        amplitudes += f' \\\n    --core-rap {args.core_rap:g}'
    if args.tag is not None:
        tag = args.tag

    dets = [int(line) for line in open(args.detectors) if line.strip()]
    for det in dets:
        name = f'{tag}-{args.visit}-{det:03d}'
        job = os.path.join(jobdir, name + '.sl')
        with open(job, 'w') as fobj:
            fobj.write(SLURM_TEMPLATE % dict(
                job_name=name, logfile=os.path.join(jobdir, name + '.log'),
                mem=args.mem, time=args.time, band=args.band,
                visit=args.visit, detector=det, gaia_file=gaia_file,
                canonical=os.path.abspath(args.canonical),
                edge_factor=args.edge_factor, amplitudes=amplitudes,
                profiles_only='' if args.images else '--profiles-only ',
                no_profiles='--no-profiles ' if args.no_profiles else '',
                diagnostics=(f'--diagnostics {args.diagnostics} '
                             if args.diagnostics else ''),
                outdir=outdir,
            ))
    print(f'{len(dets)} jobs in {jobdir} ({tag}); submit with '
          f'slurm-incsub --pattern {tag}- *.sl')


if __name__ == '__main__':
    main()
