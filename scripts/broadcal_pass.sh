#!/bin/bash
# the broad wing calibration, one pass: for every tract in
# broadcal/tracts-{band}.txt make the Gaia file if missing, write
# one slurm job per patch running the joint fit on the delivered
# coadd with the given wing, and submit them with slurm-incsub
#
# usage: broadcal_pass.sh PASSNAME WINGFILE [BAND]
set -e
pass=$1
wing=$2
band=${3:-i}
base=/sdf/home/e/esheldon/oh/starsub-visits
bc=$base/broadcal
outdir=$bc/$pass
jobdir=$bc/jobs-$pass
mkdir -p $outdir $jobdir $base/gaia
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

tracts=$(awk '{print $1}' $bc/tracts-$band.txt)
missing=""
for tr in $tracts; do
  tt=$(printf %05d $tr)
  [ -f $base/gaia/gaia-dr3-$tt.fits ] || missing="$missing $tr"
done
if [ -n "$missing" ]; then
  echo "making Gaia files for tracts:$missing"
  (cd $base/gaia && lsst-mdet-make-gaia --tracts $missing --outdir . \
     --repo dp2_prep --collections LSSTCam/runs/DRP/w_2026_32/DM-55677 \
     --refcat-collection refcats/DM-39298/gaia_dr3_20230707 \
     --skip-existing > make-gaia-broadcal-$pass.log 2>&1)
fi

for tr in $tracts; do
  tt=$(printf %05d $tr)
  [ -f $base/gaia/gaia-dr3-$tt.fits ] || { echo "no Gaia file for $tr"; continue; }
  for p in $(seq 0 99); do
    pp=$(printf %02d $p)
    name=bc-$pass-$tt-$pp
    cat > $jobdir/$name.sl <<EOS
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --output=$jobdir/$name.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=milano
#SBATCH --account=rubin:default
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/usr/bin/time -v lsst-starsub-cell-clean --tract $tr --patch $p --band $band \\
    --gaia-file $base/gaia/gaia-dr3-$tt.fits --canonical $wing \\
    --outdir $outdir --star-model joint --no-images \\
    --repo dp2_prep_future --collection LSSTCam/runs/DRP/DP2
EOS
  done
done
echo "$(ls $jobdir/*.sl | wc -l) jobs written"
cd $jobdir && slurm-incsub --pattern bc-$pass -n 1000 -p 60 *.sl > incsub.log 2>&1
tail -n 1 incsub.log
