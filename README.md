# lsst-starsub

Bright-star wing and sky characterization on LSST visit images.

The coadd-based star subtraction in `lsst-mdet` cannot undo the
visit-level 32 px background pass that absorbed the star wings
before the coadds were built.  On the visit images every
background layer the pipeline subtracted is stored, so the raw
sky image is recoverable exactly and the star wings and the sky
can be characterized together.  See the `lsst_starsub.visit`
module docstring for what the butler stores and how it was
verified.

## Install

    pip install -e .

Requires `lsst_mdet` (from source, `~/git/lsst-mdet`) for the
star census, template and amplitude machinery, and the LSST
science pipelines for the butler loaders and the command line.
The characterization and profile code, and the tests, run
without the stack.

## Command line

- `lsst-starsub-visit --tract T --patch P --band B --outdir D`
  runs the characterization on the coadd inputs of the patch
  (`deep_coadd_input_summary_tract`), best shapelets IQ score
  first, `--nbest N` of them; or `--visit V --detector D` for
  one.  `--restore initial|fine|none` picks the stored layers
  added back (default both layers of the initial model, the raw
  sky); `--gaia-file` replaces the TAP query.

- `lsst-starsub-visit ... --list-inputs` prints the selected
  visit-detectors for driving workers; `--profiles-only` writes
  just the profile tables and census (140 KB instead of 370 MB).
- `lsst-starsub-remeasure FILES --outdir D [--no-wide]`
  re-measures the d − r_mask profiles from stored image states.
- `lsst-starsub-stack FILES --output PLOT [--ref global|local]
  [--clip 3]` stacks the per-star profiles over detectors per
  state and G slice, mean (sigma-clipped) and median, in
  10^-3 sky sigma, the layout of the coadd dual-state stacks.
- `lsst-starsub-make-slurm --inputs LIST ... --gaia-file F`
  writes S3DF slurm jobs (8 detectors per job, 8 GB, one core;
  the milano default per-core share is below the 2.7 GB peak
  with margin, so the memory request matters).  Use a per-tract
  Gaia file from `lsst-mdet-make-gaia` (refcat collection
  `refcats/DM-39298/gaia_dr3_20230707`) rather than the TAP
  service, which rate-limits.

Per detector it writes a FITS file with the image states
(`delivered`, `warp` = delivered with the visit-level sky
correction applied, i.e. what the coadd inputs carried, `flat`
= restored and sky-flattened by our model, `residual` = star-
and sky-subtracted), the sky and star models, the stored layers,
the census with amplitudes and the per-star profile table, plus
a summary png of the stacked profiles in sky-sigma units.
