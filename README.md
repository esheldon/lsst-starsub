# lsst-starsub

Bright-star wings and backgrounds on LSST images: two projects on a
shared core.

- The **coadd project** (`lsst_starsub.coadd`) fixes the background
  and subtracts the stars on the cell coadds, using the wing profiles
  the visit project calibrated.  `lsst_mdet` calls it
  (`coadd.starsub.handle_stars_joint`, option `--starsub-method
  joint`) inside its metadetection run.
- The **visit project** (`lsst_starsub.visit`) characterizes the star
  wings and the sky on the visit images, where every background
  layer the pipeline subtracted is stored and the raw sky image is
  recoverable exactly; the aim is to do the fix at the visit level,
  which should work better.  See the `lsst_starsub.visit.exposure`
  module docstring for what the butler stores and how it was
  verified.  It also produces the per-band canonical wing files
  (`lsst-starsub-visit-template`, `scripts/band_wing.sh`).
- The **core** they share: the star census and masks
  (`lsst_starsub.census`), the Gaia extracts (`gaia`), the two star
  routes, stamp templates (`stamps`) and the joint star-and-sky fit
  (`joint`), the wing model with its renderers and file format
  (`wing`), synthetic stars for injection tests (`inject`), the DM
  mask bits (`maskbits`), geometry adapters (`geom`) and the site
  constants (`site`).  Neither subpackage imports the other; the
  command-line entry points (`lsst_starsub.cli`) may use both.

## Install

    pip install -e .

Nothing is imported from `lsst_mdet`, which calls this package for
both of its star routes.  The LSST science pipelines are needed for
the butler loaders and the command line; the characterization and
profile code, and the tests, run without the stack.

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
- `lsst-starsub-stack FILES --output PLOT [--ref global|local]
  [--clip 3]` stacks the per-star profiles over detectors per
  state and G slice, mean (sigma-clipped) and median, in
  10^-3 sky sigma, the layout of the coadd dual-state stacks.
- `lsst-starsub-make-gaia --tracts T ... --outdir D` writes the
  per-tract Gaia DR3 files, `gaia-dr3-{tract:05d}.fits`, from the
  DM reference catalog in the butler, to G < 21; both star routes
  and lsst_mdet's `--gaia-pattern` read them.  The defaults are
  NERSC's DP2 repo; at USDF pass `--repo dp2_prep --collections
  LSSTCam/runs/DRP/w_2026_32/DM-55677 --refcat-collection
  refcats/DM-39298/gaia_dr3_20230707`.

Per detector it writes a FITS file with the image states
(`delivered`, `warp` = delivered with the visit-level sky
correction applied, i.e. what the coadd inputs carried, `flat`
= restored and sky-flattened by our model, `residual` = star-
and sky-subtracted), the sky and star models, the stored layers,
the census with amplitudes and the per-star profile table, plus
a summary png of the stacked profiles in sky-sigma units.
