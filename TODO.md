# Plan: star wings and backgrounds, visit level to coadd

Written 2026-09-07 after the 191-detector trough test on tract 2395
patch 53 (i band, run `LSSTCam/runs/DRP/w_2026_32/DM-55677` in repo
`dp2_prep`).  Results are in `~/oh/starsub-visits/02395-53-all/`.

## The image states, from raw to what you get

| State | What was subtracted | Where it lives |
|---|---|---|
| raw sky image | nothing | not stored; preliminary image plus its stored background |
| preliminary visit image | per-detector 6x6 Chebyshev polynomial plus order-0 constant(s) | `preliminary_visit_image` (ADU), `preliminary_visit_image_background` |
| visit image | same, calibrated to nJy | `visit_image` (the `delivered` state in the plots) |
| sky-corrected visit | polynomial undone, focal-plane model (4096/8192 px bins) subtracted instead | not stored; visit image minus `skyCorr` (the `warp` state) |
| deep coadd, `None` | coadd of the preliminary images, polynomials still in | `deep_coadd_predetection`, or `deep_coadd` with `object` restored |
| deep coadd, `object` | the above plus a 128 px Akima spline fit with objects masked (`detectCoaddPeaks`) | `deep_coadd` as loaded, `deep_coadd_background` |
| pretty coadd | built from sky-corrected visits instead | separate product, not our input |

Facts established:

- The deep-coadd `makeDirectWarp` run config has
  `doApplyNewBackground=False` with `calexp_list=preliminary_visit_image`
  (read from the stored config text via `butler.getURI`; the formatter
  refuses it).  Only the pretty warps apply `skyCorr`.
- The coadd dual-state trough (-0.1 to -0.2 nJy at 100-450 px beyond the
  mask) is the visit polynomial overshooting around bright stars: the
  `delivered` visit state stacked over 191 inputs shows -0.13 nJy at
  205 px for G 14-15.2 and -0.12 nJy at 455 px for G 6-13; the `warp`
  state shows none.
- The polynomial is stored exactly per visit-detector, so it is
  recoverable; the `object` background is restorable already.
- Per-detector templates (20-25 stamps) are not usable: halo slopes
  scatter from -2.6 to -4.8 with no seeing trend, flattest on detectors
  hosting a G < 10 star.

## Step 1 status (2026-09-08)

Done on the cell coadds of tract 2395 (i, 100 patches, 7374 census
stars) with `lsst-starsub-cell-restore` and `lsst-starsub-make-slurm-cells`:

- The cell is the unit: every input covers its whole cell, so the
  background coadd is the per-cell weighted mean of the inputs'
  stored polynomials with the cell's own input weights, no per-pixel
  membership.  Geometry and calibration from `visit_summary`.
- Restoring the full polynomials is useless: their per-visit sky
  levels make the restored sky a patchwork stepping ~26 nJy at cell
  boundaries (N ~ 44; ~100 nJy at N ~ 10).  A second-order surface is
  removed from each polynomial first (`SMOOTH_ORDER`); what remains is
  the star response plus the visits' higher-order sky terms, which
  average down as 1/sqrt(N) (per-cell offset rms 0.26 nJy at N = 44,
  within-cell 0.04 nJy).  Higher fit orders cannot help: the 6x6
  Chebyshev scale (~700 px) is the star-bump scale.
- Result: the `None` trough (-15 to -19 x 10^-3 coadd sigma at 300-450
  px for G 6-13; -5 for G 14-15.2) becomes a positive, outward-falling
  wing (+21/+12 and +9/+3) with errors at the `None` level.  The
  `object` state reproduces the dual-state plot A.
- Open: at survey depth (N ~ 10) the statistical separation degrades
  (offsets ~0.5-1 nJy).  The principled route for step 4 is to forward
  model the trough: it is the projection of the star-wing model onto
  each input's Chebyshev fit (DM binning and masking), a linear
  functional of the star model with no sky terms.  Validate on a
  typical-depth tract first (2562: median 7 visits/patch, 8537 good
  cells; 2397 at 19 as a middle point).

## Steps

1. **Restore the polynomial at the coadd level.**  For a patch, take the
   coadd's input list and weights (`deep_coadd_input_summary_tract`, plus
   the per-cell input map for cell coadds), evaluate each input's stored
   polynomial at every coadd pixel through the detector WCS, and form the
   weighted mean.  Add it to the `None` state.  Validate with the
   d - r_mask dual-state stack: the trough should vanish.  This decides
   whether the clean-background coadd of option 1 exists without
   recoadding.

2. **Decide the sky model for the restored coadd.**  With the polynomial
   back, the coadd carries the true sky plus wings.  Fit a sky that stays
   out of the wings on the deep image.  The visit-level version (256 px
   pass with stars excluded to their template extents, then a 64 px
   refit) is a first draft; the refit still takes wing at G 13-14
   (`flat` state -2 to -3 x 10^-3 sigma beyond 300 px) and needs rework.

3. **Per-visit wing characterization, pooled over the focal plane.**  Run
   the visit tool on all detectors of each input visit, but build one
   template and aureole per visit from all its stars, excluding stamps
   inside bright-star halos.

4. **Carry the visit wing models into the coadd.**  Per cell, the star
   model is the input-weighted mean of the per-visit models at that
   position, with the step-1 weights.  Keeps the per-visit seeing
   dependence of the wings without special coadds (subtraction commutes
   with the weighted mean, except at clipped pixels and masked cores).

5. **Subtract and refit at coadd level.**  Subtract the step-4 model from
   the restored, sky-fitted coadd, then the existing lsst-mdet
   anchor-ring amplitude refit as the correction with the deep S/N.
   Masking stays as it is.

6. **Validate end to end.**  The dual-state stack on the final images in
   all bands, plus the amplified injection test in lsst-mdet.

Steps 1-2 need the coadd side only and settle the background question.
Steps 3-5 are the star side.  Option 2 (pre-subtract on visits and
recoadd) is the fallback if step 1 does not remove the trough, which
would mean coadd assembly also acts on the wings.

## Smaller items

- Profiles: reference each state locally or exclude neighbors to their
  wide radius (done in `lsst-starsub-remeasure` and the visit tool);
  the global ambient reference alone leaves the wing carpet of the
  bright stars in the faint bins.
- Some detectors carry 3-4 initial background layers (extra order-0
  iterations); the loader sums layers 1..n-1 into the fine model.
- `lsst_mdet.starsub.field_segmentation` now retries without deblending
  on a sep sub-object overflow (very bright stars on visit images); the
  installed lsst_mdet is a plain (non-editable) install, so edits in the
  checkout need `pip install .`.
- Slurm: 8 detectors per job, 8 GB, one core on `milano`; the default
  per-core share is below the 2.7 GB peak with margin.  Use the
  per-tract Gaia file from `lsst-mdet-make-gaia` (refcat collection
  `refcats/DM-39298/gaia_dr3_20230707`), not TAP.
