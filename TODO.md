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
- Tract 2562, the typical-depth check (i; the 31 patches with good
  cells, cells with >= 3 inputs via `--good-cells` and `--min-inputs`;
  2460 profile rows, 70 stars at G < 13): same behaviour at low
  statistics.  The bright-star `None` trough at 305 px, -26 +- 6,
  becomes -6 +- 7 restored; errors inflated ~40 percent, not the
  factor 3 seen before the order-2 removal.  Per cell the offsets are
  ~1-2 nJy at N ~ 2 (0.1-0.2 coadd sigma), so it is the per-star
  restored wing, not the stack, that suffers at low N.
- Naming: `deep_coadd` is the dataset type, not "deep field".  Every
  `deep_coadd` in the run was made the same way (warps from
  `preliminary_visit_image`, no `skyCorr`), so the trough and this
  restoration apply to every patch of DP2 as processed; only the
  input count per cell varies.  Future reprocessings may differ;
  build for the data as they are.

## Step 2 status (2026-09-08)

The operator is reproduced (`lsst_starsub/forward.py`,
`lsst-starsub-forward-check`; outputs in `~/oh/starsub-visits/forward/`).
What the stored layer 0 actually is, from the `calibrateImage` source
and its stored config, log and metadata:

- Not the first-pass fit.  `_remeasure_star_background` adds the
  first fit back and refits the raw sky image (`star_background`:
  128 px bins, MEANCLIP, weighted 6x6 Chebyshev; BAD, EDGE, DETECTED,
  DETECTED_NEGATIVE, NO_DATA ignored; SAT/SUSPECT/SPIKE are ignored
  only by the order-0 pedestal fits that follow, layers 1..).
- The mask it saw is not stored (the preliminary image carries the
  final 5-sigma detection instead).  It is rebuilt from the rules:
  detection on the raw sky image at the threshold stored in
  `calibrateImage_metadata` (`adaptive_threshold_value`, 0.2 x the
  median sky x 1.07 in pixel-sigma units: 306 and 663 on the two
  test detectors), footprints grown by 70 psf sigma (110-180 px),
  ORed with the first-pass 50-sigma detections dilated by 10 px.
  Rebuilt detected fractions match the log to 0.1 percent (0.336 vs
  0.335, 0.4485 vs 0.449); the refit matches the stored surface to
  0.03 nJy rms on 2025071900593-090 and 0.13 nJy on 2025121600098-004
  (structure rms 0.6 and 2.0 nJy).
- The fit is not linear in the image: the bin weights come from
  the scatter of the pixel values in each bin, so the star image
  must not be fit alone (15 times too small).  The response is
  fit(raw) - fit(raw - star), which agrees with fit(raw + star) -
  fit(raw) to 0.3 percent.
- Bright stars are masked to ~200 px, so the polynomial only ever
  sees the wing beyond that, and the bump is set by the first ~200
  px outside the mask where the wing is steepest.  Around the G 7.6
  star on detector 4 the stored bump is +6.2 nJy at the star,
  falling to zero at 950 px and to -2 nJy at 1400 px (the
  Chebyshev ringing that is the coadd trough); the refit gives
  5.8; the response to the visit-fitted lsst-mdet template gives
  3.1, half, because the template ends at ~900 px and is the
  unreliable per-detector kind.  Fainter stars' bumps (0.2-0.5 nJy)
  are below the sky's own order-3+ structure (+-1.5 nJy), so the
  per-input check only works on the brightest stars; the coadd
  stack remains the test for the rest.
- Remaining for step 2: the coadd comparison needs a wing model
  beyond the template extent, i.e. step 4 (pooled per-visit
  template plus aureole).  With it, per input the response is two
  fits (~10 s with the butler reads), combined per cell as in step 1.

## Step 4 status (2026-09-08)

Built and run on visit 2025071900593 (i), all 180 detectors, one
slurm job per detector (`lsst-starsub-make-slurm-template`, 1-6 min
and 2.4 GB each), pooled with `lsst-starsub-visit-template
--from-extracts`; outputs in `~/oh/starsub-visits/templates/`.

- Per detector (`lsst_starsub.template.extract_detector`, ~1 min):
  template-star stamps (G 15.5-17.5) from the restored image with a
  256 px wide-exclusion sky pass, stamps inside a brighter star's
  wide zone dropped; and the flux-normalized wing profile of every
  census star to G 15.5 on the `warp` state (no detector-scale sky
  fit), bright stars (G < 13.5) to 2500 px, the rest to 900 px,
  ambient-referenced.  A per-visit Gaia extract comes from the
  refcat shards over the visit's bounding circle
  (`lsst_starsub.gaia`; a visit spans ~10 tracts).
- Pooled: 3052 stamps, 3337 wing stars.  The wing is fit jointly to
  the stack profile (10-50 px, template units) and the per-G-bin
  cloud (40-2500 px, nJy per unit Gaia flux, error-weighted) as two
  power laws plus the zero point k_in (`fit_wing_model`).  The
  single-law stack fit alone is misleading: its 22-50 px slope
  (-3.87, the canonical i value) is the blend of both components,
  since the cloud is already flatter by 50 px.
- Result: inner slope -4.35, aureole slope -2.60; the model matches
  the stack to 1-4 percent from 10 to 50 px and the cloud to +-10
  percent from 46 to 550 px (G 6-10 stars carry it beyond 190 px).
  Beyond ~650 px the bright stars sit 1.3-1.6 (+-0.5-0.7) above the
  law and beyond 1000 px they are consistent with zero; a single
  visit cannot do better there: per-star local sky offsets of the
  warp state (a few nJy) exceed the wing.  The far cloud is kept in
  every extract for pooling across visits.
- A 6-detector subset is not enough (the aureole slope ran to its
  bound); all detectors, or a spread of a few dozen, are needed.
- Next: the other 59 visits of patch 53 (and the 2562 inputs) to
  see how the parameters vary with seeing (fwhm 1.07-1.25 within
  this visit alone), then the cross-visit far-wing pooling.  New
  extracts also carry the stamp core amplitudes for a direct zero
  point (`k_stamp`) to check the fitted k_in.

## The wide-field test set (2026-09-08)

The weekly run only covers 45 tracts around the two deep fields, so
the representative test moved to DP2 (`--repo dp2_prep_future
--collection LSSTCam/runs/DRP/DP2`): tract 7275 patch 55 i (RA 317,
Dec -14, b -37), 28 visits, 94 inputs, 16-26 inputs per cell, FWHM
0.89-1.69".  DP2 has no `visit_image` (the preliminary image is
calibrated instead) and its `deep_coadd` is the new `CellCoadd`
(`coadd.load_cell_coadd` restores its object background and converts
it); the `calibrateImage` background algorithm is the same, so the
forward model applies unchanged.  Outputs: `~/oh/starsub-visits/
07275-55/` (single-patch restoration, visit-level profiles of the 94
inputs), `07275-cells/i/` (all 100 patches), `templates/` (one
template job per detector of the 28 visits, `extracts-{visit}/`).

- Visit level, 94 inputs: the same trough as the deep field.
  `delivered` at the global reference, G 6-13: -5.3, -6.6, -5.0 x
  10^-3 sigma at 205, 305, 455 px beyond the mask (about -0.15 nJy);
  `warp` +2.3, +1.5, -0.6.
- Only patches 55 and 56 of the tract are covered entirely by the 28
  visits; the tract has 65 visits.  The forward model on more patches
  needs the other 37 visits' templates (about 6700 more jobs).

## Steps

1. **Restore the polynomial at the coadd level.**  DONE (see status
   above).  The statistical restoration stays as the deep-field method
   and as the reference for step 2.

2. **Forward-model the trough.**  Operator DONE (status above); the
   coadd-level comparison waits on the step-4 wing model.  Per input:
   rebuild the fit mask, response = fit(raw) - fit(raw - star wings),
   combine per cell with the input weights, compare the trough model
   with the `None` stack on both tracts.

3. **Sky model on the restored coadd.**  Simplified by the
   structure-only restoration: the restored sky is the `None` sky,
   smooth, so the existing wide-exclusion sky fit applies.  Rework the
   refit that still takes wing at G 13-14 (`flat` state -2 to -3 x
   10^-3 sigma beyond 300 px on the visits).

4. **Per-visit wing characterization, pooled over the focal plane.**
   Tooling DONE and validated on one visit (status above); run over
   the patch-53 visits, check the seeing dependence, pool the far
   wing across visits.  Per-detector templates (20-25 stamps)
   scatter from -2.6 to -4.8 in slope and are not usable.

5. **Carry the visit wing models into the coadd.**  Per cell, the star
   model is the input-weighted mean of the per-visit wing models minus
   their step-2 polynomial responses, with the cell's input weights.
   Keeps the per-visit seeing dependence without special coadds
   (subtraction commutes with the weighted mean, except at clipped
   pixels and masked cores).

6. **Subtract and refit at coadd level.**  Subtract the step-5 model
   from the restored, sky-fitted coadd, then the existing lsst-mdet
   anchor-ring amplitude refit with the deep S/N.  Masking stays.

7. **Integration and validation.**  Either lsst-mdet runs the
   restoration on the fly (~230 small butler gets, 3-5 minutes per
   patch) or the per-cell polynomial structure is written once as a
   small per-patch product and read; the product decouples the codes.
   Validate with the dual-state stack on the final images in all bands
   (only i checked so far) and the amplified injection test.  Check
   DP2 at NERSC with `lsst-starsub-check-datasets` (the DP2 collection
   may carry the rewritten `deep_coadd` instead of
   `deep_coadd_cell_predetection`, which needs a loading path).

Option 2 (pre-subtract on visits and recoadd) is no longer needed as
a fallback: step 1 showed the trough is removable on the existing
coadds.

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
