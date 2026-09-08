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

## Steps

1. **Restore the polynomial at the coadd level.**  DONE (see status
   above).  The statistical restoration stays as the deep-field method
   and as the reference for step 2.

2. **Forward-model the trough.**  The polynomial's response to a star
   is a linear projection of the wing image onto the input's Chebyshev
   fit, with DM's 128 px binning and the detection mask the fit saw.
   Per input: render the census stars' wings from the template on the
   detector, run the same background fit (`SubtractBackgroundTask`
   config of `calibrateImage`: binSize 128, Chebyshev 6x6, weighted,
   masked planes), take the fit surface as the star response.  Check
   against the stored polynomials' structure around bright stars on a
   few tract-2395 inputs, then combine per cell with the input weights
   and compare the trough model with the `None` stack on both tracts.
   No sky terms, no depth dependence; step 4 needs it regardless.  If
   it disagrees, the mask the fit saw (the preliminary image's mask
   plane is stored) is the first suspect.

3. **Sky model on the restored coadd.**  Simplified by the
   structure-only restoration: the restored sky is the `None` sky,
   smooth, so the existing wide-exclusion sky fit applies.  Rework the
   refit that still takes wing at G 13-14 (`flat` state -2 to -3 x
   10^-3 sigma beyond 300 px on the visits).

4. **Per-visit wing characterization, pooled over the focal plane.**
   One template and aureole per visit from all its detectors, stamps
   inside bright-star halos excluded; per-detector templates (20-25
   stamps) scatter from -2.6 to -4.8 in slope and are not usable.

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
