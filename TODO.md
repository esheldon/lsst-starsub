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
- Coadd level, all 100 patches (372 stars at G < 13, coadd sigma
  ~6.6 nJy): the dual-state pattern of the deep field.  `none` at
  the global reference -10.7, -15.1, -7.6 x 10^-3 sigma at 205, 305,
  455 px (about -0.1 nJy); `restored` +13.7, +8.0, +2.6; the
  `object` state's collar -62 at 55 px.  Stacks in
  `07275-cells/stack-cells-i-{global,local}.*`.
- Templates of all 28 visits (4,945 detector jobs; the profile
  neighbor exclusion had to be made local, a dense-field O(N^2)
  cost that took jobs from 1 to over 15 minutes; 30 s after).
  Table in `templates/params-07275-55.txt`, plot
  `params-vs-fwhm-07275-55.png`.  Findings:
  - the inner-law amplitude ln_a rises with seeing, from -0.6 at
    0.77" to ~2 at 1.45", about twice as steep as the coadd
    canonical relation (0.92 per arcsec); the stack zero point
    k_stamp falls with seeing as the 6 px core loses flux.
  - the physical wing, k_in x T(r) in nJy per unit Gaia flux, is
    stable across visits: 1.1-1.5 x 10^4 at 300 px (one outlier at
    5 x 10^3), 2.1-3.5 x 10^5 at 100 px, a factor 3 scatter at
    1000 px where it is poorly constrained.  The fitted components
    trade off against each other; the sum is what matters.
  - k_stamp / k_in is 0.7-1.0 (the direct stamp zero point is ~15
    percent below the continuity fit's); two poor-seeing visits
    (2025081600397 at 1.44", 2025090200262) hit the slope grid
    bounds and need a more robust fit.
- Forward model on patches 55 and 56 (`lsst-starsub-cell-forward`,
  `lsst_starsub.trough`: per input the census wings rendered from
  the visit template, the star_background refit, response =
  fit(raw) - fit(raw - stars), order-2 removed, coadded per cell
  with the input weights; ~15 s per input, 4.5 min per patch with
  4 workers).  Compared as images, no pixel noise
  (`forward/forward-vs-restored-55-56.png`): what the restoration
  adds (restored - none) against the template response coadd
  (forward - none), mean over stars in nJy:
  - G 6-11 (2 stars): 0.99 vs 0.96 at 55 px beyond the mask, 0.67
    vs 0.73 at 205, 0.31 vs 0.38 at 455.  The forward model
    reproduces the polynomials' star response to 10-20 percent
    with no free parameter (the zero point is the template's).
  - G 11-13 (9 stars): 0.31 vs 0.30 near the mask, 0.15 vs 0.25 at
    455 px.
  - fainter: both are at or below the per-cell offset floor of the
    polynomial coadd (~0.4 nJy rms at 22 inputs per cell, ~0.1 nJy
    on the mean over 10 stars), so nothing can be said; the noisy
    profile stacks agree within errors.
  Statistics need the whole tract: templates for its other 37
  visits (about 6,700 one-minute jobs), then the forward model on
  all 100 patches.
- One wing per band suffices for the forward model.  A canonical
  wing (`templates/canonical-wing-07275-i.fits`: the median over
  the 28 visits of k_in T_v(r), nJy per unit Gaia flux; visit
  scatter 14, 10, 8 percent at 50, 100, 300 px, 46 percent at
  1000 px) used for every input reproduces the per-visit forward
  model to 2 percent at every radius and magnitude on patches 55
  and 56 (`forward/forward-compare-55-56.png`).  So production needs
  no per-visit image processing for the trough model: the
  polynomial responses depend on the mask and the wing, and the
  wing is the same to the precision that matters.
- Tract-wide forward model with the canonical wing (100 single-core
  jobs, ~40 s per input on slurm, 22 of them past the 1 h limit and
  resubmitted with 3 h; 75 patches, 277 stars at G < 13 in the
  first pass).  Profile stacks at the global reference, 10^-3
  sigma at 205, 305, 455 px:
  - G 6-13: none -10.2, -12.8, -7.9; restored +22.0, +14.9, +5.9;
    forward +22.9, +13.2, +8.9 (errors 3-6).  The forward model
    reproduces the restoration within 1 sigma at every radius.
  - G 13-14: restored -5.0, -2.8, -11.1; forward +5.0, +4.7, -3.8
    (errors 4-6): the forward model sits ~10 x 10^-3 sigma (0.07
    nJy) above the restoration, 1.5 sigma, the same sense as the
    55/56 image comparison.
  - fainter bins: all three states agree within errors.
  `07275-cells/stack-forward-canonical-i-{global,local}.*`.
- Noise-free image comparison over the 75 patches
  (`forward/forward-compare-75patches.png`), restored - none against
  the canonical forward model, mean over stars in nJy at 55, 205,
  455 px beyond the mask:
  - G 6-11 (55 stars): 1.13, 0.75, 0.16 vs 1.26, 0.86, 0.21: the
    model is 12-15 percent high near the star, 35 percent at 455 px.
  - G 11-13 (222): 0.21, 0.18, 0.11 vs 0.14, 0.12, 0.07: 30-35
    percent low.
  - G 13-14 (282): 0.01, 0.01, -0.01 vs 0.05, 0.04, 0.02.
  - fainter: both within +-0.04 of zero.
  Each is a 2-2.5 sigma difference against the per-cell offset
  floor (0.4 nJy per cell over sqrt(n) stars), with opposite signs
  in the two bright bins, so a magnitude-dependent residual of
  order 30 percent of the bump, about 0.05 nJy (0.01 sigma) at
  200-450 px.  Suspects: the far wing beyond 500 px (46 percent
  visit scatter, the bright stars' bumps are driven by the wing
  outside their ~200 px masks) and the rebuilt mask size for stars
  near the adaptive detection threshold (G 11-13).
- Steps 5-6 first pass (`lsst-starsub-cell-clean`: the joint
  sky-plus-template subtraction of handle_stars_visit run on a
  patch's `none`, `restored` or `forward` = none + R state; 96
  patches, 3 routes, 90 s each).  The three routes give the same
  final residual within errors, e.g. G 6-13 (362 stars) at 55, 105,
  205 px beyond the mask: none -13.6, -10.0, +1.0; forward -14.4,
  -9.8, +1.2; restored -13.5, -8.9, +0.7 (errors ~2).  Reason: the
  second-round 64 px sky pass is excluded only 12 px beyond every
  mask (PRE_GROW; the lsst_mdet coadd route uses 128 px around the
  bright stars, PRE_GROW_BRIGHT), so it follows the trough, R and
  the wings themselves beyond ~76 px of the masks.  With that
  configuration the trough is irrelevant to the final image and
  the residual collar near the masks (-14 x 10^-3 sigma at 55 px
  for G < 13, -10 for G 13-14, -6 for G 14-15.2) is a property of
  the subtraction, not of the trough.  Next: the same three routes
  with the bright-star exclusion of the coadd route in the final
  sky passes, where the trough correction has to carry the load.
  `07275-cells/clean/clean-compare-100patches.png`.
- Same three routes with the coadd route's 128 px bright-star
  exclusion in the 64 px sky passes (`--bright-grow 128`,
  `clean-bright/`): still the same residual within errors.  G 6-13
  at 55, 105, 205 px: none -0.6, -21.7, -2.2; forward +1.6, -18.6,
  -2.0; restored +3.4, -17.7, -2.2 (errors 4.5, 3.4, 1.7).  The
  trough correction moves the residual by +3 x 10^-3 sigma at 105
  px, one sigma.  Conclusion: a 64 px sky pass allowed anywhere
  within ~500 px of a bright star follows the trough whether or not
  it was corrected; the correction can only matter for a sky model
  kept beyond the trough (round 1's wide exclusion alone), and no
  configuration in use does that.  The residual that remains, a
  -20 x 10^-3 sigma dip at ~105 px beyond the bright-star masks
  with the bright exclusion (-14 at 55 px without it), is the
  subtraction's own collar: template shape and anchor-ring
  amplitudes, the lsst_mdet "monster collar" problem, independent
  of the trough.  `clean-bright/clean-compare-100patches.png`.
- The large-scale negative tail of the stacks is the polynomials'
  ringing.  Mean image level against distance to the nearest G < 13
  star over 30 patches (patch median removed, 10^-3 sigma): `none`
  has the trough at 300-600 px (-24, -21) and is back to zero by
  800 px; `restored` and `forward` are positive to 500 px, zero at
  500-800 px, then a negative ring of -15 to -20 at 800-1600 px
  (restored -15.0, -16.2, -16.7; forward -20.2, -19.2, -18.5, errors
  6-10).  A 6x6 Chebyshev responding to a bump overshoots negative
  beyond it; the stored polynomials carry that ring and the forward
  model reproduces it.  It is a smooth 1000 px feature, exactly
  what the step-3 sky refit removes after the star model comes off;
  it also means the local reference at 500-600 px beyond the mask
  sits inside the ring for bright stars.
- Zero point: within the template stars the stamp zero point (core
  amplitude over Gaia flux) falls with brightness, 4.16 to 3.85 x
  10^12 from G 17-17.5 to 16-16.5 on the 0.77" visit (8 percent),
  an aperture loss of the 6 px core for brighter stars (brighter-
  fatter), so the median stamp value is biased low and the
  continuity fit's k_in, tied to the G < 15.5 wing stars, is the
  right scale for the wings; the bright-star forward match above
  used it.  The poor-seeing visit 2025081600397 (1.44") is not a
  failed fit: its stack is a single -3.5 power law to 200 px with
  no separate aureole, so the two-component model degenerates
  (aureole slope at its bound, tiny amplitude) while the physical
  wing, 1.26 x 10^4 at 300 px, is normal.

## Steps

1. **Restore the polynomial at the coadd level.**  DONE (see status
   above).  The statistical restoration stays as the deep-field method
   and as the reference for step 2.

2. **Forward-model the trough.**  DONE.  The response operator
   reproduces the stored fits; with the canonical wing it predicts
   the restoration to 10-30 percent for G < 13 over the tract.

3. **Sky model.**  Superseded by the steps 5-6 finding: the 64 px sky
   passes of the joint routine follow the trough and the correction
   alike, so no separate sky rework is needed for the trough.

4. **Per-visit wing characterization, pooled over the focal plane.**
   DONE on the 65 visits of tract 7275: one canonical wing per band
   suffices for the trough model (2 percent).  Left: the cross-visit
   far-wing pooling beyond 500 px if the bright-star residual of the
   forward model needs it, and the other bands.

5. **Carry the wing model into the coadd.**  DONE as the response
   coadd R (`lsst-starsub-cell-forward`); the star model itself is
   built on the coadd (step 6), the per-visit model is not needed.

6. **Subtract and refit at coadd level.**  DONE as
   `lsst-starsub-cell-clean` (status above).  Result: the input route
   (none / restored / forward) does not change the final residual in
   either sky configuration; the residual collar around bright stars
   is the subtraction's own.  Star models compared over the tract
   (forward route, 128 px bright exclusion; G 6-13, 362 stars, at
   55, 105, 205 px beyond the mask, 10^-3 sigma):
   - stamp template with anchor-ring amplitudes: +1.6, -18.6, -2.0
   - canonical wing, pure prediction: -20.1, -27.8, -5.0
   - canonical shape with anchor-ring amplitudes: -2.5, -23.0, -2.0
   The pure prediction over-subtracts by ~15 percent near the bright
   stars, the same excess the forward-model comparison found for
   G < 11; the amplitude fit removes that at the anchor but the -20
   to -28 dip at ~105 px survives every model.  Fainter bins agree
   within 1-2 sigma for all three.  So the dip is not the wing
   model's shape at the level the two very different shapes test:
   either the real wings fall faster than both between 55 and 105
   px, or the sky interpolated under the 128 px exclusion is too
   high there (it would bias every model the same way; in the 12 px
   configuration the collar sits at 55 px instead).  The injection
   test (step 7) separates the two.
   `clean-bright/clean-compare-model-100patches.png`.

7. **Injection simulation** (the next step, 2026-09-08).  The data
   cannot separate an algorithm artifact from star physics in the
   collar (+60 x 10^-3 sigma at the mask edge, -20 at ~100 px, zero
   by 200 px for G < 13; +45 / -20 at 0 / 20 px for the faint stars):
   the wing model's shape, the anchor-ring bias, the interpolated sky
   under the exclusion zone and real-star structure (color-dependent
   halos, the spikes left in) all give the same signature, and the
   real wings are not known independently of the same measurement.
   So: inject synthetic stars with a known wing into the real coadds
   (`lsst-starsub-cell-inject`):
   - into the `forward` state, so the trough is out of the way;
     random positions, G 6-17 spanning the bins, entered into the
     Gaia census so masking and subtraction treat them as real; the
     cores from the coadd's own PSF, so only the wing is a model;
     the canonical wing as the truth.
   - run the clean tool; compare the residual around the injected
     stars (truth known) with the residual around the real stars in
     the same bins.  Same collar: it is the algorithm, iterate with
     truth in hand (ring radius, template junction, sky
     interpolation).  Clean: the real wings differ from the model and
     the collar is physics to be modeled.
   - a wing deliberately 30 percent off, for the residual's
     sensitivity to the wing model (the uncertainty left in the
     trough work); and an amplified variant, wings 10x brighter, to
     put the systematics above the noise.
   - no butler needed: it runs on the cell files with the rendering
     and profile code; an afternoon.  Not worth simulating: the
     trough itself (provenance and operator established) and, yet,
     the full shear simulation, which is the final arbiter but a
     bigger build; the injection machinery is half of it.

   Implemented 2026-09-08 as `--inject` on `lsst-starsub-cell-clean`
   (module `lsst_starsub/inject.py`): each star is the cell's coadd
   PSF core, scaled to the canonical wing's core flux (sum within 6
   px, the stamp normalization), blended over 8-12 px into the
   canonical wing times the wing scale; integer positions clear of
   existing masks; the stars appended to the Gaia census; per-patch
   plan 2 x G 8-13, 2 x 13-14, 3 x 14-15.2, 3 x 15.2-16, 4 x 16-17,
   seed 1000 + patch.  Two extra image states carry the truth:
   `perfect` = flat - truth (the residual under a perfect star
   model, i.e. minus the sky-interpolation error) and `model_error`
   = star model - truth; residual = perfect - model_error.  The
   profile and census tables get an `injected` flag.  The real
   stars' core flux per unit Gaia flux on patch 55 is 3.8e12
   against the canonical core's 4.3e12 (the k_stamp/k_in ratio
   again).  Runs in `07275-cells/inject/` (tags inj1, inj13, inj10
   for wing scales 1, 1.3, 10; 2 min, 2.7 GB per patch);
   `scripts/compare_inject.py` stacks injected against real.

   Result (96 patches, template star model, 128 px bright
   exclusion; 10^-3 sigma at 55, 105, 205, 305, 455 px beyond the
   mask; G 8-13 bin, 192 injected against 362 real):
   - truth wing (mean):        393, 242, 118, 68, 38
   - injected residual:        +43, +7, -2, +1, +4
   - real residual:            +5, -17, -3, -1, +3
   - perfect-model residual:   -75, -75, -49, -29, -19 (the sky is
     too high by 20-50 percent of the wing, out to 500 px)
   - model - truth:            -121, -84, -46, -30, -25 (the model
     is 30-45 percent low inside the template extent, and absent
     beyond it: the truth wing runs to 3000 px, the template to 3x
     the mask radius)
   The two errors cancel to within 10 percent of the wing beyond
   100 px: the 64 px sky passes absorb whatever wing the model
   leaves, and the anchor-ring amplitude, fit on that flattened
   image, is low in turn.  What survives is the collar inside the
   exclusion zone, where the boxes cannot follow: +43 at 55 px for
   the canonical shape.  Wing scale 1.3 scales the collar (+58);
   wing scale 10 gives +1125 / +297 / -25, with the sky 1000 low:
   the errors are fractional, so the amplified variant is only the
   same story louder.  The fainter bins (13-17) show injected and
   real residuals consistent within 1-2 sigma of ~3 x 10^-3 sigma.
   The real stars' signature (+5 at 55, -17 at 105) differs from
   the canonical wing's (+43, +7), so the real wings are not the
   canonical shape at the 40 x 10^-3 sigma level in the first 100
   px beyond the mask, or carry structure the model lacks.
   Re-fitting the round-2 sky on the star-subtracted image
   (`--sky-from residual`) moves the bright-bin sky error from -97
   to -73 on four patches: the re-add is a minor part.  Control
   runs with the canonical wing as the star model (pure
   prediction: model error zero by construction; and shape with
   fitted amplitudes) separate the sky interpolation from the
   template shape: `inject/clean-forward-canonical[-fit]-inj1-*`.

   Control runs (96 patches, G 8-13 bin, same five radii):
   - canonical wing as pure prediction (model error zero, it
     extends to 3000 px): sky error -10, -15, -1, +2, +5; injected
     residual -14, -18, -3, +2, +4; real residual -21, -32, -6, -1,
     +4.  The algorithm's own collar with a perfect model is -15 at
     105 px (6 percent of the wing, the interpolation under the
     exclusion); the real stars sit 7 +- 8 and 14 +- 6 below the
     injected ones at 55 and 105 px, i.e. the canonical wing is
     within ~5 percent of the real wings there.
   - canonical shape with fitted amplitudes: injected residual +40,
     +3, -2; sky -53, -55, -35; model -98, -60, -35: the same collar
     as the stamp template, and the amplitude 25 percent low with
     the shape exact.  So the collar is the sky-amplitude coupling,
     not the template shape: the wing beyond the exclusion (and
     beyond the template extent, 3x the mask radius) goes into the
     64 px sky, is interpolated under the exclusion, the anchor
     ring then reads the wing 20-25 percent low, and the final sky
     pass absorbs what the model left.  The final image is clean
     beyond ~150 px by construction; the collar inside is what the
     interpolation cannot cancel.
   Variants on 12 patches (24 injected, noisy, baseline collar
   +56 / +12 at 55 / 105 px): sky refit on the star-subtracted
   image with 4 rounds (`--sky-from residual --nround 4`): +13 /
   -3, sky and model errors still 20 percent; a 400 px bright
   exclusion: +43 / +22 and noisier.  Next iteration: the template
   extent to the wing's full range (the canonical shape to 3000
   px, as the pure prediction has) with the residual-sky rounds,
   which should remove the far-wing leak into the sky as well;
   then the real stars under that configuration, and the pure
   prediction as the fallback (its collar is -15).
   `inject/inject-compare-*-100patches.png`.

7b. **Ideal-conditions simulation** (2026-09-09, `lsst-starsub-sim`,
   module `lsst_starsub/sim.py`).  The question behind it: how much
   of the limitation is the pre-processing, and would the algorithm
   work with the processing in our hands.  One pixel grid, a smooth
   sky (1900 nJy per visit pixel with a 2 percent gradient), the
   real Gaia census of the patch as the stars with the canonical
   wing as truth (Gaussian core of the visit's seeing scaled to the
   canonical core flux, blended over 8-12 px), variance following
   the image as on the data (26 nJy sigma at the sky level: this
   is what keeps the moderately bright stars out of the adaptive
   detection), saturation at 1e5 nJy, 20 visits of FWHM 0.7-1.3
   arcsec, each fit on a 4100 px detector frame with the patch at a
   random offset.  The calibrateImage star_background pass is
   reproduced in full: first-pass 50 sigma detection dilated 10 px,
   the adaptive detection at 0.2 x median sky in pixel-sigma units
   with the per-amplifier loop of `_remeasure_star_background`
   (without it the threshold settles at the sky level in the
   convolved image and half the detector is one footprint: the fit
   was off by 70 nJy), footprints grown 70 psf sigma, the weighted
   6x6 Chebyshev on 128 px bins.  The coadd is the visit mean; the
   response coadd R is the fit(raw) - fit(raw - stars) mean.  The
   clean runs through the same `run_clean` as the data with the
   truth star coadd, so `scripts/compare_inject.py` reads the
   output (tag `sim{seed}-{state}...`); `scripts/compare_sim_trough.py`
   stacks the trough (delivered - ideal) and what R leaves.
   Patch 55 (307 census stars, 161 on image; 5 min with 4
   workers): the single-visit fit error is 0.13 nJy rms; the
   trough around G < 13 stars is -16 x 10^-3 sigma flat to 455 px
   (the data: -15 to -20), -6.5 for G 13-14, and none + R is zero
   within 1 x 10^-3 sigma in every bin: the response coadd
   recovers the polynomial's imprint under ideal conditions.  The
   cleaning shows the data's collar (+58 at 55 px on the two bright
   stars, sky and model errors of opposite sign).  Runs in
   `~/oh/starsub-visits/sim/` (one slurm job per patch: the sim,
   then the template / canonical-fit / canonical / residual-sky /
   none / restored variants on the saved sim file).

   Tract result (100 patches, 372 stars G 8-13 with truth, 10^-3
   sigma at 55, 105, 205, 305, 455 px beyond the mask):
   - trough, delivered coadd: -32, -31, -27, -23, -16 (G 13-14: -5);
     delivered + R: 0.0 +- 0.2 in every bin.  The response coadd
     recovers the polynomial's imprint exactly under ideal
     conditions; the forward model is validated.
   - cleaning, template model (truth wing 263, 145, 64, 36, 19):
     residual +27, +4, +1; sky error -34, -33, -25, -17, -10; model
     error -62, -38, -30, -21, -11.  As fractions of the wing this
     is the data's signature (sky 13-50 percent, model 24-58
     percent, cancelling beyond 100 px, the collar inside the
     exclusion).  With a perfect wing shape, no galaxies and a
     smooth sky the collar is there: it is the sequential
     sky-then-amplitude scheme, not the data.
   - canonical shape with fitted amplitudes: the same (+24, 0);
     pure prediction: -10, -10, -2, +1, 0 with the sky error the
     same, i.e. the interpolation under the exclusion alone.
   - the input state does not matter (none +26 / +3, restored +27
     / +4): the 64 px passes absorb the trough, as on the data.
   - the residual-sky rounds do not help (+26, +9).
   Toy linear fits (scripts run in the session, 2026-09-09): with
   known profiles the joint solve of amplitudes plus sky is at the
   noise floor, ~2 x 10^-3 sigma at the mask edge, for a constant,
   a plane, or a global cubic sky; off-image stars must be pinned
   to the prediction (their wing on the image is a plane).  A
   single plane over a CCD is not enough even for a smooth sky.
   Real sky planes (`scripts/sky_smoothness.py`): after a cubic the
   coadd has 0.23-0.43 nJy rms at 512-64 px (3-5x the noise-only
   passes), the visits 0.75-1.4 nJy; the far wings are not it; the
   pattern repeats on the same detector between consecutive visits
   of one night (0.54) but not two nights apart (-0.06), so not
   the flat; the pipeline's 128 px layer sees part of it (0.3-0.4)
   and skyCorr anticorrelates (-0.2 to -0.3).  The flat is excluded
   (no correlation with its structure; detector 004's two visits
   share one flat).  By eye (`skytest/sky-*.fits`,
   `image-boxcar64.png`, the direct masked boxcar of the image; the
   `sky_model` planes show the sep mesh's rectangles and are not
   the sky): the field between the stars is dense with faint
   sources, and the "sky structure" is their light around and
   below the 1.5 sigma segmentation, blotchy on 100-200 px at
   +-1-2 nJy everywhere, not concentrated near the stars (rms 1.3
   nJy within 12 px of a mask, 1.0 beyond 256 px on 044).  That
   reading fits every test: it repeats on the co-pointed pair, it
   survives in the coadd (0.4 nJy), and the noisier visit leaks
   more of it past the segmentation (0.85 vs 0.5 nJy excess).  So
   neither sky physics nor processing: undetected source light,
   present on any image, which the joint fit's 256 px mesh handles
   (7c).

7c. **The joint fit** (2026-09-09, `lsst_starsub/joint.py`, star
   model `joint` in both tools): amplitudes of the on-image stars
   brighter than G 17 (canonical shape, A = 1 the prediction) and a
   bilinear sky mesh (256 px nodes) solved together by weighted
   least squares on 4 x 4 binned cells outside the star masks,
   detections and bad pixels; off-image and fainter stars pinned
   to the prediction; two passes, the second with the residual
   re-segmented.  No exclusion zone, no interpolation.  Results
   (10^-3 sigma at 55, 105, 205, 305, 455 px beyond the mask):
   - simulation, 100 patches, G 8-13 (372 stars, truth wing 263,
     145, 64, 36, 19): residual +0.9 +- 1.1, +0.7, -1.0, +0.2,
     -0.4; sky error +1.7, +1.2, -0.5, +0.8, +0.8; model error
     -0.7, -0.4, -0.5, -0.4, -0.2.  Every bin within 4 x 10^-3
     sigma; the 128 px mesh is the same.  Against the sequential
     scheme's +27 / +4 with sky and model errors of 30-60 (the
     noise floor the toy fits predicted).
   - real coadds with injected stars, 96 patches: injected
     residual -3 +- 3, -1, +2, +1, +2 (was +43, +7, -2); model
     error +6 +- 7, +6 +- 4, +6 +- 2, +5, +4 (a ~2-5 percent high
     amplitude on the injected stars); real stars -1 +- 3, -9.5
     +- 2.7, -3, -1, +2 (was +5, -17, -3).  The remaining -9.5 at
     105 px on the real stars is the real wings against the
     canonical shape, ~4 percent of the wing there, as the pure
     prediction had found; the fainter bins are within 3.
   2 min and 2.5 GB per patch.  Runs: `sim/clean-sim*-forward-joint[128]-*`,
   `07275-cells/inject/clean-forward-joint-inj1-*`.
   - on the delivered coadd (`--state none`, trough in) the joint
     fit gives the same numbers as on the forward state: simulation
     +0.9, +0.7, -0.9 with sky error +2.0, +1.5, -0.4 (forward: +0.9,
     +0.7, -1.0; +1.7, +1.2, -0.5); data injected -3.3, -0.9, +1.6,
     real -1.0, -10.7, -2.5 (forward: -3.2, -0.6, +1.7; -1.2, -9.5,
     -3.4).  The 256 px mesh absorbs the trough (a 6x6 Chebyshev
     per input has no structure below ~700 px).  So production
     needs no response coadd: read the delivered coadd, one joint
     fit, subtract.  The forward model stays as the validated
     account of the trough.
   - the segmentation inside the joint fit is now the
     metadetection one (`joint.deep_segmentation`: 0.8 arcsec
     kernel, 0.8 sigma kernel-scale threshold, minarea 4, grown 4
     px): on visit detectors it masks twice the area of the 1.5
     sigma per-pixel pass and removes 40 percent of the excess
     faint-source light from the sky boxes (64 px boxcar rms 1.44
     to 0.99 nJy on 044, 0.91 to 0.60 on 004; noise 0.53 / 0.35);
     what remains is below even that threshold.  Runs with the tag
     `joint-deep`.  On the injected coadds it excludes 11 percent
     of the good pixels (the coadd is deeper) and changes little:
     injected residual -3.6, -2.0, -0.4 (shallow -3.2, -0.6, +1.7),
     real -4.0, -11.3, -3.8 (shallow -1.2, -9.5, -3.4), the
     sky-error scatter 10-20 percent smaller (bright bin +-7.2 vs
     +-8.8 at 55 px).  Kept as the default for being the
     principled choice; the source light below it is what limits
     the sky term on real coadds (+-3 against +-1 in the
     source-free simulation).
   - NERSC has no visit images (the user, 2026-09-09), so the
     shipped canonical wing was tested for transferability: the
     joint fit on the delivered coadds of tracts 2395 (100
     patches, weekly collection, deep field) and 2562 (31
     patches) with the 7275 wing, straight from the butler
     (`lsst-starsub-cell-clean` without `--cell-file` stitches the
     delivered coadd; 42 s per patch).  Real-star residual, G 6-13,
     at 55, 105, 205, 305, 455 px: 7275 -1, -10, -3, -1, +2 (362
     stars); 2395 +4, -5, -5, -1, +2 (260); 2562 -2, -9, +2, -7,
     -3 (70).  Fainter bins within +-8, mostly a +5 to +8 at 55 px
     for G 14-16.  The 7275 wing transfers across regions and a
     processing run at the 10 x 10^-3 sigma level.
   - the coadd-derived shape (`lsst_starsub/shape.py`,
     `--joint-shape coadd`: lsst_mdet's stamp stack, inner law and
     aureole on a WIDE_BW flattening, zero point from the stamps'
     cores) is wired in.  Against the truth on the simulated
     patch 55 the stamp part is right (ratio 0.8-1.0 at 20-44 px;
     the core differs by the coadd's seeing) but the aureole is
     not measurable from the mid-bright cloud on a coadd: with
     lsst_mdet's joint scale-plus-aureole fit the amplitude came
     out 3x high (a degeneracy where the aureole dominates the
     40-260 px cloud); with the scale fixed by the core zero
     point the slope ran to the flattest allowed (-1.5, truth
     -2.6) because the 256 px box-flattened sky leaves ~1 nJy
     residuals where a G 14.5 star's wing at 200 px is 0.06 nJy.
     So the far shape must come from the bright stars after the
     joint fit has taken the sky out: a self-calibration, the
     stacked residual-to-model ratio of the G < 13 stars over a
     tract in radial bins, applied as a correction to the
     provisional shape and iterated.  Production would then be
     two-stage per tract: joint fit with the provisional shape
     (the shipped wing, or the coadd stamps plus lsst_mdet's
     continuity prior), stack, correct, refit.
   - the self-calibration (`scripts/selfcal_wing.py`): per bright
     on-image star (G < 13, not injected) the residual profile of
     the joint fit (the 'profiles' table, 12-900 px log bins)
     divided by its model A F T(r); per bin a clipped weighted
     mean over stars with the weights (npix x model^2) capped at
     their 80th percentile and the error from the star-to-star
     scatter (with plain inverse-variance weights one star
     dominated each bin and tract 2562 swung by +-40 percent);
     the correction applied only where measured to better than 3
     percent, smoothed over 3 bins, tapered to zero inward and
     held outward.  Results c(r) at 93, 116, 144, 178, 221, 274
     px: 7275 (352 stars) +2.9, +3.1, +2.2, +0.4, -2.7, -0.9
     percent (+-1.3-2.7); 2395 (257) +2.7, +3.2, +3.6, -1.3, -3.3,
     +1.0 (+-0.6-1.0); 2562 (68) consistent within +-3-8.  The two
     well-measured tracts agree: the canonical wing is ~3 percent
     low at 90-150 px and ~2-3 percent high at 200-250 px; beyond
     300 px the bright stars of a tract cannot measure it to 3
     percent.  Wings in `selfcal/wing-selfcal-{tract}-i.fits`.
     Applied on 7275 (96 patches, no injection, `07275-cells/selfcal/`):
     real-star residual G 6-13 -2.4, -8.4, -4.3, +0.5, +2.5 against
     -2.5, -8.9, -4.2, +0.8, +2.6 with the shipped wing: no change.
     A 2-3 percent correction in r bins dominated by the brightest
     stars does not move a d - r_mask stack that weights every
     star equally, so the -9 at 105 px is not a single shape
     error common to all bright stars; the broad calibration
     measures c(r) per G bin to see whether the shape depends on
     brightness (saturation, bleed trails, brighter-fatter in the
     coadd).
   - beyond ~500 px the far wing cannot be calibrated on the
     delivered coadd at all: a 256 px mesh absorbs it, a coarser
     mesh lets the trough (0.13 nJy at 800-1600 px, the same size
     as a G 10 wing there) into the residual.  The far shape stays
     the visit-derived one; the calibration corrects 90-400 px.
   - the broad calibration (i band, 2026-09-09): a stratified
     random sample of full DP2 tracts (6 RA x 4 Dec cells over the
     1602 tracts with >= 90 i-band patches, one per occupied cell:
     `broadcal/tracts-i.txt`), pass 1 the joint fit with the
     shipped wing on every patch (`broadcal/pass1/`), the pooled
     correction per G bin, pass 2 with the corrected wing.
     Pass 1: the Gaia maker skips |b| < 20 deg, so 12 tracts, 1200
     patches (10 min, 3 GB at most); 65 singular solves (a free
     star with no cell under it, now pinned).  Pooled correction
     from 2679 stars G < 13 at 93, 116, 144, 178, 221, 274, 341,
     423 px: +3.2, +3.3, +3.5, -0.2, -2.8, +0.6, +2.1, -2.3
     percent (+-0.3-1.3), the same +3 / -3 pattern as the single
     tracts with 5x smaller errors; per G bin it is broadly the
     same shape (G 6-10: +4.1 at 178, -4.2 at 423; G 10-11.5: +2.9
     at 144, -2.2 at 221, +6.6 at 341; G 11.5-13: +3.2 to +4.0 at
     93-144, -5.0 at 221), so no strong magnitude term; usable to
     650 px.  `broadcal/wing-broadcal-i-pass1.fits`.
   - the amplitude census of pass 1 (29,914 free stars): 3 percent
     negative at every magnitude, star-to-star scatter 25-36
     percent (the colour term of i-band flux against Gaia G: real,
     expected).  Of the 73 negative bright stars 26 are pairs
     closer than 30 px (the partner at +100), 18 are within 300 px
     of the edge, 3 are G < 6 stars whose wing covers the patch
     (degenerate with the mesh), most of the rest wider pairs with
     merged masks.  Fixed by a Gaussian prior on each amplitude
     about the prediction with width 0.3 (the colour scatter;
     `joint.PRIOR_SIGMA`, `--joint-prior`), in the normal matrix's
     units sky_sigma^2 / 0.3^2 (a first version in inverse units
     did nothing): the pairs go to ~1, no negatives, isolated stars
     move by 1-2 percent.  Pass 2 runs with it, plus a control pass
     2b (shipped wing + prior) to separate the two effects.
   - Pass 2 (corrected wing + prior, 1067 patches; the 40 failures
     per pass are patches absent from DP2) does not converge: the
     pooled residual / model comes back the same as pass 1, +3.2,
     +3.1, +3.5 percent at 93-144 px, -1.9 at 221, although the
     wing was raised by exactly that (verified: +3.2 percent at 93
     px in the file the jobs used).  Pass 2b (shipped wing + prior)
     is the same again: the prior does not change the pooled
     ratio.  The free amplitudes absorb an r-correction; what is
     left is orthogonal to the shape under the fit's weights.  In
     units of the mask radius the pattern is a plateau: G 11.5-13
     stars are +3 percent above the model from the mask edge to 2
     mask radii (60-200 px) and zero beyond; G 10-11.5 +3.7 percent
     in the first 15 percent of a mask radius, then -1.5, then +4
     at 2-3 radii; G 6-10 +1.6 percent at the edge and zero
     elsewhere.  So the residual is a magnitude-dependent shape
     difference concentrated at 1-2 mask radii, largest for the
     fainter bright stars (for G 12.5: 3 percent of 3 nJy at 100 px
     = 14 x 10^-3 sigma, falling to 2 by 200 px), not a single
     wing shape error.  A per-star r-correction cannot remove it;
     a magnitude term in the shape (the inner wing relatively
     shallower for fainter bright stars, as the visit stamps'
     falling zero point k_stamp / k_in hinted) is the candidate.
     `broadcal/wing-broadcal-i-pass{1,2,2b}.fits`.
   - the equal-weight d - r_mask stacks of the real stars G 6-13
     per tract (150-280 stars each, errors +-2 at 55 and 105 px,
     +-1-1.5 beyond): all three passes within +-5 x 10^-3 sigma
     at every radius on 8 of 9 tracts (10804 is noisier, +-8);
     at 105 px the tract mean goes -2.2 (pass 1), -1.8 (2b), -0.6
     (pass 2), so the broad correction helps there a little; 55
     px is +3 to +5 on three tracts, -2 to -3 on three.  The
     delivered-coadd joint fit with the broad-calibrated wing and
     the prior leaves the bright stars within a few x 10^-3 sigma
     of flat beyond the mask on a survey-wide sample; the residual
     structure that remains is the magnitude-dependent plateau
     above.
   - is the plateau the star's own detected features (spikes,
     halo blobs), which the joint fit excludes through the deep
     segmentation but the residual profiles keep?  Checked on
     tract 2558 with a second set of profile tables measured on
     the fit's own pixels (`profiles_fitpix`,
     `profiles_dmask_fitpix`, with their own ambient reference:
     the masked pixels sit 28 x 10^-3 sigma below the all-pixel
     level, the faint-source light the mask removes).  135 stars
     G < 13: the residual / model ratio on the fit's pixels equals
     the all-pixel one within 1-2 sigma at every radius (144 px:
     +3.2 vs +3.2 percent; 178: -2.9 vs -3.9; 221: -2.9 vs -6.3,
     errors 1.5-3), and the equal-weight stacks agree within 1.5
     sigma.  So the plateau is in the pixels the fit used: wing
     light with a shape the model lacks, not spikes.  The
     magnitude term is worth building.
   - slurm on milano / rubin:default is a preemptable QOS
     ("CANCELLED by 0"): pass 2 lost 94 patches and 2b 71 to
     preemption, the fit-pixel run 68 then 48; `scripts/
     resubmit_preempted.sh JOBDIR PATTERN` sweeps and resubmits.
     Always sweep before counting outputs.
   - the magnitude term (`lsst_starsub/wing.py` WingModel, HDU
     'magterm' in the wing file, `scripts/magterm_wing.py`):
     T_G(r) = T(r) (1 + c(G, r / r_mask)), c measured on pass 2b
     (28,583 stars) in 5 G bins x 9 bins of r / r_mask, applied
     where measured to 3 percent.  It grows toward the faint
     end: G 6-10 within +-1 percent; 10-11.5 +4 at the mask edge,
     -1.5 at 1.3-1.75, +4 at 2-3 radii; 11.5-13 +3 to +4 at 1-2
     radii; 13-14.5 +4 to +12 at 1-2.5 radii, -10 at 3-4; 14.5-17
     +8 to +20 at 1-3 radii.  Read as the coadd PSF's wing at
     20-100 px being broader than the visit-derived shape, met at
     different mask-relative radii by each magnitude.  The joint
     fit renders each star with its own profile (87 s per patch).
     Pass 3 (12 tracts) and an injected control on 7275 (the
     injected truth is the base wing, so their model error shows
     the term itself) run 2026-09-09; the test is whether the
     re-measured term goes to zero and the equal-weight stacks
     improve, or whether the fit absorbs it as it absorbed the
     radial correction.
     Result: absorbed.  Re-measured on pass 3 (28,971 stars) the
     term is unchanged at the mask edge (+2.9, +4.6, +7.5 percent
     for G 11.5-13, 13-14.5, 14.5-17) and reduced by a third at
     1.5-2.5 radii (G 13-14.5: +8.8 / +11.9 to +6.4 / +8.5; G
     14.5-17: +13 / +17 to +9 / +11); the equal-weight stacks are
     identical to pass 2b within errors on every tract (9812 G
     14-15.2 at 55 px 10.1 to 6.5 is the largest move, 1 sigma);
     the injected control is undamaged (injected residual within
     +-4, the model error of the bright bin from +8 to -3 as the
     amplitudes re-anchored 3 percent lower).  The amplitude fit
     re-anchors on whatever shape it is given, so the profile's
     residual-to-model ratio at the anchor is invariant to shape
     corrections; the ratio there is a data property (zero in the
     simulation), a few percent of a small wing: +3 to +5 x 10^-3
     sigma just outside the masks of G 13-17 stars, the collar at
     its current size, comparable to the injection test's own
     errors.  Stop iterating on the shape here; the acceptance
     test (metadetection on cleaned patches vs the product) decides
     whether that level matters.  If it does, the coadd's own
     stamp stack for the inner 44 px joined to the broad far wing
     is the physically motivated shape to try, not another table.
   - the injected stars' model error, +6 to +8 (a ~2 percent high
     amplitude), is the same with both segmentations; still to be
     understood (a bias of the ring-free amplitude toward the
     neighbours' light, or the pinned faint stars' prediction
     being high by the k_stamp / k_in ratio and the free
     amplitudes compensating).

7d. **Cleanup** (2026-09-09, before the shear test).  Removed as
   dead ends or superseded: the coadd-derived shape (`shape.py`),
   the magnitude term and the self-calibration scripts, the
   `sky_from`, `joint_final`, `joint_shape` and shallow-segmentation
   options, the `canonical-fit` star model, the sequential-scheme
   comparison scripts and the sky-smoothness diagnostic, the
   statistical restoration (`cell_restore`, `make_slurm_cells`, the
   polynomial cache in `coadd.py`), the response coadd
   (`cell_forward`, `ResponseCache`) and the `forward` / `restored`
   input states of the clean tool, the visit-level `remeasure` and
   `make_slurm` tools.  `lsst-starsub-cell-clean` now reads the
   delivered coadd from the butler and defaults to the joint model;
   `wing.py` is the wing reader; `forward.py` / `forward_check`
   stay as the account of the trough (the simulation uses the DM
   fit), `run_visit` and `stack` as the visit-level diagnostics.
   The old pipeline chart is kept as `docs/flow-full-forward.dot`;
   the earlier run outputs under `~/oh/starsub-visits` still read
   with the stack and comparison scripts.

7e. **Timing for on-the-fly use** (2026-09-09; the coadds cannot
   be stored, so the clean runs inside the metadetection job).
   Patch 55, one core, warm process: butler get plus object
   background restore 16 s and stitch 1 s (the metadetection job
   pays these anyway), census and star masks 1.3 s, joint fit 5.6
   s (14 s on the first call of a process: numba compiling the
   detection kernel, and the star renders), subtraction 0.  Peak
   2.3 GB with the cell coadd object held.  The mesh render was
   6.7 of the 17 s before: scipy's RegularGridInterpolator over
   10^7 pixels, replaced by a direct separable bilinear
   evaluation (0.06 s, identical to 1e-15).  So about 7 s per
   patch per band added to a metadetection run.

7f. **Wiring into lsst_mdet** (2026-09-09).  Direction: lsst_mdet
   calls lsst_starsub as a library; the old `lsst_mdet.starsub`
   route stays untouched as the reference.  `lsst_starsub/starsub.py`
   provides `handle_stars_joint(deep_coadd, wcs, gaia, wing, gsub)`
   with the contract of `lsst_mdet.starsub.handle_stars` (restores
   the stored object background, joint fit, subtracts the mesh and
   the stars in place, returns starmask / star_table / dstar) and
   `load_wing(path)`.  In lsst_mdet: `cells.load_coadds_butler` and
   `cli/getimages.prepare_band_stars` take `starsub_method`
   ('template' default, 'joint') and a `wing_pattern` with a
   `{band}` placeholder; `lsst-mdet-process-cells` / the node driver
   and `lsst-mdet-getimages` expose `--starsub-method` and
   `--wing-pattern`; the provenance records both.  lsst_mdet must be
   reinstalled (`pip install .`) for the options to exist.  Only the
   i-band wing is calibrated; the test links r and z to it (the
   shape per band is a known gap: run the per-visit template
   pooling for the other bands at S3DF).
   Pipeline test on 7275/55, cells (10,10) and (11,11), r i z with
   metadetection: joint route 79 s and 3.0 GB against the reference
   route's 83 s and 2.1 GB; the same 144 catalog rows, 47-48 of 48
   objects matched per metacal step, fluxes within 0.12-0.17 of
   their errors and shapes within 0.03-0.08 (no bright star in
   those cells, so the routes should agree).  `mdet-test/`.
   The r and z wings: `scripts/band_wing.sh BAND VISITFILE
   [MAXVISITS]` runs the per-visit extracts on slurm with the
   preemption sweeps, pools per visit and writes the canonical
   wing; launched 2026-09-09 with all 21 r visits and 24 of the 43
   z visits of tract 7275 (`07275-cells/tract-visits-{r,z}.txt`
   from the cell inputs of 8 patches).  The first launch failed on
   every detector: DP2 now serves `visit_image` as an lsst.images
   VisitImage (packed two-byte mask planes under other names, no
   getWcs / getPsf), which the loader took because the dataset
   exists; `load_visit_exposure` now falls back to the calibrated
   preliminary image unless the visit image is an afw exposure
   (the path validated on DP2 before, 1-2 nJy from the visit
   image).  Extract jobs that exit with an error are not caught
   by the preemption sweep: check `Exit status` counts, not only
   the queue.  The transfer check for r and z (the 12-tract joint
   fit per band, `broadcal/run-transfer-rz.sh`) is chained behind
   the wings.
   The r and z wings (all 8,054 extract jobs succeeded on the
   rerun): the per-visit pooled fits fail on some visits, with the
   inner slope at the scan bound (-5.0), a zero point k_in <= 0 or
   a chi2 many times the median; 7 of the 24 z visits (all six of
   2025-09-02, FWHM 1.15-1.38 arcsec) and 4 of the 21 r visits.  Left
   in, they put the visit-to-visit scatter of the z wing at 110
   percent at 100 px.  `scripts/canonical_from_templates.py` drops
   them (k_in <= 0, slope <= -4.95, chi2 > 5 x median) and is now
   the canonical step of `band_wing.sh`, which also removes the
   extracts afterwards.  Survivors: r 17 visits, scatter 13 / 28 /
   77 percent at 100 / 300 / 1000 px; z 17 visits, 14 / 22 / 51 (i
   was 10 / 8 / 46).  Against the i wing: r is 1.05 at 100 px, 0.93
   at 300, 0.83 at 1000; z 1.13, 1.06, 0.98; the bands differ by
   10-20 percent, so the per-band wings matter.  Why the pooled fit
   fails on those visits (poor seeing, the September night) is not
   understood; the templates are kept for a look.  Looked: the
   September 2 visits are bright time, sky 12,800 ADU on every
   detector against 4,300 on a good visit (visit_summary skyBg);
   the far cloud against that sky is noise and the joint fit runs
   to the bound.  A galaxy near detector 90 of 2025090200293 is a
   local feature, not the cause.  A cut on skyBg would be the
   physical criterion (`skytest/failed-visits/`: detector-90 FITS
   of three failed and one good z visit, and focal-plane mosaics
   from `scripts/visit_mosaic.py`).  The mosaic of 2025090200293
   shows stray light, not moonlight: the sky runs from 11,100 nJy
   in the north to 16,600 in the south-east corner, 40 percent
   across the field, in broad diagonal bands crossing the
   detectors (the user: a local light source in the dome); the
   good visit 2025060900547 spans 3,600-3,900, 8 percent.  The
   visit is an input to the DP2 z coadd of tract 7275 (375 of the
   484 cells of patch 0), so such exposures do reach the coadds;
   the visit summary's skyBg (12,800 against 4,300 ADU) flags them.
   Transfer check of the r and z wings (delivered-coadd joint fit
   on the 12-tract sample; 956 r and 837 z patches exist in DP2):
   real-star residual G 6-13 at 55 / 105 / 205 / 305 / 455 px within
   +-3 x 10^-3 sigma on every z tract (errors +-1-2) and on 7 of 9 r
   tracts; r shows +7 +- 3 at 105 px on 9812 and +11 +- 4 at 205 px
   on 2558 (32 stars), and 10804 is noisy in every band.  One wing
   per band transfers for r and z as it did for i.

7g. **The first shear test** (launched 2026-09-09 evening,
   `~/oh/shear-test/`).  Two full metadetection runs of the same
   243 patches (tracts 7275, 8982, 9941, the good-cells file) with
   matched per-patch seeds: `template/` the reference star route,
   `joint/` the joint fit with the calibrated r, i, z wings
   (`wings/canonical-wing-{band}.fits`).  Generated with
   `lsst-mdet-make-slurm --tracts ... --extra-args ...` (the
   generator gained `--tracts` and `--extra-args`; the joint jobs
   carry `--mem=8G`), submitted by `run-shear-test.sh`, which
   sweeps preemptions and error exits, then runs
   `lsst-mdet-make-corr-cats` (`stats/sums_config.yaml`, the
   documented example selection) and `lsst-mdet-starcorr` on each
   run.  The comparison: the tangential shear around stars by
   magnitude bin, `stats/starcorr.fits` and its pdf in each run,
   and the object counts near stars.  486 jobs, ~2 h each at most.

   Result (2026-09-10).  All 486 jobs completed in 52 min (joint
   3.0 GB per patch).  Galaxy selection (s2n > 10, T/T_psf > 0.5,
   mfrac < 0.1, g_flags 0): template 86,003, joint 86,480 (+0.55
   percent), the excess uniform in distance from G < 13 and G < 15
   stars (0-30" 256 vs 259, 30-60" 3717 vs 3800, 60-120" 17,137 vs
   17,235), so the joint route does not lose area near stars.
   Objects paired within 0.5" (83,000, 96 percent of each): the
   shear difference joint - template is consistent with zero in
   every distance bin from G < 13 stars, e.g. 30-60" +4 +- 4 x 10^-4
   in g1, 60-600" within +-1 x 10^-4 (errors 1-2 x 10^-4); the i
   flux differs by a median +0.06-0.09 percent (joint brighter, the
   collar removed) with 4-8 percent rms from the different sky and
   star models.  Whole-sample <g1> +0.0028 template vs +0.0026 joint
   (err 0.0010).  The tangential shear around stars (starcorr,
   15 bins 0.05-30', compensated with randoms, jackknife errors) is
   in each run's `stats/starcorr.fits` and pdf: both routes are
   consistent with zero in every magnitude bin (bins with < 100
   pairs dropped; the G < 16 stars have no pairs inside 0.17' because
   of the star mask): gamma_t chi2/dof template 6.1/12, 16.7/13,
   21.5/15, 15.3/15 and joint 8.3/12, 19.1/13, 19.7/15, 15.2/15 for
   G 6-16, 16-18, 18-20, 20-21; the inverse-variance mean gamma_t
   inside 1' around G < 16 stars is -0.3 +- 1.5 x 10^-3 (template)
   and -0.6 +- 1.5 x 10^-3 (joint); the two runs differ by < 1 sigma
   in every bin.  Conclusion: on 3.4 deg^2 the joint route is at
   least as good as the template route (no residual star-galaxy
   shear, 0.5 percent more galaxies, no lost area near stars); the
   test has no power below ~1.5 x 10^-3 at 0.2-1' around bright
   stars, so a survey-scale run is needed to see the collar-level
   (10^-3 sigma) differences from the residual tests.

7h. **Memory of the joint route** (2026-09-10, for the NERSC nodes:
   128 patches per node).  The shear-test patches peaked at 2.9 GB
   (joint) against 2.05 GB (template).  Profiled with tracemalloc
   and an RSS sampler (scratch `memprof_joint.py`,
   `rss_timeline.py`): both routes peak in the third band's star
   stage, on top of the loaded coadds; the joint fit added 0.97 GB
   of numpy peak over its entry, nearly all transients on the
   3300x3300 patch: the full-resolution mesh render in double
   precision (245 MB), the wing render of the whole-patch stars
   through np.mgrid (270 MB), and the pass-1 sky, star model and
   residual images kept alive in double precision across pass 2.
   Fixed in lsst_starsub only: `render_mesh` renders blocks of rows
   into a single-precision image; `render_wing_image` uses 1-d
   offsets and blocks of rows; the pass-1 residual is built in one
   f4 array and freed after the segmentation; the sparse blocks are
   dropped after stacking; the sky and star model are subtracted in
   place.  The fit's numpy peak is 0.80 GB over the baseline
   instead of 1.31; amplitudes and star mask identical, image
   differences at f4 rounding (rms 1.5 x 10^-4 nJy).  Two-cell
   pipeline peak 2.25 -> 1.94 GB, level with the template route
   (1.93).  Full patch 55 with metadetection, RSS sampled every
   0.5 s: peak 1.88 GB, but the kernel high-water mark of the
   same run was 2.68 GB (2.57 on two cells against 1.79 sampled
   at 20 ms).  Tracking ru_maxrss stage by stage found it: sep's
   pixel stack.  `deep_segmentation` set it to 2e7 entries, sep
   touches the whole stack on every extract call (41 bytes per
   entry: 0.82 GB, measured alone), the setting is process-global,
   and every later sep call in the run paid it too (the redo, the
   per-cell detections): that is the shear test's 2.9 GB.  Fixed:
   the stack starts at 2e6, grows by 4 on a 'pixel buffer full'
   overflow, and the previous settings are restored after.  Two
   cells by /usr/bin/time: 2.57 -> 1.93 GB (template 2.02); the
   full patch with metadetection 2.89 (shear test) -> 1.96 GB in
   29.5 min.  lsst_mdet's own `field_segmentation` set 1.2e7
   (0.5 GB) and left it global as well; given the same treatment
   (SEG_PIXSTACK 2e6, grown on overflow, restored).  Numba
   would take at most ~0.3 GB more out of the fit (the design
   matrix, the cell binning) and ~3 s per band of its ~9 s (sep's
   segmentation is 5 s and already C); not worth it.

7i. **The background redo with the joint route** (2026-09-10).
   lsst_mdet's `--redo-bg` ran its 64 px sep background on the
   joint-cleaned image as well (the shear test ran this
   combination; the injection, simulation and residual tests did
   not).  Measured on 7275/55 i (scratch `redo_after_joint.py`):
   the second background is +0.21 nJy (0.03 sigma) with 0.21 nJy
   rms at 64 px, flat in distance from the star masks (+0.22 at
   12-30 px, +0.21 far), so the stars were untouched; but it is
   driven by faint-source light (+0.25 within 4 px of the deep
   detections, +0.12 at 32-64 px) and the joint-cleaned empty sky
   12-64 px from any deep detection sits at -0.20 to -0.25 nJy, so
   the redo left it near -0.4 nJy (0.06 sigma per pixel under the
   galaxies): the 64 px over-subtraction the deep mask avoids,
   re-imprinted.  Fix (temporary until the routes are
   streamlined): `redo_background(..., subtract=False)` for the
   joint method in lsst_mdet cells.py and getimages, keeping the
   masks, the noise calibration and the sky-variance map for the
   weights.  Note the joint mesh itself is +0.2 nJy high in the
   empty sky for the same reason at 256 px.  On two cells of
   7275/55 the change adds 3 detections (48 -> 51) and raises the
   faint fluxes by ~40 nJy per object (7 percent at 300-1000 nJy,
   0.4 percent above 3000): the sky zero point under the galaxies
   is a real systematic at the 0.2-0.4 nJy per pixel level, and
   which level is right needs the simulation (known sky).

7j. **Output additions** (2026-09-10).  `g1g2_cov` in the catalog:
   the ml path from ngmix's g_cov; the deblend path propagates the
   full e covariance through the e -> g jacobian, with the e1-e2
   cross term added to kdeblend (`_shape_cov`, the same family
   covariance sandwich as the errors); correlation coefficients
   -0.3..+0.7 on the test cells.  The joint fit is written to four
   small tables in the catalog file (`starsub_meta` per band:
   shape, mesh geometry, settings, chi2, sky sigma, the background
   restored, the wing file; `starsub_stars`: the census with
   A_{band} and free_{band}; `starsub_sky`: one row per node per
   band; `starsub_wing`: the radial profile per band; 45 kB per
   patch).  `lsst_starsub.starsub.render_fit(tables, band)`
   rebuilds the sky and star images: identical amplitudes, images
   within f4 rounding of the direct call.  The frame: the fit is
   relative to the delivered coadd with the stored object
   background restored; the redo's noise factor is not in the
   tables (it scales the variance, not the image).

7k. **All star code in lsst-starsub** (2026-09-13).  lsst_mdet/starsub.py
   moved unchanged into `lsst_starsub.census` (census, mask circles,
   field segmentation, taper, star table, diffuse mask) and
   `lsst_starsub.stamps` (the stamp-template route, still the
   reference); lsst_mdet/gaia.py merged into `lsst_starsub.gaia`; the
   Gaia maker is now `lsst-starsub-make-gaia` (no lsst-mdet alias).
   lsst_starsub imports nothing from lsst_mdet (tests/test_independence):
   it keeps its own DM mask bits (`maskbits`), `SimpleBox`/`ButlerWcs`
   (`geom`) and a copy of metadetection's detection settings
   (`joint.DETECT_SETTINGS`), which lsst_mdet overrides by passing its
   own (`detect_settings`); lsst-mdet's tests/test_starsub_settings
   checks the copies.  The make-gaia butler defaults (NERSC's dp2) now
   live in `lsst_starsub.cli.make_gaia` as well as lsst_mdet.defaults.
   The lsst_mdet meta table gains the joint-fit and faint-wing
   settings (`joint_*`) and `version_lsst_starsub`.  Verified: 112 of
   116 moved definitions AST-identical (the 4 differ in docstrings
   and one import line); 21 reference outputs on 7275/55 identical
   before and after (process-cells both routes, getimages both
   routes, make-slurm-nersc, lsst-starsub-visit, visit-template
   extracts, cell-clean joint and template, the make-gaia file);
   both test suites pass (lsst-mdet 52, lsst-starsub 42).  Found on
   the way: `lsst-mdet-getimages --starsub-method joint` fails on the
   g band, which has no wing file (the reference used r for g).

8. **Integration and validation.**  The per-input response is
   computed once per visit-detector (~40 s on slurm) and stored as a
   small coarse array; the coadd stage sums stored arrays per cell;
   lsst-mdet reads the result.  Validate with the dual-state stack on
   the final images in all bands (only i so far), the injection test
   above, then metadetection on cleaned patches versus the current
   product (star-galaxy correlations).  Check DP2 at NERSC with
   `lsst-starsub-check-datasets`.

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
