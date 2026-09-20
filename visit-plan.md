# Star and sky models at the visit level: a plan

A discussion from 2026-09-09, while the r and z wing calibrations ran.
The question: instead of fixing the coadds for lensing by subtracting
stars and redoing the sky at the patch level, do it on the visits and
save new per-visit data products that serve this and other purposes.

## What the coadd-level work says about it

Three results from the injection tests, the simulation and the broad
calibration bear directly on a visit-level version.

- **The trough is made at the visit level**, by the background
  polynomial of calibrateImage absorbing the star wings.  Fitting stars
  and sky together on each detector before any background removes it at
  the source; nothing downstream has to model it.  The forward model
  showed the polynomial's response is exactly that absorption, and the
  simulation showed the joint fit leaves nothing at the 10^-3 sigma
  level when the wing shape is right.
- **The wing is right at the visit level.**  The residual structure the
  coadd fit could not remove, a magnitude-dependent plateau of a few
  percent at 1-2 mask radii, came from the coadd PSF's inner wing being
  a seeing mixture that the visit-median shape does not describe.  On a
  visit the stamps and the wing belong to the same PSF; the per-visit
  pooling measures the shape and zero point per visit to a few percent.
  Per-visit fit errors are independent across the coadd's inputs and
  average down; a coadd-level fit's error is coherent.
- **The cost is small.**  The joint fit on a 4k detector is about 10 s
  (6 s for a 3300 px patch, warm process); 190 detectors times a few
  thousand visits is a few thousand core-hours.  The per-visit wing
  pooling as done here is heavier, ~3 core-hours per visit (one job
  per detector), but the canonical shape with a seeing term can replace
  it: the pooled fits showed the inner amplitude tracks the FWHM.

## The product

Disk rules out new images, so the product is the model, not pixels.

- Per detector: a table of star amplitudes (relative to the prediction
  from the Gaia flux and the wing they refer to) and the sky mesh, a
  few hundred numbers.  This is the same kind of object as the stored
  backgrounds the pipeline keeps and applies at warp time.  Any
  consumer renders image - sky - stars on the fly: the warp step for
  coadds, the cell reader, difference imaging, anyone wanting a clean
  visit.
- The right home is the background step of calibrateImage: replace the
  polynomial fit with the joint fit, Gaia stars in, and every DRP
  product is trough-free with the star model as a standard layer
  beside the background.  This is the highest-leverage version and the
  one to propose to DM; everything else is a workaround around that
  step.
- What has to be solved that the coadd version did not face: stars
  whose center is on a neighboring detector or in a gap, with part of
  their inner wing on this one.  At detector scale that is common
  (about 30 percent of a detector's stars are within 300 px of an
  edge), and it is the case the passes below are built for.  The far
  wings of stars well off the detector are the lesser problem: nearly
  planes, absorbed by the mesh, and pinning those stars to the
  prediction is what the coadd fit does at patch edges.  The per-visit
  Gaia extracts already cover the neighbors.

## The sky mesh as it stands

A bilinear mesh: nodes every 256 px from the image origin, one extra
row and column past the far edge (14 x 14 = 196 nodes on a 3300 px
patch, ~17 x 17 on a 4k detector), each node a hat function, 1 at the
node falling linearly to 0 at the neighboring nodes.  The hats are
columns of the same linear system as the star wings, evaluated at the
centers of the 4 x 4 cells the fit works on, weighted by each cell's
good-pixel count over its variance; one sparse normal-equation solve
gives node values and amplitudes together, with a tiny ridge on the
nodes so a node with no cells under it stays finite.  Rendering is a
separable bilinear evaluation, 0.06 s per patch.

The 256 px spacing is about the scale where the far wing is flat enough
for the mesh to absorb it (so the trough goes into the sky term) and
coarse enough that a wing's curvature within a few hundred pixels of
the star stays in the star term; 128 px gave the same residuals in the
simulation.  For a visit-level product the node values plus spacing
and origin reproduce the sky anywhere.  A bicubic or B-spline basis on
the same nodes is a drop-in change if a smoother sky is wanted; the
kinks of the bilinear form have not shown in any residual.

## Detector or full visit

Not a need, but two real advantages to the visit-wide view, and both
can be had without one giant fit.

- **Stars across a detector edge.**  A G 8 star's wing runs 3000 px,
  across several detectors, but the far wing is not the problem: for a
  star 500 px away the 256 px mesh absorbs all but ~3 percent of its
  wing, so a wrong amplitude there only books the light to the sky,
  which is subtracted anyway.  That is why the coadd fit gets away with
  pinning off-patch stars to the prediction.  The case that matters is
  the star whose center is on a neighboring detector or in a gap with
  part of its inner wing on this one: there the wing is steep, the mesh
  cannot absorb it, and this detector's data cannot constrain the
  amplitude (the core and most of the inner wing are elsewhere), so the
  fit pins it to the Gaia prediction, which scatters by ~30 percent
  with the star's color.  On a 4k detector the stars within 300 px of
  an edge are ~30 percent of all its stars.
- **Sky continuity.**  The sky pattern (scattered light, the
  focal-plane structure skyCorr models) is continuous across detector
  boundaries; a per-detector mesh constrains its edge nodes from one
  side only.  There are also real per-detector and per-amplifier
  offsets, 0.2-0.3 nJy on the visits tested.  The physically right
  model is one continuous focal-plane sky plus low-order terms per
  chip (constants per amplifier, perhaps a gradient per chip), which is
  what skyCorr does for the backgrounds.  The vignetting of the outer
  detectors is a physical reason to expect the per-chip terms: the sky
  and dome flats disagree most where the illumination is steepest.
- **Vignetting and the wings.**  The flat takes out the mean
  throughput, so sky levels and star fluxes come out right, but nothing
  corrects the wing shape: a partly obscured pupil changes the
  scattering, so the wing fraction, and possibly the inner shape,
  differ at the field edge from the center.  The calibrated r, i, z
  wings are a focal-plane average (pooled over all detectors of the
  tract 7275 visits); the outer detectors may sit systematically off
  it.

Why not one visit-wide fit: a visit is 3 gigapixels, 200 million rows
even on 4 x 4 cells, with tens of thousands of amplitudes and ~50,000
mesh nodes.  The normal matrix is too large for a dense solve, the
image no longer fits comfortably in memory, and the DRP's unit of work
is the detector.  Doable with a sparse iterative solver, but a different
implementation and deployment.

## The passes (2026-09-14)

Following the DRP's own pattern of per-detector tasks plus one
visit-level gather (as skyCorr does).  Keep it simple at first and let
the measurements say what more is needed: the heavily vignetted
detectors, centers beyond 300 mm from the focal-plane axis (23 of 178
on the test visit), are left out until the wing has a field-radius
term (`visit.exposure.MAX_FIELD_RADIUS`, `lsst-starsub-visit-detectors`).

1. **Pass 1, per detector, in parallel.**  The joint fit as now, but
   with every star whose inner wing touches the detector free (within
   ~3 mask radii of the edge), not only the stars centered on it, and
   with the amplitude errors returned from the normal matrix beside the
   node errors.  Stars farther off the detector stay pinned.
2. **Consolidation, one visit-level gather.**  Each star takes the
   amplitude from the detector that constrains it best, by the
   amplitude error, not from the detector holding its center: that
   covers the gap case, where two detectors hold halves of the inner
   wing, and falls back to the prediction when no detector constrains a
   star.  Two things come almost free here:
   - the visit's own amplitude scale, the median amplitude of the
     well-constrained stars (the pooled fits showed the inner-wing
     amplitude tracks the seeing); with the vignetting in mind it may
     have to be per detector or a smooth function of field radius, and
     the prior for the poorly constrained stars then pulls toward the
     local scale rather than the survey's;
   - a cross-detector check: a star near an edge that two detectors
     both constrain gives two independent amplitudes, and their
     agreement tests the shipped wing shape and zero point per visit,
     and shows whether the outer detectors need their own scale or
     shape.
3. **Pass 2, per detector.**  A sky-only refit with every amplitude
   fixed: subtract the star model, solve the mesh (no star columns, one
   small solve, a second or two).  Needed regardless of the edge stars,
   because the pass-1 sky absorbed the prediction errors of every
   pinned star.  The mesh nodes along a shared edge then come out from
   both sides; their disagreement, against the 0.2-0.3 nJy per-amplifier
   offsets, says whether the visit-level sky with its per-chip terms is
   worth building.  Left out of the first version.

The product per detector: the star table with amplitude, error and the
detector that constrained it, and the node values with spacing and
origin; per visit, the scale.  The consolidation makes it
self-consistent, one amplitude per star per visit, which a consumer
rendering image - sky - stars on several detectors needs.  Cost: pass 1
~10 s per detector, pass 2 a couple of seconds, the gather trivial;
~40 core-minutes per visit.

Two things to keep in view.  The amplitude absorbs the inner wing's
scale but not its shape, and the inner shape is what the seeing
changes (the far aureole is scattering and stable); a per-visit inner
shape from a stamp stack of the G 15.5-17.5 stars across the focal
plane would be a cheap pass 0, if the cross-detector check says it is
needed.  And a star's color term is the same in every visit of a band,
so stars always near an edge or in a gap, which never get a good
per-visit amplitude, would get one from all their visits together: a
later refinement, once per-visit products exist to gather.

## First experiment

Possible with the tools that exist, before any new data product.  One
visit, every detector through `handle_stars_visit` with the joint
model, then two measurements: the amplitude agreement for stars near
edges that two detectors both constrain, and the residual stacks around
stars whose center is off-detector, with their pinned prediction versus
the neighbor's amplitude.  The second number is the gain of the whole
scheme.  It needs pass 1 to fit the edge stars and to return amplitude
errors, both small changes to `joint_fit` and the visit wrapper, and a
gather script.

Then, as planned before: the joint fit on the visits of one patch,
the rendered models coadded with the coadd's own inputs and weights,
and the residual stacks around the bright stars compared with the
coadd-level result on the same patch.  The coadd-level route stays the
one for the shear test; the visit-level version is what to build after
that test says the concept is worth it.

## Open items the coadd-level galaxy and star masks leave for this route (2026-09-19)

Recorded so they are not lost when the visit route replaces the
coadd joint fit in production.  Both come from
`lsst_starsub.galaxies` and the G < 6 output mask
(`census.add_disk_masks`), which live in the coadd joint route only;
the visit-level products are untouched by them.

1. **The large-galaxy mask needs a segmentation.**  `lsst-mdet
   --galaxy-file` sizes each catalog galaxy's mask from the joint
   fit's last segmentation (`joint_fit` result `big_sources`, kept
   by `handle_stars_joint`) and refuses `--starsub-method visit`,
   which applies the correction coadds and runs no fit.  When that
   route is used for a run, run `joint.deep_segmentation(...,
   return_big=True)` on the corrected coadd in
   `handle_stars_correction` and hand the table on in its fit dict,
   so `galaxy_mask` works unchanged.  Cheap: one segmentation per
   band.

2. **The catalog galaxies should be kept out of the visit-level sky
   determination.**  The coadd joint fit excludes each HyperLEDA
   galaxy's D25 ellipse x `galaxies.GAL_SKY_SCALE` from the sky mesh
   (`galaxies.catalog_exclusion`); without it a galaxy larger than
   the fit's own capped source exclusion is fit as sky and leaves a
   dark halo (IC 5078, 3.5', in 06804-00062; the DM object background
   has the same halo).  The visit passes have the same exposure to it
   and can use the same catalog and the same function on the
   visit-frame positions.

The G < 6 star output mask is coadd-only by design (the fit keeps the
450 px circle and the ghost-disk term; growing the fit's hole to the
disk made the sky mesh run away, 2026-09-19) and needs nothing from
this route.
