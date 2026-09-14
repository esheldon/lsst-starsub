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
- What has to be solved that the coadd version did not face: bright
  stars off the detector whose wings cross it.  At detector scale that
  is common, and there the wing is a plane, degenerate with the sky.
  The per-visit Gaia extracts already cover the neighbors, and pinning
  off-detector stars to the prediction is what the coadd fit does at
  patch edges; at the visit level the prediction is better known.  It
  is the case to test first.

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

- **Bright stars off the detector.**  A G 8 star's wing runs 3000 px,
  across several detectors.  On a neighboring detector it is nearly a
  plane and cannot be fit; on the detector holding the star the inner
  wing fixes the amplitude to a percent.  A visit-wide view lets the one
  measurement serve every detector the wing crosses; per detector alone
  the neighbors carry a predicted wing with the shipped zero point's
  few-percent uncertainty.  The far wing is exactly where the trough
  lived, so this is the case that matters.
- **Sky continuity.**  The sky pattern (scattered light, the
  focal-plane structure skyCorr models) is continuous across detector
  boundaries; a per-detector mesh constrains its edge nodes from one
  side only.  There are also real per-detector and per-amplifier
  offsets, 0.2-0.3 nJy on the visits tested.  The physically right
  model is one continuous focal-plane sky plus a constant per detector,
  which is what skyCorr does for the backgrounds.

Why not one visit-wide fit: a visit is 3 gigapixels, 200 million rows
even on 4 x 4 cells, with tens of thousands of amplitudes and ~50,000
mesh nodes.  The normal matrix is too large for a dense solve, the
image no longer fits comfortably in memory, and the DRP's unit of work
is the detector.  Doable with a sparse iterative solver, but a different
implementation and deployment.

The practical form, following the DRP's own pattern of per-detector
tasks plus one visit-level gather (as skyCorr does):

1. Per-detector joint fits, in parallel, as now.
2. A visit-level consolidation: each star takes its amplitude from the
   detector that holds its center, where it is measured best; the wings
   crossing other detectors are re-rendered with those amplitudes.
3. A per-detector refit of the sky with the amplitudes held fixed, with
   the continuous focal-plane term and per-detector offsets in the sky
   model if the continuity turns out to matter.

This gives the off-detector stars their measured amplitudes and the sky
its continuity at per-detector cost.  Whether step 3 needs the
focal-plane term is something the per-detector residuals near detector
edges would show, and it is the first thing to measure if the project
goes ahead.

## First experiment

Possible with the tools that exist, before any new data product: the
joint fit on the visits of one patch (the per-visit loader,
`lsst_starsub.visit`, and `lsst_starsub.joint` on the detector arrays),
coadd the rendered models with the coadd's own inputs and weights, and
compare the residual stacks around the bright stars with the
coadd-level result on the same patch.  The coadd-level route stays the
one for the shear test; the visit-level version is what to build after
that test says the concept is worth it.
