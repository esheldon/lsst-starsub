"""
The joint fit of the star amplitudes and the sky.

    image = sum_k A_k F_k P(r - r_k) + S(x) + noise

with P the canonical wing (nJy per unit Gaia flux, core included),
F_k = 10^(-0.4 G_k), so A_k = 1 is the pure prediction; S a
bilinear mesh of node spacing `spacing`; solved by weighted least
squares on the pixels outside the star masks, the detections and
the bad pixels, binned BIN x BIN (the wing is smooth on that
scale, the cores are masked anyway).  Stars off the image or
fainter than gfit are pinned to the prediction (their wing on the
image is a plane, degenerate with the sky) and subtracted first.
The first pass fits without a source mask; the second fits
outside the sources segmented on the image minus the first pass's
star model.  The first pass's sky is deliberately not subtracted
before segmenting: under a bright galaxy it rises with the
galaxy's outer light, which would then fall below threshold and
stay in the sky fit, digging a dark halo (09813-00033, 2026-09-11).
The star model is subtracted so the wings outside the star masks,
which constrain the amplitudes, are not masked.

No exclusion zone and no interpolation: the toy fits showed this
reaches the noise floor at the mask edge where the sequential
sky-then-amplitude scheme leaves the collar
"""

import numpy as np

# the source segmentation's detection settings: metadetection's own
# (lsst_mdet.detect.DETECT_SETTINGS, whose threshold, kernel and minimum
# area these copy), so the sky fit masks what metadetection will detect.
# lsst_mdet passes its dict in (detect_settings); a test there checks
# the copy.  thresh in units of the kernel-scale noise, the gaussian
# kernel's fwhm in arcsec at pixel_scale arcsec per px, minarea in px
DETECT_SETTINGS = dict(
    thresh=0.8,
    kernel_fwhm=0.8,
    pixel_scale=0.2,
    minarea=4,
)

BIN = 4
SPACING = 256
GFIT = 17.0
EPS = 0.005  # nJy: each star's column extends to where it falls below
MIN_CELL_FRAC = 0.5
NPASS = 2
SEG_GROW = 4  # px: the deep segmentation's footprints are grown by this
# large sources also get an elliptical mask: a big galaxy's wing beyond
# its isophote still lifts the sky mesh under it (a dark halo).  The
# ellipse reaches the isophotal radius plus SEG_BIG_K rms sizes (sep's
# a).  For an exponential profile the wing falls from the threshold t to
# eps over ln(t/eps) scale lengths, K ~ 2; heavier wings need more, so K
# is calibrated on the halo sim (2026-09-11: K = 4 above 3000 px removes
# ~60% of the halo under bright ellipticals for +7 percentage points of
# masked area; smaller area thresholds also grow ordinary galaxies).
# Read at call time
SEG_BIG_NPIX = 3000  # px: isophotal area from which the growth applies
SEG_BIG_K = 4.0
# px: cap on the ellipse's semi-major axis.  650 reaches the giant
# elliptical of 09224-00089 (617 px uncapped); with 300 its envelope
# stayed in the sky fit (a dark ring; 3x the halo of 650 in the sim).
# A cap this large needs the mesh prior: with the ridge alone the
# masked nodes ran away (2026-09-11)
SEG_BIG_RMAX = 650.0
# the sky mesh can be tied together: a penalty (v_i - v_j)^2 /
# (MESH_SMOOTH_DELTA sky_sigma)^2 on each pair of neighboring node
# values, so a node with little data of its own (under a large source
# or star mask, where the cells left sit at the edge of its hat) follows
# its neighbors instead of extrapolating from those cells.  A constant
# sky costs nothing, so the level is not pulled (the ridge pulls to
# zero).  Neighboring well-supported nodes differ by 0.08-0.11 sky
# sigma (8 patches, 2026-09-11).  0.1 fixes the ridge's outlier nodes
# under large star masks (04777-00069: -4.5 sigma, 10x its error) and
# leaves the star residuals as they were; stiffer (0.03) loses real
# structure (halo sim, 2026-09-11).  None: the ridge alone.  Read at
# call time
MESH_SMOOTH_DELTA = 0.1
# large segments of diffuse emission (cirrus): a segment of at least
# SEG_BIG_NPIX px whose median pixel value is below SEG_DIFFUSE_MEDIAN
# sky sigma is left out of the source mask, so the mesh fits it as sky,
# and is not grown; the compact sources inside it are found again above
# a local background of SEG_DIFFUSE_BW px boxes, which takes out the
# diffuse light, and stay masked (sep merges everything connected, so
# without this every galaxy on cirrus would lift the mesh).  Cirrus
# medians are 1.2-1.4, galaxies >= 2.0 sky sigma (the large r-band
# segments of 37 patches, 2026-09-12).  Of the treatments in the cirrus
# injection test (run-dp2-test-cirrus-inject, 2026-09-12) this gave the
# best colors on and near the cirrus, no excess detections near it and
# the shortest run times; masked and grown, its faint surroundings stay
# in the image.  Faint objects on the cirrus are still biased (-0.11 in
# r-i at S/N 10-20), so joint_fit returns the region for masking.
# None: off, every segment is a source.  Read at call time
SEG_DIFFUSE_MEDIAN = 1.6
SEG_DIFFUSE_BW = 32  # px
RENDER_BLOCK = 256  # rows per block when rendering the mesh
# the amplitude prior about the prediction (A = 1): the color
# scatter of the i-band to Gaia G flux ratio; isolated stars are
# constrained 10x better by their pixels, close pairs, edge stars
# and the whole-patch wings of the brightest stars are not (3
# percent of the amplitudes were negative without it, pass 1 of
# the broad calibration)
PRIOR_SIGMA = 0.3


def make_kernel(detect_settings=None):
    """
    Make the detection kernel, a 7x7 gaussian of the settings' fwhm.

    As lsst_mdet.detect.make_kernel.

    Parameters
    ----------
    detect_settings: dict, optional
        With kernel_fwhm (arcsec) and pixel_scale (arcsec per px);
        default DETECT_SETTINGS

    Returns
    -------
    kernel: array
        The 7x7 kernel
    """
    import ngmix

    s = DETECT_SETTINGS if detect_settings is None else detect_settings
    fwhm = s['kernel_fwhm'] / s['pixel_scale']  # pixels
    T = ngmix.moments.fwhm_to_T(fwhm)

    kernel_gm = ngmix.GMixModel(
        pars=[0.0, 0.0, 0.0, 0.0, T, 1.0],
        model='gauss',
    )

    return kernel_gm.make_image([7, 7])


def deep_segmentation(
    image,
    good,
    sig,
    grow=SEG_GROW,
    return_diffuse=False,
    detect_settings=None
):
    """
    Segment the sources with the metadetection detection settings.

    The metadetection detection settings (DETECT_SETTINGS: the
    0.8 arcsec Gaussian kernel, threshold 0.8 in kernel-scale
    noise, minarea 4) as a mask of the sources, grown by `grow`
    px; twice the area of the 1.5 sigma per-pixel segmentation
    and 40 percent less of the faint-source light left in the
    sky (visit detectors 044 and 004, 2026-09-09).  Large sources
    are further masked out to an ellipse set by their size
    (grow_big_sources); large diffuse segments are left to the sky
    fit when SEG_DIFFUSE_MEDIAN is set (diffuse_segments).

    Parameters
    ----------
    image: array
        The image to segment
    good: bool array
        The usable pixels; the rest are masked in the extraction
    sig: float
        The per-pixel sky noise (sep's err)
    grow: int, optional
        Px by which the source footprints are grown; default SEG_GROW
    return_diffuse: bool, optional
        Also return the mask of the diffuse segments
    detect_settings: dict, optional
        The detection settings, thresh, kernel_fwhm, pixel_scale and
        minarea; default DETECT_SETTINGS

    Returns
    -------
    det: bool array
        The source mask
    region: bool array
        With return_diffuse only: the diffuse segments left to the
        sky fit (none when SEG_DIFFUSE_MEDIAN is None)
    """
    from scipy import ndimage

    imf = np.ascontiguousarray(image, dtype='f4')

    objs, seg = _extract(imf, sig, ~good, detect_settings=detect_settings)
    det = seg > 0
    diffuse = diffuse_segments(imf, seg, objs, sig)
    region = np.zeros(det.shape, dtype=bool)

    if diffuse.size:
        region = np.isin(seg, diffuse)

        det &= ~region

        det |= compact_in_diffuse(
            imf, good & region, sig, detect_settings=detect_settings
        )

        objs = np.delete(objs, diffuse - 1)

        print(
            f'    {diffuse.size} diffuse segments left to the sky fit '
            f'({region[good].mean() * 100:.1f} percent of the good '
            f'pixels)'
        )

    if grow > 0:
        det = ndimage.binary_dilation(det, iterations=int(grow))

    grow_big_sources(det, objs)

    if return_diffuse:
        return det, region

    return det


def _extract(imf, sig, mask, detect_settings=None):
    """
    Run sep.extract with the metadetection detection settings.

    No deblending; the pixel stack is managed by census.sep_extract.

    Parameters
    ----------
    imf: array
        The image, contiguous float32
    sig: float
        The per-pixel sky noise (sep's err)
    mask: bool array
        True for the pixels to ignore
    detect_settings: dict, optional
        The detection settings, thresh, kernel_fwhm, pixel_scale and
        minarea; default DETECT_SETTINGS

    Returns
    -------
    objs: structured array
        The sep object table
    seg: int array
        The segmentation map: label k for objs[k - 1], 0 for none
    """
    from .census import sep_extract

    s = DETECT_SETTINGS if detect_settings is None else detect_settings
    return sep_extract(
        imf,
        s['thresh'],
        sig,
        mask,
        filter_kernel=make_kernel(s),
        filter_type='conv',
        minarea=s['minarea'],
        deblend_nthresh=1,
        deblend_cont=1.0,
    )


def diffuse_segments(imf, seg, objs, sig):
    """
    Find the labels of the large segments of diffuse emission.

    Segments of at least SEG_BIG_NPIX px whose median pixel value is
    below SEG_DIFFUSE_MEDIAN sky sigma; none when that is None.

    Parameters
    ----------
    imf: array
        The segmented image
    seg: int array
        Its segmentation map (_extract)
    objs: structured array
        Its sep object table (_extract)
    sig: float
        The per-pixel sky noise

    Returns
    -------
    ids: int array
        The segmentation labels of the diffuse segments
    """

    if SEG_DIFFUSE_MEDIAN is None:
        return np.zeros(0, dtype=int)

    ids = np.flatnonzero(objs['npix'] >= SEG_BIG_NPIX) + 1
    if ids.size == 0:
        return ids

    # the pixels of each large segment, sorted by label once
    sel = np.isin(seg, ids)
    lab, val = seg[sel], imf[sel]
    order = np.argsort(lab, kind='stable')
    lab, val = lab[order], val[order]

    bounds = np.searchsorted(lab, np.append(ids, ids[-1] + 1))

    med = np.array(
        [np.median(val[bounds[k] : bounds[k + 1]]) for k in range(ids.size)]
    )

    return ids[med < SEG_DIFFUSE_MEDIAN * sig]


def compact_in_diffuse(imf, region, sig, detect_settings=None):
    """
    Find the compact sources inside the diffuse segments.

    Detected with the same settings above a local background of
    SEG_DIFFUSE_BW px boxes estimated on the region, which takes out
    the diffuse light.

    Parameters
    ----------
    imf: array
        The segmented image
    region: bool array
        The usable pixels of the diffuse segments, to which the
        search is limited
    sig: float
        The per-pixel sky noise
    detect_settings: dict, optional
        The detection settings; default DETECT_SETTINGS

    Returns
    -------
    compact: bool array
        The mask of the compact sources
    """
    import sep

    bkg = sep.Background(
        imf, mask=~region, bw=SEG_DIFFUSE_BW, bh=SEG_DIFFUSE_BW, fw=3, fh=3
    )

    resid = np.ascontiguousarray(imf - bkg.back(), dtype='f4')
    _, seg = _extract(resid, sig, ~region, detect_settings=detect_settings)

    return seg > 0


def grow_big_sources(det, objs):
    """
    Mask an ellipse around each large source, in place.

    For each source of isophotal area at least SEG_BIG_NPIX: the
    orientation and axis ratio from its moments, reaching its
    isophotal radius plus SEG_BIG_K rms sizes along the major axis,
    capped at SEG_BIG_RMAX.  Nothing when SEG_BIG_K <= 0.

    Parameters
    ----------
    det: bool array
        The source mask, modified in place
    objs: structured array
        The sep.extract object table
    """
    import sep

    big = objs[objs['npix'] >= SEG_BIG_NPIX]

    if big.size == 0 or SEG_BIG_K <= 0:
        return

    a = np.maximum(big['a'], 1.0)
    b = np.maximum(big['b'], 1.0)

    riso = np.sqrt(big['npix'] / np.pi)
    scale = np.minimum(riso / a + SEG_BIG_K, SEG_BIG_RMAX / a)

    sep.mask_ellipse(det, big['x'], big['y'], a, b, big['theta'], r=scale)


def binned_cells(image, ok, b=BIN):
    """
    Bin an image into b x b cells over the usable pixels.

    Rows and columns beyond the last whole cell are dropped.

    Parameters
    ----------
    image: array
        The image
    ok: bool array
        The pixels to use
    b: int, optional
        The cell side in px; default BIN

    Returns
    -------
    mean: array (my, mx)
        The mean of the ok pixels per cell, 0 where there are none
    n: int array (my, mx)
        The number of ok pixels per cell
    cy, cx: arrays (my, mx)
        The cell centers, pixel coordinates
    """

    ny, nx = image.shape
    my, mx = ny // b, nx // b

    o = ok[: my * b, : mx * b].reshape(my, b, mx, b)

    img = np.where(o, image[: my * b, : mx * b].reshape(my, b, mx, b), 0.0)

    n = o.sum(axis=(1, 3))
    mean = img.sum(axis=(1, 3)) / np.maximum(n, 1)

    cy, cx = (np.mgrid[0:my, 0:mx] + 0.5) * b - 0.5

    return mean, n, cy, cx


def mesh_columns(cy, cx, shape, spacing):
    """
    Build the bilinear sky-mesh columns at the cell centers.

    Bilinear hat functions on a node grid covering the image,
    evaluated at the cell centers.

    Parameters
    ----------
    cy, cx: arrays
        The cell centers (binned_cells)
    shape: (ny, nx)
        The image shape
    spacing: float
        The node spacing in px; the nodes run from 0 to at least the
        image size

    Returns
    -------
    H: sparse matrix (ncell, nnode)
        The hat functions at the cell centers, the cells in the ravel
        order of cy, cx
    nodes: (xn, yn)
        The node coordinates; node k is at (xn[k % xn.size],
        yn[k // xn.size])
    """
    from scipy import sparse

    ny, nx = shape

    xn = np.arange(0, nx + spacing, spacing, dtype='f8')
    yn = np.arange(0, ny + spacing, spacing, dtype='f8')

    nxn, nyn = xn.size, yn.size
    cxf, cyf = cx.ravel(), cy.ravel()

    ix = np.clip((cxf // spacing).astype(int), 0, nxn - 2)
    iy = np.clip((cyf // spacing).astype(int), 0, nyn - 2)

    fx = (cxf - xn[ix]) / spacing
    fy = (cyf - yn[iy]) / spacing

    rows, cols, vals = [], [], []
    ncell = cxf.size

    idx = np.arange(ncell)

    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            rows.append(idx)
            cols.append((iy + dy) * nxn + ix + dx)
            vals.append(wy * wx)

    H = sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ncell, nxn * nyn),
    )

    return H, (xn, yn)


def mesh_difference_matrix(nodes):
    """
    Build the first differences of neighboring mesh nodes.

    One row per horizontally or vertically adjacent node pair: +1 at
    one node, -1 at the other.

    Parameters
    ----------
    nodes: (xn, yn)
        The node coordinates (mesh_columns)

    Returns
    -------
    D: sparse matrix (npair, nnode)
        The difference operator, in the node order of mesh_columns
    """
    from scipy import sparse

    xn, yn = nodes

    idx = np.arange(xn.size * yn.size).reshape(yn.size, xn.size)

    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])

    rows = np.arange(a.size)

    return sparse.csr_matrix(
        (
            np.concatenate([np.ones(a.size), -np.ones(a.size)]),
            (np.concatenate([rows, rows]), np.concatenate([a, b])),
        ),
        shape=(a.size, idx.size),
    )


def render_mesh(nodes, values, shape, block=RENDER_BLOCK):
    """
    Render the bilinear mesh at full resolution.

    Separable bilinear weights: each row blends two node rows,
    then the columns blend two node columns.  Rendered in blocks
    of rows into a single-precision image, so the transient
    double-precision arrays are a block, not the image.

    Parameters
    ----------
    nodes: (xn, yn)
        The node coordinates (mesh_columns), evenly spaced
    values: array
        The node values, in the node order of mesh_columns
    shape: (ny, nx)
        The image shape
    block: int, optional
        Rows per block; default RENDER_BLOCK

    Returns
    -------
    sky: array (ny, nx) f4
        The rendered mesh
    """
    xn, yn = nodes

    vals = np.asarray(values, dtype='f8').reshape(yn.size, xn.size)

    ny, nx = shape
    spacing = float(xn[1] - xn[0])

    xs = np.arange(nx, dtype='f8')
    ys = np.arange(ny, dtype='f8')

    ix = np.clip((xs // spacing).astype(int), 0, xn.size - 2)
    iy = np.clip((ys // spacing).astype(int), 0, yn.size - 2)

    fx = (xs - xn[ix]) / spacing
    fy = (ys - yn[iy]) / spacing

    wx0 = (1 - fx)[None, :]
    wx1 = fx[None, :]

    out = np.empty((ny, nx), dtype='f4')

    for y0 in range(0, ny, block):
        y1 = min(ny, y0 + block)
        fyb = fy[y0:y1, None]
        rows = (1 - fyb) * vals[iy[y0:y1]] + fyb * vals[iy[y0:y1] + 1]
        out[y0:y1] = wx0 * rows[:, ix] + wx1 * rows[:, ix + 1]

    return out


def star_column(cy, cx, x, y, G, canonical, b=BIN, eps=EPS):
    """
    Evaluate one star's wing at the cell centers of its window.

    The window reaches the radius where the wing falls below eps;
    the values are for A = 1, the prediction.

    Parameters
    ----------
    cy, cx: arrays (my, mx)
        The cell centers (binned_cells)
    x, y: float
        The star's position, pixels
    G: float
        Its Gaia G magnitude
    canonical: (r, T) or WingModel
        The wing, nJy per unit Gaia flux at radius r px; a WingModel
        gives a per-star shape (its profile(G))
    b: int, optional
        The cell side in px; default BIN
    eps: float, optional
        nJy: the window's edge; default EPS

    Returns
    -------
    idx: int array
        The cells with a nonzero value, flat indices into cy, cx
    vals: array
        The wing at those cells
    """
    from .wing import profile_of

    r, T = profile_of(canonical, float(G))

    flux = 10.0 ** (-0.4 * G)
    prof = flux * T

    below = np.flatnonzero(prof < eps)

    rmax = float(r[below[0]]) if below.size else float(r[-1])

    my, mx = cy.shape

    i0 = max(0, int((y - rmax) // b) - 1)
    i1 = min(my, int((y + rmax) // b) + 2)
    j0 = max(0, int((x - rmax) // b) - 1)
    j1 = min(mx, int((x + rmax) // b) + 2)

    if i1 <= i0 or j1 <= j0:
        return np.zeros(0, dtype=int), np.zeros(0)

    sub_y = cy[i0:i1, j0:j1]
    sub_x = cx[i0:i1, j0:j1]

    rr = np.hypot(sub_y - y, sub_x - x)
    vals = np.interp(rr, r, prof, right=0.0)

    w = vals > 0

    ii, jj = np.nonzero(w)

    idx = (ii + i0) * mx + (jj + j0)

    return idx, vals[w]


def joint_fit(
    image,
    good,
    stars,
    canonical,
    sky_sigma,
    spacing=SPACING,
    gfit=GFIT,
    b=BIN,
    npass=NPASS,
    eps=EPS,
    variance=None,
    prior_sigma=PRIOR_SIGMA,
    detect_settings=None,
    free_margin=None,
    amps=None,
    free=None,
    verbose=True,
):
    """
    Fit the star amplitudes and the sky mesh together.

    Weighted least squares on the good pixels binned b x b (see the
    module docstring), in npass passes: the first without a source
    mask, each later one outside the sources segmented
    (deep_segmentation) on the image minus the previous pass's star
    model.

    Parameters
    ----------
    image: array (nJy)
        The input state: stars in, and the sky the mesh is to model
        not yet subtracted (for a coadd, the image with only the
        background determined without object masking subtracted;
        see starsub.handle_stars_joint)
    good: bool array
        Pixels usable for the fit: not bad, not in a star mask
    stars: structured array
        The census, with x, y (pixels) and G per star
    canonical: (r, T) or WingModel
        The wing (star_column)
    sky_sigma: float
        The per-pixel sky noise: the segmentation threshold, and the
        unit of the cell weights, the priors and node_err
    spacing: float, optional
        The mesh node spacing in px; default SPACING
    gfit: float, optional
        Stars on the image brighter than this G are fit, the rest
        pinned to the prediction; default GFIT
    b: int, optional
        The cell side in px; default BIN
    npass: int, optional
        The number of passes; default NPASS
    eps: float, optional
        nJy: each star's column extends to where its wing falls
        below this; default EPS
    variance: array, optional
        Per-pixel variance; the cells are then weighted by their
        good-pixel count over their mean variance (uniform
        variance sky_sigma^2 otherwise)
    prior_sigma: float, optional
        Gaussian prior on each free amplitude about 1, in the
        cells' chi2 units; None for none.  Default PRIOR_SIGMA
    detect_settings: dict, optional
        The detection settings of the source segmentation
        (deep_segmentation); default DETECT_SETTINGS
    free_margin: float or array, optional
        Stars brighter than gfit whose center is within this many px
        outside the image are fit as well, one value or one per
        census star (e.g. a multiple of the mask radius, so the
        stars across a detector edge with inner wing on the image
        get their own amplitude); default 0, on-image stars only
    amps: array, optional
        Per census star, the amplitude the pinned stars take and the
        center of the free stars' prior; default 1, the prediction.
        With gfit below every star this is a sky-only fit at fixed
        amplitudes
    free: bool array, optional
        Per census star, fit its amplitude; replaces the gfit and
        free_margin selection (a star without cells is still pinned)
    verbose: bool, optional
        Print the per-pass summaries

    Returns
    -------
    result: dict
        A (per census star; the amps value where pinned), A_err (1
        sigma of the free amplitudes, priors included; nan where
        pinned), free (bool per star), sky (full res), star_model
        (full res), nodes, node_values,
        node_err (their 1 sigma, priors included), ncell, chi2 (per
        cell), diffuse (full res bool: the large diffuse segments the
        last segmentation left to the sky fit; none with
        SEG_DIFFUSE_MEDIAN None or a single pass)
    """
    from scipy import sparse
    from .wing import render_canonical_stars

    ny, nx = image.shape

    good = good & np.isfinite(image)

    if variance is not None:
        good &= np.isfinite(variance) & (variance > 0)

    x, y, G = stars['x'], stars['y'], stars['G'].astype('f8')

    # the free stars: bright enough, and on the image or within the
    # margin of it
    m = np.zeros(stars.size) if free_margin is None else \
        np.broadcast_to(np.asarray(free_margin, dtype='f8'), (stars.size,))
    near = (x >= -m) & (x < nx + m) & (y >= -m) & (y < ny + m)

    if free is None:
        free = near & (G < gfit)
    else:
        free = np.array(free, dtype=bool)
        if free.shape != (stars.size,):
            raise ValueError('free must have one value per census star')
    nfree = int(free.sum())

    # the pinned amplitudes and the prior centers
    if amps is None:
        prior_center = np.ones(stars.size)
    else:
        prior_center = np.array(amps, dtype='f8')
        if prior_center.shape != (stars.size,):
            raise ValueError('amps must have one value per census star')

    # free stars need cells to be fit on: those whose window holds
    # no good cell (inside masks or no-data regions) are pinned as
    # well, or the normal matrix is singular

    mean0, n0, cy, cx = binned_cells(image, good, b)
    has_cells = n0 >= MIN_CELL_FRAC * b * b

    for si in np.flatnonzero(free):
        idx, v = star_column(
            cy,
            cx,
            float(x[si]),
            float(y[si]),
            float(G[si]),
            canonical,
            b=b,
            eps=eps,
        )

        if idx.size == 0 or not has_cells.ravel()[idx].any():
            free[si] = False
    nfree = int(free.sum())

    # the pinned stars' model, subtracted from the data
    pinned = stars[~free]
    if pinned.size:
        work = render_canonical_stars(
            image.shape,
            pinned,
            canonical,
            amps=prior_center[~free],
        )
        np.subtract(image, work, out=work)
    else:
        work = image.astype('f4')

    # the columns of the free stars at the cell centers
    ncell = cy.size
    rows, cols, vals = [], [], []
    for k, si in enumerate(np.flatnonzero(free)):
        idx, v = star_column(
            cy,
            cx,
            float(x[si]),
            float(y[si]),
            float(G[si]),
            canonical,
            b=b,
            eps=eps,
        )
        rows.append(idx)
        cols.append(np.full(idx.size, k))
        vals.append(v)

    if nfree > 0:
        P = sparse.csc_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(ncell, nfree),
        )
    else:
        # a sky-only fit (every star pinned)
        P = sparse.csc_matrix((ncell, 0))

    H, nodes = mesh_columns(cy, cx, image.shape, spacing)
    X = sparse.hstack([P, H]).tocsr()

    del P, H, rows, cols, vals

    # the smoothness prior on the mesh (MESH_SMOOTH_DELTA) in the
    # units of F, chi2 x sky_sigma^2: 1 / delta^2 per pair
    smooth = None
    if MESH_SMOOTH_DELTA is not None:
        D = mesh_difference_matrix(nodes)
        smooth = (D.T @ D).toarray() / MESH_SMOOTH_DELTA**2

    # first flattening for the segmentation: the mesh alone on
    # the good pixels
    seg_excl = np.zeros(image.shape, dtype=bool)
    diffuse = np.zeros(image.shape, dtype=bool)
    A = prior_center.copy()

    for ipass in range(npass):
        ok = good & ~seg_excl
        mean, n, _, _ = binned_cells(work, ok, b)

        w = (n >= MIN_CELL_FRAC * b * b).ravel() * n.ravel().astype('f8')

        if variance is not None:
            vmean, _, _, _ = binned_cells(variance, ok, b)
            w = w * sky_sigma**2 / np.maximum(vmean.ravel(), 1e-12)

        W = sparse.diags(w)
        F = (X.T @ W @ X).toarray()
        rhs = X.T @ (w * mean.ravel())

        # a tiny ridge on the mesh keeps unsupported nodes finite
        F[nfree:, nfree:] += (
            np.eye(F.shape[0] - nfree) * 1e-6 * w.sum() / ncell
        )

        if smooth is not None:
            F[nfree:, nfree:] += smooth

        # and on the amplitudes, at a level far below any star's
        # own information (a star losing its cells to the pass-2
        # segmentation would otherwise make F singular)

        if nfree > 0:
            F[:nfree, :nfree] += (
                np.eye(nfree) * 1e-9 * np.diag(F)[:nfree].max()
            )

        if prior_sigma is not None:
            # the prior in the data term's units: F = X^T W X with
            # w the pixel counts (over the sky_sigma^2-relative
            # variance), so F is chi2 x sky_sigma^2 and the prior
            # 1/prior_sigma^2 enters times sky_sigma^2
            pw = sky_sigma**2 / prior_sigma**2
            F[:nfree, :nfree] += np.eye(nfree) * pw
            rhs[:nfree] += pw * prior_center[free]

        sol = np.linalg.solve(F, rhs)
        A[free] = sol[:nfree]
        node_values = sol[nfree:]
        model_cells = X @ sol
        resid_cells = (mean.ravel() - model_cells) * (w > 0)

        # each cell's variance is sky_sigma^2 / n, so w resid^2 /
        # sigma^2 is its chi2
        chi2 = float(
            np.sum(w * resid_cells**2) / sky_sigma**2 / max(1, (w > 0).sum())
        )

        if verbose:
            print(
                f'    joint fit pass {ipass + 1}: {nfree} amplitudes, '
                f'{node_values.size} nodes ({spacing} px), '
                f'{int((w > 0).sum())} cells, chi2/cell {chi2:.3f}; '
                f'A of the brightest: '
                + ' '.join(
                    f'{A[si]:.3f}' for si in np.argsort(G)[:5] if free[si]
                )
            )

        if ipass == npass - 1:
            break

        # segment the full-resolution image minus the star model,
        # but not minus this pass's sky (see the module docstring)
        resid = render_canonical_stars(
            image.shape,
            stars,
            canonical,
            amps=A,
            verbose=False,
        )

        np.subtract(image, resid, out=resid)

        seg_excl, diffuse = deep_segmentation(
            resid,
            good,
            sky_sigma,
            return_diffuse=True,
            detect_settings=detect_settings,
        )

        del resid

        if verbose:
            print(
                f'    segmentation excludes '
                f'{seg_excl[good].mean() * 100:.1f} percent of the '
                f'good pixels'
            )

    del work

    # the node uncertainties, sky_sigma^2 F^-1 with the priors in
    # (white noise: the coadd's correlated noise makes them lower
    # bounds)

    cov = np.diag(np.linalg.inv(F))
    node_err = sky_sigma * np.sqrt(np.maximum(cov[nfree:], 0.0))
    A_err = np.full(stars.size, np.nan)
    A_err[free] = sky_sigma * np.sqrt(np.maximum(cov[:nfree], 0.0))
    sky_full = render_mesh(nodes, node_values, image.shape)

    model_full = render_canonical_stars(
        image.shape,
        stars,
        canonical,
        amps=A,
        verbose=False,
    )

    return dict(
        A=A,
        A_err=A_err,
        free=free,
        sky=sky_full,
        star_model=model_full,
        nodes=nodes,
        node_values=node_values,
        node_err=node_err,
        ncell=ncell,
        chi2=chi2,
        diffuse=diffuse,
    )
