"""
the joint fit of the star amplitudes and the sky (TODO step 7c)

    image = sum_k A_k F_k P(r - r_k) + S(x) + noise

with P the canonical wing (nJy per unit Gaia flux, core included),
F_k = 10^(-0.4 G_k), so A_k = 1 is the pure prediction; S a
bilinear mesh of node spacing `spacing`; solved by weighted least
squares on the pixels outside the star masks, the detections and
the bad pixels, binned BIN x BIN (the wing is smooth on that
scale, the cores are masked anyway).  Stars off the image or
fainter than gfit are pinned to the prediction (their wing on the
image is a plane, degenerate with the sky) and subtracted first.
A second pass re-segments the residual of the first, so sources
the initial flattening missed are out of the fit.

No exclusion zone and no interpolation: the toy fits showed this
reaches the noise floor at the mask edge where the sequential
sky-then-amplitude scheme leaves the collar
"""
import numpy as np

BIN = 4
SPACING = 256
GFIT = 17.0
EPS = 0.005      # nJy: each star's column extends to where it falls below
MIN_CELL_FRAC = 0.5
NPASS = 2
SEG_GROW = 4     # px: the deep segmentation's footprints are grown by this
PRIOR_SIGMA = 0.3   # the amplitude prior about the prediction (A = 1):
                    # the colour scatter of the i-band to Gaia G flux
                    # ratio; isolated stars are constrained 10x better
                    # by their pixels, close pairs, edge stars and the
                    # whole-patch wings of the brightest stars are not
                    # (3 percent of the amplitudes were negative
                    # without it, pass 1 of the broad calibration)


def deep_segmentation(image, good, sig, grow=SEG_GROW):
    """
    the metadetection detection settings (lsst_mdet.detect: the
    0.8 arcsec Gaussian kernel, threshold 0.8 in kernel-scale
    noise, minarea 4) as a mask of the sources, grown by `grow`
    px; twice the area of the 1.5 sigma per-pixel segmentation
    and 40 percent less of the faint-source light left in the
    sky (visit detectors 044 and 004, 2026-09-09)
    """
    import sep
    from scipy import ndimage
    from lsst_mdet.detect import DETECT_SETTINGS, make_kernel

    sep.set_extract_pixstack(int(2e7))
    sep.set_sub_object_limit(10240)
    kernel = make_kernel()
    imf = np.ascontiguousarray(image, dtype='f4')
    _, seg = sep.extract(
        imf, DETECT_SETTINGS['thresh'], err=sig, mask=~good,
        segmentation_map=True, filter_kernel=kernel, filter_type='conv',
        minarea=DETECT_SETTINGS['minarea'], deblend_nthresh=1,
        deblend_cont=1.0,
    )
    det = seg > 0
    if grow > 0:
        det = ndimage.binary_dilation(det, iterations=int(grow))
    return det


def binned_cells(image, ok, b=BIN):
    """
    the means of the ok pixels per b x b cell, the ok counts and
    the cell centers (pixel coordinates)
    """
    ny, nx = image.shape
    my, mx = ny // b, nx // b
    o = ok[:my * b, :mx * b].reshape(my, b, mx, b)
    img = np.where(o, image[:my * b, :mx * b].reshape(my, b, mx, b), 0.0)
    n = o.sum(axis=(1, 3))
    mean = img.sum(axis=(1, 3)) / np.maximum(n, 1)
    cy, cx = (np.mgrid[0:my, 0:mx] + 0.5) * b - 0.5
    return mean, n, cy, cx


def mesh_columns(cy, cx, shape, spacing):
    """
    bilinear hat functions on a node grid covering the image, as
    a sparse (ncell, nnode) matrix evaluated at the cell centers
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


def render_mesh(nodes, values, shape):
    """the bilinear mesh at full resolution"""
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
    # separable bilinear weights: rows blend two node rows, then
    # columns blend two node columns
    top = vals[iy]                       # (ny, nxn)
    bot = vals[iy + 1]
    rows = (1 - fy)[:, None] * top + fy[:, None] * bot
    out = (1 - fx)[None, :] * rows[:, ix] + fx[None, :] * rows[:, ix + 1]
    return out


def star_column(cy, cx, x, y, G, canonical, b=BIN, eps=EPS):
    """
    the star's wing at the cell centers within its window:
    (cell indices, values) with A = 1 the prediction; canonical
    is (r, T) or a WingModel (per-star shape)
    """
    profile_fn = getattr(canonical, 'profile', None)
    r, T = canonical if profile_fn is None else profile_fn(float(G))
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


def joint_fit(image, good, stars, canonical, sky_sigma, spacing=SPACING,
              gfit=GFIT, b=BIN, npass=NPASS, eps=EPS, variance=None,
              prior_sigma=PRIOR_SIGMA, verbose=True):
    """
    Parameters
    ----------
    image: array (nJy)
        The input state (sky in, stars in)
    good: bool array
        Pixels usable for the fit: not bad, not in a star mask
    stars: structured array
        The census (x, y, G, on_image)
    canonical: (r, T)
    sky_sigma: float
        For the segmentation threshold
    variance: array, optional
        Per-pixel variance; the cells are then weighted by their
        good-pixel count over their mean variance (uniform
        variance sky_sigma^2 otherwise)
    prior_sigma: float, optional
        Gaussian prior on each free amplitude about 1, in the
        cells' chi2 units; None for none

    Returns
    -------
    dict with A (per census star; 1 where pinned), free (bool per
    star), sky (full res), star_model (full res), nodes, node
    values, ncell, chi2 per cell
    """
    from scipy import sparse
    from .visit import render_canonical_stars

    ny, nx = image.shape
    good = good & np.isfinite(image)
    if variance is not None:
        good &= np.isfinite(variance) & (variance > 0)
    x, y, G = stars['x'], stars['y'], stars['G'].astype('f8')
    on = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    free = on & (G < gfit)
    nfree = int(free.sum())

    # free stars need cells to be fit on: those whose window holds
    # no good cell (inside masks or no-data regions) are pinned as
    # well, or the normal matrix is singular
    mean0, n0, cy, cx = binned_cells(image, good, b)
    has_cells = (n0 >= MIN_CELL_FRAC * b * b)
    for si in np.flatnonzero(free):
        idx, v = star_column(cy, cx, float(x[si]), float(y[si]), float(G[si]),
                             canonical, b=b, eps=eps)
        if idx.size == 0 or not has_cells.ravel()[idx].any():
            free[si] = False
    nfree = int(free.sum())

    # the pinned stars' prediction, subtracted from the data
    pinned = stars[~free]
    fixed = render_canonical_stars(image.shape, pinned, canonical, gsub=99.0) \
        if pinned.size else np.zeros(image.shape, dtype='f4')
    work = image - fixed

    # the columns of the free stars at the cell centers
    ncell = cy.size
    rows, cols, vals = [], [], []
    for k, si in enumerate(np.flatnonzero(free)):
        idx, v = star_column(cy, cx, float(x[si]), float(y[si]), float(G[si]),
                             canonical, b=b, eps=eps)
        rows.append(idx)
        cols.append(np.full(idx.size, k))
        vals.append(v)
    P = sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ncell, nfree),
    )
    H, nodes = mesh_columns(cy, cx, image.shape, spacing)
    X = sparse.hstack([P, H]).tocsr()

    # first flattening for the segmentation: the mesh alone on
    # the good pixels
    seg_excl = np.zeros(image.shape, dtype=bool)
    A = np.ones(stars.size)
    for ipass in range(npass):
        ok = good & ~seg_excl
        mean, n, _, _ = binned_cells(work, ok, b)
        w = (n >= MIN_CELL_FRAC * b * b).ravel() * n.ravel().astype('f8')
        if variance is not None:
            vmean, _, _, _ = binned_cells(variance, ok, b)
            w = w * sky_sigma ** 2 / np.maximum(vmean.ravel(), 1e-12)
        W = sparse.diags(w)
        F = (X.T @ W @ X).toarray()
        rhs = X.T @ (w * mean.ravel())
        # a tiny ridge on the mesh keeps unsupported nodes finite
        F[nfree:, nfree:] += np.eye(F.shape[0] - nfree) * 1e-6 * w.sum() / ncell
        # and on the amplitudes, at a level far below any star's
        # own information (a star losing its cells to the pass-2
        # segmentation would otherwise make F singular)
        F[:nfree, :nfree] += np.eye(nfree) * 1e-9 * np.diag(F)[:nfree].max()
        if prior_sigma is not None:
            # the prior in the data term's units: F = X^T W X with
            # w the pixel counts (over the sky_sigma^2-relative
            # variance), so F is chi2 x sky_sigma^2 and the prior
            # 1/prior_sigma^2 enters times sky_sigma^2
            pw = sky_sigma ** 2 / prior_sigma ** 2
            F[:nfree, :nfree] += np.eye(nfree) * pw
            rhs[:nfree] += pw
        sol = np.linalg.solve(F, rhs)
        A[free] = sol[:nfree]
        node_values = sol[nfree:]
        model_cells = X @ sol
        resid_cells = (mean.ravel() - model_cells) * (w > 0)
        # each cell's variance is sky_sigma^2 / n, so w resid^2 /
        # sigma^2 is its chi2
        chi2 = float(np.sum(w * resid_cells ** 2) / sky_sigma ** 2
                     / max(1, (w > 0).sum()))
        if verbose:
            print(f'    joint fit pass {ipass + 1}: {nfree} amplitudes, '
                  f'{node_values.size} nodes ({spacing} px), '
                  f'{int((w > 0).sum())} cells, chi2/cell {chi2:.3f}; '
                  f'A of the brightest: '
                  + ' '.join(f'{A[si]:.3f}' for si in np.argsort(G)[:5]
                             if free[si]))
        if ipass == npass - 1:
            break
        # re-segment the full-resolution residual
        sky_full = render_mesh(nodes, node_values, image.shape)
        model_full = render_canonical_stars(
            image.shape, stars, canonical, gsub=99.0, amps=A, verbose=False,
        )
        resid = image - sky_full - model_full
        seg_excl = deep_segmentation(resid, good, sky_sigma)
        if verbose:
            print(f'    segmentation excludes {seg_excl[good].mean() * 100:.1f} '
                  f'percent of the good pixels')

    sky_full = render_mesh(nodes, node_values, image.shape).astype('f4')
    model_full = render_canonical_stars(
        image.shape, stars, canonical, gsub=99.0, amps=A, verbose=False,
    )
    return dict(
        A=A, free=free, sky=sky_full, star_model=model_full, nodes=nodes,
        node_values=node_values, ncell=ncell, chi2=chi2,
    )
