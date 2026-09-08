"""
restoring the visit-level backgrounds at the coadd level

The deep coadd is a weighted mean of direct warps of the
preliminary visit images, each of which had its own stored
background (a 6 x 6 Chebyshev polynomial plus constants,
preliminary_visit_image_background) subtracted; the sky
correction is not applied for the deep coadd (see visit.py).
Since every subtracted model is stored and smooth, the
background the coadd lost is itself a coadd,

    B(x) = sum_v w_v m_v(x) P_v(x) / sum_v w_v m_v(x)

with the same per-visit weights w_v the coadd used, m_v the
warp's valid-pixel mask, and P_v the visit's polynomial (of
whichever detector covers x) evaluated at the coadd pixel
through the detector WCS.  Adding B to the coadd's None state
(deep_coadd with the object background restored) gives the
coadd of the raw sky images: the state in which the star
wings are intact and the sky is the true sky.

The polynomials are smooth, so each is sampled on a coarse
coadd-pixel grid and interpolated; the warp masks are applied
at full resolution.  The per-detector WCS and photometric
calibration come from visit_summary, which is what the warp
task used (useVisitSummaryWcs / useVisitSummaryPhotoCalib)
"""
import numpy as np

from .visit import INSTRUMENT, SKYMAP, render_background_list

# coarse sampling step of the polynomials, in coadd pixels
SAMPLE_STEP = 16


def coadd_data_id(tract, patch, band):
    return dict(band=band, skymap=SKYMAP, tract=tract, patch=patch)


def load_cell_coadd(butler, did):
    """
    the None-state cell coadd as a MultipleCellCoadd:
    deep_coadd_cell_predetection where present (the weekly
    runs), else DP2's deep_coadd, a lsst.images CellCoadd with
    the object background subtracted, restored
    (apply_background(None)) and converted to the legacy class
    (per-cell inputs with weights, stitch, grid, wcs)
    """
    try:
        have = bool(butler.exists('deep_coadd_cell_predetection', did))
    except Exception:
        have = False
    if have:
        return butler.get('deep_coadd_cell_predetection', dataId=did)
    cc = butler.get('deep_coadd', dataId=did)
    if not hasattr(cc, 'to_legacy_cell_coadd'):
        raise RuntimeError(
            'deep_coadd is not a cell coadd and '
            'deep_coadd_cell_predetection is absent'
        )
    print('    deep_coadd is a CellCoadd: restoring its object '
          'background and converting to the legacy class')
    cc.apply_background(None)
    return cc.to_legacy_cell_coadd()


def load_coadd(butler, tract, patch, band):
    """
    the deep coadd (object background subtracted, as delivered),
    its object background rendered, and the input ccd table as
    a dict (visit, detector) -> weight (zero-weight inputs
    dropped)

    Returns
    -------
    coadd, object_bg, weights
    """
    did = coadd_data_id(tract, patch, band)
    coadd = butler.get('deep_coadd', dataId=did)
    obj = butler.get('deep_coadd_background', dataId=did)
    object_bg = obj.getImage().array.astype('f4')

    # every input ccd is kept for the polynomial sampling; the
    # zero-weight ones only touch the warp's 256 px border
    # outside the coadd, and the visit weight comes from the
    # others
    ccds = coadd.getInfo().getCoaddInputs().ccds
    weights = {}
    for rec in ccds:
        weights[(int(rec['visit']), int(rec['ccd']))] = float(rec['weight'])
    nzero = sum(1 for w in weights.values() if w == 0)
    print(
        f'    coadd {tract} {patch} {band}: {len(ccds)} inputs, '
        f'{nzero} with zero weight'
    )
    return coadd, object_bg, weights


def coarse_grid(bbox, step=SAMPLE_STEP):
    """
    the coarse sampling grid over the coadd bbox: 1-d tract-frame
    x and y sample coordinates (including the far edge) and the
    full-resolution index arrays for the interpolation
    """
    x0, y0 = bbox.getBeginX(), bbox.getBeginY()
    nx, ny = bbox.getWidth(), bbox.getHeight()
    xs = np.arange(0, nx + step, step, dtype='f8')
    ys = np.arange(0, ny + step, step, dtype='f8')
    xs = xs[xs <= nx - 1]
    ys = ys[ys <= ny - 1]
    if xs[-1] < nx - 1:
        xs = np.append(xs, nx - 1)
    if ys[-1] < ny - 1:
        ys = np.append(ys, ny - 1)
    return xs + x0, ys + y0


def sample_polynomial(bglist, calib, det_wcs, det_bbox, ra, dec):
    """
    the stored background of one detector (nJy) at the given
    sky positions, NaN outside the detector; bilinear in the
    rendered model

    Parameters
    ----------
    bglist: BackgroundList
        The stored initial background of the detector
    calib: float
        ADU -> nJy
    det_wcs: SkyWcs
        The detector WCS (visit_summary)
    det_bbox: Box2I
        The detector bbox
    ra, dec: arrays
        Sky positions, degrees
    """
    from scipy.ndimage import map_coordinates

    model = render_background_list(bglist, range(len(bglist)), calib)
    x, y = det_wcs.skyToPixelArray(ra, dec, degrees=True)
    x = x - det_bbox.getBeginX()
    y = y - det_bbox.getBeginY()
    ny, nx = model.shape
    inside = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
    out = np.full(ra.shape, np.nan, dtype='f8')
    if inside.any():
        out[inside] = map_coordinates(
            model.astype('f8'), [y[inside], x[inside]], order=1,
            mode='nearest',
        )
    return out


def polynomial_coadd(butler, coadd, weights, tract, patch, band,
                     step=SAMPLE_STEP):
    """
    the weighted mean of the visits' stored backgrounds over the
    coadd, with each visit's warp mask (finite, not NO_DATA)
    applied at full resolution

    Parameters
    ----------
    butler: Butler
    coadd: ExposureF
        The deep coadd
    weights: dict
        (visit, detector) -> weight from load_coadd
    tract, patch, band
    step: int, optional
        Coarse sampling step in coadd pixels

    Returns
    -------
    bcoadd, wsum, count:
        The background coadd (nJy, NaN where no input), the
        weight sum, and the number of contributing visits per
        pixel
    """
    from scipy.ndimage import map_coordinates

    bbox = coadd.getBBox()
    wcs = coadd.getWcs()
    ny, nx = coadd.image.array.shape
    xs, ys = coarse_grid(bbox, step)
    gx, gy = np.meshgrid(xs, ys)
    ra, dec = wcs.pixelToSkyArray(gx.ravel(), gy.ravel(), degrees=True)

    # full-resolution coordinates in coarse-grid index units
    fy, fx = np.mgrid[0:ny, 0:nx]
    cx = np.interp(fx.ravel() + bbox.getBeginX(), xs, np.arange(xs.size))
    cy = np.interp(fy.ravel() + bbox.getBeginY(), ys, np.arange(ys.size))
    coords = [cy.reshape(ny, nx), cx.reshape(ny, nx)]

    byvisit = {}
    for (visit, det), w in weights.items():
        byvisit.setdefault(visit, []).append((det, w))

    num = np.zeros((ny, nx), dtype='f8')
    wsum = np.zeros((ny, nx), dtype='f8')
    count = np.zeros((ny, nx), dtype='i4')
    nodata_bit = None

    for k, (visit, dets) in enumerate(sorted(byvisit.items())):
        vs = butler.get(
            'visit_summary',
            dataId=dict(instrument=INSTRUMENT, visit=visit),
        )
        # the visit's polynomial on the coarse grid: each grid
        # point belongs to at most one detector
        coarse = np.full(ra.shape, np.nan)
        for det, w in dets:
            rec = vs.find(det)
            if rec is None:
                print(f'    visit {visit} det {det}: no summary record')
                continue
            calib = float(rec.getPhotoCalib().getCalibrationMean())
            bglist = butler.get(
                'preliminary_visit_image_background',
                dataId=dict(instrument=INSTRUMENT, visit=visit,
                            detector=det),
            )
            samp = sample_polynomial(
                bglist, calib, rec.getWcs(), rec.getBBox(), ra, dec,
            )
            fill = np.isfinite(samp) & ~np.isfinite(coarse)
            coarse[fill] = samp[fill]

        # the visit weight: the coadd combines whole warps, so
        # one weight per visit (the ccd values differ by a
        # percent or so; zero means the ccd only touches the
        # warp border outside the coadd)
        wpos = [w for _, w in dets if w > 0]
        if len(wpos) == 0:
            print(f'    visit {visit}: no weight, skipped')
            continue
        w = float(np.mean(wpos))

        warp = butler.get(
            'direct_warp',
            dataId=dict(band=band, skymap=SKYMAP, tract=tract,
                        patch=patch, visit=visit, instrument=INSTRUMENT),
        )
        if nodata_bit is None:
            # pixels the coadd did not use: no data, or rejected
            # by the artifact clipping
            nodata_bit = 0
            for plane in ('NO_DATA', 'CLIPPED', 'REJECTED'):
                try:
                    nodata_bit |= warp.mask.getPlaneBitMask(plane)
                except Exception:
                    pass
        # the warp carries a border beyond the coadd bbox
        wb = warp.getBBox()
        oy = bbox.getBeginY() - wb.getBeginY()
        ox = bbox.getBeginX() - wb.getBeginX()
        wimg = warp.image.array[oy:oy + ny, ox:ox + nx]
        wmask = warp.mask.array[oy:oy + ny, ox:ox + nx]
        valid = np.isfinite(wimg) & ((wmask & nodata_bit) == 0)

        # interpolate the coarse polynomial to full resolution;
        # grid points off every detector are NaN and must not
        # leak: fill them from the nearest finite values first,
        # the warp mask then limits the contribution
        cg = coarse.reshape(gy.shape)
        if not np.isfinite(cg).any():
            print(f'    visit {visit}: no polynomial coverage')
            continue
        cg = _fill_nan_nearest(cg)
        full = map_coordinates(cg, coords, order=1, mode='nearest')

        num[valid] += w * full[valid]
        wsum[valid] += w
        count[valid] += 1
        if (k + 1) % 10 == 0:
            print(f'    {k + 1} of {len(byvisit)} visits')

    bcoadd = np.full((ny, nx), np.nan, dtype='f4')
    ok = wsum > 0
    bcoadd[ok] = (num[ok] / wsum[ok]).astype('f4')
    print(
        f'    background coadd: {ok.mean():.3f} covered, '
        f'median {np.nanmedian(bcoadd):.1f} nJy, '
        f'median count {np.median(count[ok]):.0f}'
    )
    return bcoadd, wsum.astype('f4'), count


# ---------------------------------------------------------------
# the cell coadd route: the cell is the building block, every
# input covers its whole cell, so the background coadd is exact
# per cell with the cell's own input weights
# ---------------------------------------------------------------

# the stored polynomials are smooth over the detector; a
# rendering downsampled by this factor is interpolated bilinearly
COARSE_FACTOR = 8
# sampling step within a cell, in coadd pixels
CELL_STEP = 10
# a polynomial's detector-scale part (its level and gradients,
# the visit's sky and flat residuals) is removed by a surface fit
# of this order before the restoration: it carries no star
# information, and its per-cell weighted mean steps at the cell
# boundaries as the input set changes.  What remains are the
# ~600 px bumps around bright stars, anchored on the sky.  0
# removes the mean only
SMOOTH_ORDER = 2


def fit_smooth_surface(model, order):
    """
    least-squares polynomial surface of the given total order
    over the model's grid, returned evaluated on the grid
    """
    ny, nx = model.shape
    y, x = np.mgrid[0:ny, 0:nx]
    # scaled to [-1, 1] for conditioning
    xs = 2.0 * x / max(nx - 1, 1) - 1.0
    ys = 2.0 * y / max(ny - 1, 1) - 1.0
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((xs ** i * ys ** j).ravel())
    basis = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(basis, model.ravel(), rcond=None)
    return (basis @ coef).reshape(ny, nx)


class PolynomialCache(object):
    """
    per (visit, detector) coarse renderings of the stored
    initial background (nJy) with the detector WCS and bbox from
    visit_summary, loaded on first use
    """

    def __init__(self, butler, factor=COARSE_FACTOR, order=SMOOTH_ORDER):
        self.butler = butler
        self.factor = factor
        self.order = order
        # per visit: detector -> (wcs, bbox, calib).  The summary
        # catalog itself is not kept: it carries the PSF models
        # of every detector, tens of MB per visit
        self._geometry = {}
        self._models = {}

    def geometry(self, visit, detector):
        visit = int(visit)
        if visit not in self._geometry:
            cat = self.butler.get(
                'visit_summary',
                dataId=dict(instrument=INSTRUMENT, visit=visit),
            )
            geo = {}
            for rec in cat:
                pc = rec.getPhotoCalib()
                geo[int(rec['id'])] = (
                    rec.getWcs(), rec.getBBox(),
                    float(pc.getCalibrationMean()) if pc else None,
                )
            self._geometry[visit] = geo
            del cat
        return self._geometry[visit].get(int(detector))

    def get(self, visit, detector):
        """
        (coarse model, wcs, bbox) for one input; None when the
        visit summary has no record for the detector
        """
        key = (int(visit), int(detector))
        if key not in self._models:
            geo = self.geometry(visit, detector)
            if geo is None or geo[2] is None:
                self._models[key] = None
            else:
                wcs, bbox, calib = geo
                bglist = self.butler.get(
                    'preliminary_visit_image_background',
                    dataId=dict(instrument=INSTRUMENT, visit=int(visit),
                                detector=int(detector)),
                )
                full = render_background_list(
                    bglist, range(len(bglist)), calib,
                )
                coarse = np.ascontiguousarray(
                    full[::self.factor, ::self.factor], dtype='f8',
                )
                del full, bglist
                # the detector-scale part (level and gradients)
                # comes off: it is the visit's sky, not star
                # information, and its per-cell weighted mean
                # steps at the cell boundaries as the input set
                # changes.  The stored model is the remainder,
                # the star bumps; level is the removed mean, for
                # the record
                level = float(np.mean(coarse))
                smooth = fit_smooth_surface(coarse, self.order)
                coarse = coarse - smooth
                self._models[key] = (coarse, wcs, bbox, level)
        return self._models[key]

    def level(self, visit, detector):
        entry = self.get(visit, detector)
        return None if entry is None else entry[3]

    def _spawn(self, butler):
        """a fresh cache of the same kind on another butler"""
        return PolynomialCache(butler, factor=self.factor, order=self.order)

    def prefetch_parallel(self, keys, repo, collection, nproc):
        """
        prefetch with nproc forked workers, each with its own
        butler (the butler traffic is the bottleneck, and it is
        i/o bound)
        """
        import multiprocessing as mp

        keys = sorted(set((int(v), int(d)) for v, d in keys))
        keys = [k for k in keys if k not in self._models]
        if len(keys) == 0:
            return
        nvis = len(set(v for v, _ in keys))
        print(
            f'    prefetching {len(keys)} inputs from {nvis} visits '
            f'with {nproc} processes'
        )
        nchunk = min(len(keys), nproc * 4)
        chunks = [keys[i::nchunk] for i in range(nchunk)]
        global _WORKER_PROTO
        _WORKER_PROTO = self
        ctx = mp.get_context('fork')
        done = 0
        with ctx.Pool(nproc) as pool:
            for out in pool.imap_unordered(
                _prefetch_worker,
                [(repo, collection, c) for c in chunks],
            ):
                for key, entry, stats in out:
                    self._models[key] = entry
                    if stats is not None:
                        getattr(self, 'stats', {})[key] = stats
                done += len(out)
                print(f'    {done} of {len(keys)} inputs loaded')
        _WORKER_PROTO = None

    def prefetch(self, keys):
        """load every (visit, detector) in keys, with progress"""
        keys = sorted(set((int(v), int(d)) for v, d in keys))
        nvis = len(set(v for v, _ in keys))
        print(f'    prefetching {len(keys)} inputs from {nvis} visits')
        for i, (v, d) in enumerate(keys):
            self.get(v, d)
            if (i + 1) % 25 == 0:
                print(f'    {i + 1} of {len(keys)} inputs loaded')

    def sample(self, visit, detector, ra, dec):
        """
        the input's background (nJy) at sky positions; NaN
        outside the detector
        """
        from scipy.ndimage import map_coordinates

        entry = self.get(visit, detector)
        out = np.full(ra.shape, np.nan)
        if entry is None:
            return out
        coarse, wcs, bbox, _ = entry
        x, y = wcs.skyToPixelArray(ra, dec, degrees=True)
        x = (x - bbox.getBeginX()) / self.factor
        y = (y - bbox.getBeginY()) / self.factor
        ny, nx = coarse.shape
        inside = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
        if inside.any():
            out[inside] = map_coordinates(
                coarse, [y[inside], x[inside]], order=1, mode='nearest',
            )
        return out


def polynomial_cell_coadd(butler, mcoadd, step=CELL_STEP,
                          factor=COARSE_FACTOR, repo=None,
                          collection=None, nproc=1, cache=None):
    """
    the background the cell coadd lost, cell by cell: for each
    cell the weighted mean over its inputs of their stored
    polynomials, with the cell's input weights, sampled on a
    step grid over the cell and interpolated

    Parameters
    ----------
    butler: Butler
    mcoadd: MultipleCellCoadd
        The predetection cell coadd
    step: int, optional
        Sampling step within a cell (pixels)
    factor: int, optional
        Downsampling of the rendered polynomials
    cache: PolynomialCache, optional
        A prebuilt cache (e.g. a trough.ResponseCache) instead of
        the stored polynomials

    Returns
    -------
    bcoadd, blevel, wsum, count:
        Patch-frame arrays over the stitched (inner) bbox: the
        background coadd (nJy), the per-cell weighted mean of
        the inputs' detector-mean levels (the sky patchwork,
        constant within a cell; bcoadd - blevel is the
        structure to restore), the weight sum per cell and the
        input count per cell
    """
    if cache is None:
        cache = PolynomialCache(butler, factor=factor)
    bb = mcoadd.inner_bbox
    ny, nx = bb.getHeight(), bb.getWidth()

    bcoadd = np.full((ny, nx), np.nan, dtype='f4')
    blevel = np.full((ny, nx), np.nan, dtype='f4')
    wsum = np.zeros((ny, nx), dtype='f4')
    count = np.zeros((ny, nx), dtype='i4')

    cells = mcoadd.cells
    ncell = len(cells)
    keys = [
        (ident.visit, ident.detector)
        for cell in cells.values() for ident in cell.inputs
    ]
    # all the butler traffic up front: this is the slow, i/o-bound
    # part, a registry query and a file read per input
    if nproc > 1 and repo is not None:
        cache.prefetch_parallel(keys, repo, collection, nproc)
    else:
        cache.prefetch(keys)

    indices = list(cells.keys())
    if nproc > 1:
        # forked workers inherit the cache and the coadd
        # copy-on-write; each returns its cells' arrays
        import multiprocessing as mp
        global _WORKER_CACHE, _WORKER_MCOADD, _WORKER_STEP
        _WORKER_CACHE, _WORKER_MCOADD, _WORKER_STEP = cache, mcoadd, step
        chunks = [indices[i::nproc * 4] for i in range(nproc * 4)]
        chunks = [c for c in chunks if c]
        ctx = mp.get_context('fork')
        done = 0
        with ctx.Pool(nproc) as pool:
            for results in pool.imap_unordered(_cell_worker, chunks):
                for res in results:
                    _store_cell(res, bb, bcoadd, blevel, wsum, count)
                done += len(results)
                print(f'    {done} of {ncell} cells')
        _WORKER_CACHE = _WORKER_MCOADD = None
    else:
        for k, index in enumerate(indices):
            res = _compute_cell(cache, mcoadd, index, step)
            _store_cell(res, bb, bcoadd, blevel, wsum, count)
            if (k + 1) % 50 == 0:
                print(f'    {k + 1} of {ncell} cells')

    ok = np.isfinite(bcoadd)
    print(
        f'    cell background coadd: {ok.mean():.3f} covered, '
        f'{nproc} processes, '
        f'structure rms {np.nanstd(bcoadd[ok]):.3f} nJy '
        f'(removed level median {np.nanmedian(blevel):.1f}), '
        f'median input count {np.median(count[ok]):.0f}, '
        f'{len(cache._models)} polynomials rendered'
    )
    return bcoadd, blevel, wsum, count


# worker state for the forked cell workers (set before the
# pool starts, inherited copy-on-write)
_WORKER_CACHE = None
_WORKER_MCOADD = None
_WORKER_STEP = CELL_STEP
_WORKER_BUTLER = None
_WORKER_PROTO = None


def _compute_cell(cache, mcoadd, index, step):
    """
    one cell's background: the weighted mean over its inputs of
    their polynomials on the cell's coarse grid, interpolated to
    full resolution.  Returns (bbox tuple, full, level, wtot, n)
    or None when no input has weight
    """
    from scipy.ndimage import map_coordinates

    cell = mcoadd.cells[index]
    wcs = mcoadd.wcs
    cb = cell.inner.bbox
    cx0, cy0 = cb.getBeginX(), cb.getBeginY()
    cnx, cny = cb.getWidth(), cb.getHeight()
    xs = np.unique(np.append(np.arange(0, cnx, step), cnx - 1))
    ys = np.unique(np.append(np.arange(0, cny, step), cny - 1))
    gx, gy = np.meshgrid(xs + cx0, ys + cy0)
    ra, dec = wcs.pixelToSkyArray(
        gx.ravel().astype('f8'), gy.ravel().astype('f8'), degrees=True,
    )

    num = np.zeros(ra.shape)
    wtot = 0.0
    lev = 0.0
    n = 0
    for ident, inp in cell.inputs.items():
        w = float(inp.weight)
        if not w > 0:
            continue
        samp = cache.sample(ident.visit, ident.detector, ra, dec)
        level = cache.level(ident.visit, ident.detector)
        if level is None:
            continue
        if not np.isfinite(samp).all():
            # an input is supposed to cover the whole cell; fill
            # any grid point outside its detector from the
            # covered ones rather than drop it
            bad = ~np.isfinite(samp)
            if bad.all():
                print(f'    cell {index}: input {ident.visit} '
                      f'{ident.detector} has no coverage, skipped')
                continue
            samp = samp.copy()
            samp[bad] = np.nanmedian(samp)
        num += w * samp
        lev += w * level
        wtot += w
        n += 1
    if wtot <= 0:
        return None
    coarse = (num / wtot).reshape(gy.shape)
    fy, fx = np.mgrid[0:cny, 0:cnx]
    cy = np.interp(fy.ravel(), ys, np.arange(ys.size))
    cx = np.interp(fx.ravel(), xs, np.arange(xs.size))
    full = map_coordinates(
        coarse, [cy, cx], order=1, mode='nearest',
    ).reshape(cny, cnx).astype('f4')
    return (cx0, cy0, cnx, cny), full, lev / wtot, wtot, n


def _store_cell(res, bb, bcoadd, blevel, wsum, count):
    if res is None:
        return
    (cx0, cy0, cnx, cny), full, level, wtot, n = res
    sl = np.s_[
        cy0 - bb.getBeginY():cy0 - bb.getBeginY() + cny,
        cx0 - bb.getBeginX():cx0 - bb.getBeginX() + cnx,
    ]
    bcoadd[sl] = full
    blevel[sl] = level
    wsum[sl] = wtot
    count[sl] = n


def _cell_worker(indices):
    return [
        _compute_cell(_WORKER_CACHE, _WORKER_MCOADD, index, _WORKER_STEP)
        for index in indices
    ]


def _prefetch_worker(args):
    """
    load a chunk of inputs in a worker with its own butler and a
    cache spawned from the prototype set by prefetch_parallel;
    returns [(key, entry, stats), ...]
    """
    global _WORKER_BUTLER
    repo, collection, keys = args
    if _WORKER_BUTLER is None:
        from lsst.daf.butler import Butler
        _WORKER_BUTLER = Butler(repo, collections=[collection])
    cache = _WORKER_PROTO._spawn(_WORKER_BUTLER)
    out = []
    for v, d in keys:
        key = (int(v), int(d))
        try:
            entry = cache.get(v, d)
        except Exception as err:
            print(f'    input {v} {d} FAILED: {err!r}')
            entry = None
        stats = getattr(cache, 'stats', {}).get(key)
        out.append((key, entry, stats))
    return out


def _fill_nan_nearest(arr):
    """NaNs replaced by the nearest finite value"""
    from scipy.ndimage import distance_transform_edt

    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr
    idx = distance_transform_edt(bad, return_distances=False,
                                 return_indices=True)
    return arr[tuple(idx)]
