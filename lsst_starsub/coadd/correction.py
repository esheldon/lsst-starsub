"""
The correction coadd: the per-visit models carried into a patch coadd.

A clean visit is the delivered image plus a correction: the pipeline's
initial background put back, minus our sky model, minus our star
model.  The coadd is a weighted mean of the warped inputs, so the coadd
of the clean visits is the delivered coadd plus the coadd of the
corrections, each input's correction evaluated at the coadd's pixels
through the two WCSs and weighted as the CellCoadd weighted that
input.  The models are smooth or analytic (visit.product), so this
needs no resampling: the correction at a coadd pixel is the model at
the pixel's position in the detector.

Inputs without a product (a visit or detector the scheme did not run
on) contribute zero: their pixels stay as the pipeline delivered them,
and the weight fraction corrected is recorded per pixel.
"""
import numpy as np

from ..site import INSTRUMENT

# the initial background is a 6 x 6 Chebyshev plus constants: sampled
# at this stride and interpolated it is exact to well below 0.1 nJy
BG_STRIDE = 8


class InputModel:
    """
    One coadd input's models, ready to evaluate at detector positions.

    Attributes
    ----------
    product: visit.product.Product
    wing: WingModel
    wcs: lsst.afw.geom.SkyWcs
    nx, ny: int
    bg: array
        The initial background (nJy) at every BG_STRIDE-th pixel
    """

    def correction_at(self, x, y):
        """
        The correction at detector positions: background - sky - stars.

        Parameters
        ----------
        x, y: arrays
            Positions in the detector's px

        Returns
        -------
        corr: array, nJy
        """
        from scipy import ndimage
        from ..visit.product import sky_at, star_model_at

        x = np.asarray(x, dtype='f8')
        y = np.asarray(y, dtype='f8')
        coords = np.array([y.ravel() / BG_STRIDE, x.ravel() / BG_STRIDE])
        bg = ndimage.map_coordinates(
            self.bg, coords, order=1, mode='nearest',
        ).reshape(x.shape)
        return bg - sky_at(self.product, x, y) - star_model_at(
            self.product, self.wing, x, y,
        )


def visit_geometry(butler, visit):
    """
    The WCS, bbox and calibration of every detector of a visit.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    visit: int

    Returns
    -------
    geometry: dict
        detector -> (wcs, bbox, calib nJy per ADU), from visit_summary
    """
    cat = butler.get(
        'visit_summary', dataId=dict(instrument=INSTRUMENT, visit=int(visit)),
    )
    geo = {}
    for rec in cat:
        pc = rec.getPhotoCalib()
        geo[int(rec['id'])] = (
            rec.getWcs(), rec.getBBox(),
            float(pc.getCalibrationMean()) if pc else None,
        )
    return geo


def load_input_model(butler, geometry, visit, detector, product_file,
                     wings):
    """
    Build one input's InputModel.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    geometry: dict
        visit_geometry(butler, visit)
    visit, detector: int
    product_file: str
        The detector's profiles file with the product
    wings: dict
        A cache of wing models by file name, filled here

    Returns
    -------
    model: InputModel, or None when the visit summary lacks the
        detector or its calibration
    """
    from ..visit.exposure import render_background_list
    from ..visit.product import read_product, wing_file
    from ..wing import read_wing_model

    geo = geometry.get(int(detector))
    if geo is None or geo[2] is None:
        return None
    wcs, bbox, _ = geo
    # the calibration the product was built with: the loader
    # calibrates the preliminary image with its own photoCalib, whose
    # mean sits 0.15-0.3 percent above the visit summary's final one;
    # on a 1200 nJy sky that is a 2-4 nJy pedestal per detector
    calib = butler.get(
        'preliminary_visit_image.photoCalib',
        dataId=dict(instrument=INSTRUMENT, visit=int(visit),
                    detector=int(detector)),
    ).getCalibrationMean()
    m = InputModel()
    m.product = read_product(product_file)
    wf = wing_file(m.product)
    if wf not in wings:
        wings[wf] = read_wing_model(wf)
    m.wing = wings[wf]
    m.wcs = wcs
    m.ny, m.nx = m.product.shape
    bglist = butler.get(
        'preliminary_visit_image_background',
        dataId=dict(instrument=INSTRUMENT, visit=int(visit),
                    detector=int(detector)),
    )
    full = render_background_list(bglist, range(len(bglist)), calib)
    m.bg = np.ascontiguousarray(full[::BG_STRIDE, ::BG_STRIDE], dtype='f8')
    return m


def correction_coadd(butler, mcoadd, product_path, verbose=True):
    """
    Coadd the per-input corrections over a patch, cell by cell.

    Parameters
    ----------
    butler: lsst.daf.butler.Butler
    mcoadd: MultipleCellCoadd
        The patch's cell coadd (cellcoadd.load_cell_coadd), for the
        cells' inputs and weights and the coadd WCS
    product_path: callable
        product_path(visit, detector) -> the profiles file with the
        product, or None when the scheme did not run on that input
    verbose: bool, optional

    Returns
    -------
    result: dict
        correction (ny, nx, nJy, the inner bbox frame; the weighted
        mean of the inputs' corrections over the cell's full weight),
        wfrac (the weight fraction corrected), ninput (inputs
        corrected per pixel), bbox (x0, y0, nx, ny), inputs (a table:
        visit, detector, has_product, weight_sum)
    """
    import os

    bb = mcoadd.inner_bbox
    x0, y0 = bb.getBeginX(), bb.getBeginY()
    nx, ny = bb.getWidth(), bb.getHeight()
    correction = np.zeros((ny, nx), dtype='f4')
    wfrac = np.zeros((ny, nx), dtype='f4')
    ninput = np.zeros((ny, nx), dtype='i2')

    # the models, loaded on first use; the geometry per visit
    models = {}
    geometry = {}
    wings = {}
    weight_sum = {}
    has_product = {}
    cells = mcoadd.cells
    wcs = mcoadd.wcs

    def get_model(visit, detector):
        key = (int(visit), int(detector))
        if key in models:
            return models[key]
        f = product_path(*key)
        if f is None or not os.path.exists(f):
            models[key] = None
            has_product[key] = False
            return None
        if key[0] not in geometry:
            geometry[key[0]] = visit_geometry(butler, key[0])
        models[key] = load_input_model(
            butler, geometry[key[0]], key[0], key[1], f, wings,
        )
        has_product[key] = models[key] is not None
        return models[key]

    outside = 0
    for k, (index, cell) in enumerate(cells.items()):
        cb = cell.inner.bbox
        cx0, cy0 = cb.getBeginX(), cb.getBeginY()
        cnx, cny = cb.getWidth(), cb.getHeight()
        gy, gx = np.mgrid[cy0:cy0 + cny, cx0:cx0 + cnx]
        ra, dec = wcs.pixelToSkyArray(
            gx.ravel().astype('f8'), gy.ravel().astype('f8'), degrees=True,
        )
        num = np.zeros(ra.size)
        wtot = 0.0
        wcorr = 0.0
        n = 0
        for ident, inp in cell.inputs.items():
            w = float(inp.weight)
            if not w > 0:
                continue
            wtot += w
            key = (int(ident.visit), int(ident.detector))
            weight_sum[key] = weight_sum.get(key, 0.0) + w
            model = get_model(*key)
            if model is None:
                continue
            xv, yv = model.wcs.skyToPixelArray(ra, dec, degrees=True)
            inside = ((xv > -0.5) & (xv < model.nx - 0.5)
                      & (yv > -0.5) & (yv < model.ny - 0.5))
            corr = np.zeros(ra.size)
            if inside.all():
                corr = model.correction_at(xv, yv)
            else:
                outside += int((~inside).sum())
                corr[inside] = model.correction_at(xv[inside], yv[inside])
            num += w * corr
            wcorr += w
            n += 1
        if wtot > 0:
            sl = np.s_[cy0 - y0:cy0 - y0 + cny, cx0 - x0:cx0 - x0 + cnx]
            correction[sl] = (num / wtot).reshape(cny, cnx)
            wfrac[sl] = wcorr / wtot
            ninput[sl] = n
        if verbose and (k + 1) % 50 == 0:
            print(f'    {k + 1} of {len(cells)} cells', flush=True)

    keys = sorted(weight_sum)
    inputs = np.array(
        [(v, d, int(has_product.get((v, d), False)), weight_sum[(v, d)])
         for v, d in keys],
        dtype=[('visit', 'i8'), ('detector', 'i4'), ('has_product', 'i2'),
               ('weight_sum', 'f8')],
    )
    if verbose:
        nprod = int(inputs['has_product'].sum())
        print(f'    {len(cells)} cells, {len(keys)} inputs, {nprod} with a '
              f'product; weight fraction corrected: median '
              f'{np.median(wfrac):.3f}, min {wfrac.min():.3f}; '
              f'{outside} input pixels fell outside their detector')
    return dict(correction=correction, wfrac=wfrac, ninput=ninput,
                bbox=(x0, y0, nx, ny), inputs=inputs)
