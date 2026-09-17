"""
The per-detector product of the visit scheme, and its renderer.

A profiles file from lsst-starsub-visit --profiles-only with the joint
model holds everything that reproduces the fit: the census with the
amplitudes (gaia_stars), the sky mesh nodes (nodes), the wide-pass box
grid (sky_boxes) and the run meta (the wing file the amplitudes refer
to, the mesh spacing, the image shape).  The sky model is the wide
boxes plus the mesh, both bilinear, so it evaluates at any position;
the star model is the wing at each star's amplitude.  A consumer
renders image - sky - stars from this, on the detector grid or at
arbitrary positions (a coadd's pixels through the WCS).
"""
import numpy as np


class Product:
    """
    One detector's fit as read from its profiles file.

    Attributes
    ----------
    stars: structured array
        The census with x, y, G, A (and A_err, free)
    xn, yn: arrays
        The mesh node coordinates in px
    node_values: array (yn.size, xn.size)
    boxes: array
        The wide-pass box grid
    bw: int
        Its box size in px
    shape: (ny, nx)
        The detector image shape
    meta: structured array
        The run meta, one row (visit, detector, band, canonical ...)
    """


def read_product(fname):
    """
    Read a detector's product.

    Parameters
    ----------
    fname: str
        The profiles file (joint model, with sky_boxes)

    Returns
    -------
    product: Product
    """
    import rustfits

    p = Product()
    with rustfits.FITS(fname) as fits:
        p.meta = fits['meta'].read()
        p.stars = fits['gaia_stars'].read()
        nodes = fits['nodes'].read()
        hdu = fits['sky_boxes']
        p.boxes = hdu.read().astype('f8')
        p.bw = int(hdu.header['BW'])
    m = p.meta[0]
    nx_, ny_ = int(m['nxnode']), int(m['nynode'])
    spacing = float(m['spacing'])
    p.xn = np.arange(nx_) * spacing
    p.yn = np.arange(ny_) * spacing
    vals = np.zeros((ny_, nx_))
    vals[nodes['iy'], nodes['ix']] = nodes['value']
    p.node_values = vals
    p.shape = (int(m['ny']), int(m['nx']))
    return p


def mesh_at(xn, yn, values, x, y):
    """
    Evaluate the bilinear mesh at positions.

    The evaluation lsst_starsub.joint.render_mesh makes on the pixel
    grid, at arbitrary positions: pixels beyond the outer nodes use
    the last interval.

    Parameters
    ----------
    xn, yn: arrays
        The node coordinates, evenly spaced
    values: array (yn.size, xn.size)
    x, y: arrays
        Positions in px

    Returns
    -------
    sky: array, x's shape
    """
    x = np.asarray(x, dtype='f8')
    y = np.asarray(y, dtype='f8')
    spacing = float(xn[1] - xn[0])
    ix = np.clip((x // spacing).astype(int), 0, xn.size - 2)
    iy = np.clip((y // spacing).astype(int), 0, yn.size - 2)
    fx = (x - xn[ix]) / spacing
    fy = (y - yn[iy]) / spacing
    rows0 = (1 - fy) * values[iy, ix] + fy * values[iy + 1, ix]
    rows1 = (1 - fy) * values[iy, ix + 1] + fy * values[iy + 1, ix + 1]
    return (1 - fx) * rows0 + fx * rows1


def sky_at(product, x, y):
    """
    The total sky model at positions: the wide boxes plus the mesh.

    Parameters
    ----------
    product: Product
    x, y: arrays
        Positions in the detector's px

    Returns
    -------
    sky: array, x's shape, nJy
    """
    from .exposure import boxes_at

    return (boxes_at(product.boxes, product.bw, x, y)
            + mesh_at(product.xn, product.yn, product.node_values, x, y))


def render_sky(product):
    """
    The total sky model on the detector grid.

    Returns
    -------
    sky: array (ny, nx) f4
    """
    ny, nx = product.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return sky_at(product, xx, yy).astype('f4')


def render_stars(product, wing):
    """
    The star model on the detector grid.

    Parameters
    ----------
    product: Product
    wing: WingModel or (r, T)
        The wing the amplitudes refer to (the file's meta names it:
        lsst_starsub.wing.read_wing_model)

    Returns
    -------
    stars: array (ny, nx) f4
    """
    from ..joint import render_disks
    from ..wing import render_canonical_stars

    model = render_canonical_stars(
        product.shape, product.stars, wing, amps=product.stars['A'],
        verbose=False,
    )
    if 'D' in product.stars.dtype.names:
        model += render_disks(product.shape, product.stars,
                              product.stars['D'], product_band(product))
    return model


def product_band(product):
    """The band of a product, from its meta."""
    band = product.meta['band']
    band = band.decode() if isinstance(band, bytes) else str(band)
    return band.strip()


def disk_model_at(product, x, y):
    """
    The ghost disks at positions (joint.render_disks at positions).

    Parameters
    ----------
    product: Product
    x, y: arrays
        Positions in the detector's px

    Returns
    -------
    model: array, x's shape, nJy; zero for files without disks
    """
    from ..joint import DISK_EDGE, DISK_RADIUS, disk_prediction, disk_profile

    x = np.asarray(x, dtype='f8')
    y = np.asarray(y, dtype='f8')
    out = np.zeros(x.shape)
    stars = product.stars
    if 'D' not in stars.dtype.names:
        return out
    flat_x, flat_y, flat_out = x.ravel(), y.ravel(), out.ravel()
    rmax = DISK_RADIUS + DISK_EDGE
    band = product_band(product)
    for xk, yk, gk, dk in zip(stars['x'], stars['y'], stars['G'],
                              stars['D']):
        if not dk > 0:
            continue
        m = int(np.ceil(rmax)) + 1
        ix, iy = int(round(xk)), int(round(yk))
        sel = ((flat_x >= ix - m) & (flat_x < ix + m + 1)
               & (flat_y >= iy - m) & (flat_y < iy + m + 1))
        if not sel.any():
            continue
        rr = np.hypot(flat_x[sel] - xk, flat_y[sel] - yk)
        flat_out[sel] += (float(dk) * float(disk_prediction(gk, band))
                          * disk_profile(rr))
    return out


def star_model_at(product, wing, x, y, eps=None):
    """
    The star model at positions.

    The evaluation lsst_starsub.wing.render_wing_image makes on the
    pixel grid (each star's wing at its amplitude, within the square
    window where it exceeds eps), plus the ghost disks (disk_model_at),
    at arbitrary positions in the detector's frame: the star model of
    a coadd pixel, once the pixel is mapped into the detector through
    the WCS.

    Parameters
    ----------
    product: Product
    wing: WingModel or (r, T)
        The wing the amplitudes refer to
    x, y: arrays
        Positions in the detector's px
    eps: float, optional
        The window cutoff in nJy; default lsst_starsub.wing.RENDER_EPS
        as render_canonical_stars uses it (0.005)

    Returns
    -------
    model: array, x's shape, nJy
    """
    from ..wing import profile_of

    eps = 0.005 if eps is None else eps
    x = np.asarray(x, dtype='f8')
    y = np.asarray(y, dtype='f8')
    out = np.zeros(x.shape)
    flat_x, flat_y, flat_out = x.ravel(), y.ravel(), out.ravel()
    stars = product.stars
    for xk, yk, gk, ak in zip(stars['x'], stars['y'], stars['G'],
                              stars['A']):
        if not ak > 0:
            continue
        r, T = profile_of(wing, float(gk))
        prof = ak * 10.0 ** (-0.4 * float(gk)) * T
        below = np.flatnonzero(prof < eps)
        rmax = float(r[below[0]]) if below.size else float(r[-1])
        m = int(np.ceil(rmax)) + 1
        ix, iy = int(round(xk)), int(round(yk))
        sel = ((flat_x >= ix - m) & (flat_x < ix + m + 1)
               & (flat_y >= iy - m) & (flat_y < iy + m + 1))
        if not sel.any():
            continue
        rr = np.hypot(flat_x[sel] - xk, flat_y[sel] - yk)
        flat_out[sel] += np.interp(rr, r, prof, right=0.0)
    return out + disk_model_at(product, x, y)


def wing_file(product):
    """The wing file the product's amplitudes refer to."""
    return str(product.meta['canonical'][0])
