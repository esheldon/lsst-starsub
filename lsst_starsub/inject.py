"""
synthetic stars with a known wing for the injection test (TODO
step 7)

Each injected star is

    10^(-0.4 G) [ (1 - f(r)) C P(r) + f(r) w T(r) ]

with P the coadd's own PSF at the position (the cell's psf
image), T the canonical wing (nJy per unit Gaia flux, core
included), C the factor that gives the PSF core the canonical's
core flux (the sum within CORE_R, the stamp normalization radius
of the template code), f a linear ramp from BLEND_R0 to BLEND_R1
and w the wing scale: 1 for the truth as measured, 1.3 for the
sensitivity variant, 10 for the amplified one.  Everything beyond
BLEND_R1 is the canonical wing times w; the join lies inside every
mask circle (MINRAD 20 px).

The stars go into the Gaia census as rows with proper motion zero,
so masking, template selection and subtraction treat them as real
stars.  Positions are integer pixels (no core resampling), at
least CENTER_CLEAR px from any existing star mask and far enough
inside the patch for the anchor ring
"""
import numpy as np

CORE_R = 6.0          # stamp core normalization radius (template.py)
BLEND_R0 = 8.0
BLEND_R1 = 12.0
CENTER_CLEAR = 30.0   # min dstar at an injected center
EDGE_EXTRA = 60.0     # inside the patch by circle radius + this
EPS = 0.005           # render out to where the star falls below (nJy)

# per-patch injection counts per G range; the bright range is
# where the collar is, the faint ones span the profile bins
DEFAULT_PLAN = [
    ((8.0, 13.0), 2),
    ((13.0, 14.0), 2),
    ((14.0, 15.2), 3),
    ((15.2, 16.0), 3),
    ((16.0, 17.0), 4),
]


def parse_plan(text):
    """
    Parse an injection plan string.

    'glo-ghi:n,glo-ghi:n' -> [((glo, ghi), n), ...]
    """
    plan = []
    for item in text.split(','):
        rng, n = item.split(':')
        glo, ghi = rng.split('-')
        plan.append(((float(glo), float(ghi)), int(n)))
    return plan


def draw_positions(rng, plan, dstar, shape):
    """
    Draw random positions and magnitudes for the injected stars.

    Random integer positions and magnitudes: the center clear
    of existing masks, the star's mask circle plus EDGE_EXTRA
    inside the image

    Returns
    -------
    x, y (int arrays), G
    """
    from lsst_mdet.starsub import circle_radius

    ny, nx = shape
    clear = dstar > CENTER_CLEAR
    xs, ys, gs = [], [], []
    for (glo, ghi), n in plan:
        for _ in range(n):
            g = float(rng.uniform(glo, ghi))
            margin = int(np.ceil(circle_radius(g) + EDGE_EXTRA))
            for _try in range(1000):
                x = int(rng.integers(margin, nx - margin))
                y = int(rng.integers(margin, ny - margin))
                if clear[y, x] and all(
                    np.hypot(x - xo, y - yo) > CENTER_CLEAR
                    for xo, yo in zip(xs, ys)
                ):
                    break
            else:
                raise RuntimeError(f'no clear position for G {g:.2f}')
            xs.append(x)
            ys.append(y)
            gs.append(g)
    return (np.array(xs, dtype='i8'), np.array(ys, dtype='i8'),
            np.array(gs, dtype='f8'))


def psf_cube(mcoadd):
    """
    Collect the per-cell psf images of a cell coadd.

    The per-cell psf images of the cell coadd

    Returns
    -------
    cube (ny_cell, nx_cell, npsf, npsf), the grid's tract-frame
    origin (x0, y0) and the cell size
    """
    grid = mcoadd.grid
    shape = grid.shape
    cube = None
    for idx, cell in mcoadd.cells.items():
        p = cell.psf_image.array
        if cube is None:
            cube = np.zeros((shape.y, shape.x) + p.shape, dtype='f8')
        cube[idx.y, idx.x] = p
    gb = grid.bbox
    return cube, (gb.getBeginX(), gb.getBeginY()), grid.cell_size.x


def core_factor(psf, canonical):
    """
    Scale a unit-sum psf to the canonical core flux.

    The factor scaling the unit-sum psf so its sum within CORE_R
    equals the canonical wing's
    """
    r, T = canonical
    c = (psf.shape[0] - 1) // 2
    gy, gx = np.mgrid[0:psf.shape[0], 0:psf.shape[1]]
    rr = np.hypot(gy - c, gx - c)
    psf_core = psf[rr < CORE_R].sum()
    m = int(np.ceil(CORE_R)) + 1
    gy, gx = np.mgrid[-m:m + 1, -m:m + 1]
    rr = np.hypot(gy, gx)
    can_core = np.interp(rr, r, T)[rr < CORE_R].sum()
    return can_core / psf_core


def render_injected(shape, x, y, G, canonical, cube, origin, cell_size,
                    bbox_start, wing_scale=1.0, eps=EPS):
    """
    Render the injected stars.

    The summed injected-star image (nJy) and, per star, the
    canonical core flux and the truth parameters

    Parameters
    ----------
    shape: (ny, nx) of the patch image
    x, y, G: arrays, patch-frame integer positions
    canonical: (r, T)
    cube, origin, cell_size: from psf_cube
    bbox_start: (x0, y0) tract-frame origin of the patch image
    wing_scale: float
    """
    ny, nx = shape
    r, T = canonical
    image = np.zeros((ny, nx), dtype='f8')
    npsf = cube.shape[-1]
    hp = (npsf - 1) // 2
    cores = []
    for xk, yk, gk in zip(x, y, G):
        flux = 10.0 ** (-0.4 * gk)
        # the cell psf at the position
        tx = xk + bbox_start[0]
        ty = yk + bbox_start[1]
        ci = int((tx - origin[0]) // cell_size)
        cj = int((ty - origin[1]) // cell_size)
        ci = min(max(ci, 0), cube.shape[1] - 1)
        cj = min(max(cj, 0), cube.shape[0] - 1)
        psf = cube[cj, ci]
        if not psf.sum() > 0:
            raise RuntimeError(f'empty psf in cell ({ci}, {cj})')
        C = core_factor(psf, canonical)
        cores.append(C * flux)

        # window: where the wing falls below eps
        prof = flux * wing_scale * T
        below = np.flatnonzero(prof < eps)
        rmax = float(r[below[0]]) if below.size else float(r[-1])
        m = int(np.ceil(rmax)) + 1
        x0, x1 = max(0, xk - m), min(nx, xk + m + 1)
        y0, y1 = max(0, yk - m), min(ny, yk + m + 1)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(gy - yk, gx - xk)
        f = np.clip((rr - BLEND_R0) / (BLEND_R1 - BLEND_R0), 0.0, 1.0)
        star = f * np.interp(rr, r, prof, right=0.0)
        # the psf core on its own footprint
        px0, px1 = max(0, xk - hp), min(nx, xk + hp + 1)
        py0, py1 = max(0, yk - hp), min(ny, yk + hp + 1)
        pcut = psf[py0 - (yk - hp):py1 - (yk - hp),
                   px0 - (xk - hp):px1 - (xk - hp)]
        core = np.zeros_like(star)
        core[py0 - y0:py1 - y0, px0 - x0:px1 - x0] = C * flux * pcut
        star += (1.0 - f) * core
        image[y0:y1, x0:x1] += star
    return image.astype('f4'), np.array(cores, dtype='f8')


def census_rows(x, y, wcs, bbox_start, G, gaia_dtype):
    """
    Build Gaia census rows for the injected stars.

    The injected stars as Gaia rows (proper motion 0, ruwe 1)
    """
    ra, dec = wcs.pixelToSkyArray(
        (x + bbox_start[0]).astype('f8'), (y + bbox_start[1]).astype('f8'),
        degrees=True,
    )
    rows = np.zeros(x.size, dtype=gaia_dtype)
    rows['ra'] = ra
    rows['dec'] = dec
    rows['phot_g_mean_mag'] = G
    rows['ruwe'] = 1.0
    return rows


def injected_table(x, y, G, ra, dec, cores, wing_scale):
    """
    Build the injected-star truth table.
    """
    t = np.zeros(x.size, dtype=[
        ('x', 'f8'), ('y', 'f8'), ('G', 'f4'), ('ra', 'f8'), ('dec', 'f8'),
        ('core_flux', 'f8'), ('wing_scale', 'f8'),
    ])
    t['x'], t['y'], t['G'] = x, y, G
    t['ra'], t['dec'] = ra, dec
    t['core_flux'] = cores
    t['wing_scale'] = wing_scale
    return t


def flag_injected(table, inj, tol=0.5):
    """
    Add an injected flag column to a table.

    Add an 'injected' column (1 where the row's x, y match an
    injected star) to a profile or census table
    """
    from numpy.lib import recfunctions as rfn

    flag = np.zeros(table.size, dtype='i2')
    if inj is not None and inj.size > 0:
        for k in range(table.size):
            d = np.hypot(table['x'][k] - inj['x'], table['y'][k] - inj['y'])
            if d.min() < tol:
                flag[k] = 1
    return rfn.append_fields(table, 'injected', flag, usemask=False)


def measure_core_zero_point(image, good, stars, sky_sigma, glo=15.5,
                            ghi=17.0):
    """
    Measure the real stars' core flux per unit Gaia flux.

    The real stars' core flux per unit Gaia flux (sum within
    CORE_R above the local median), for comparison with the
    canonical core: prints and returns the median
    """
    ny, nx = image.shape
    m = 20
    gy, gx = np.mgrid[-m:m + 1, -m:m + 1]
    rr = np.hypot(gy, gx)
    vals = []
    for st in stars:
        g = float(st['G'])
        if not (glo <= g < ghi) or not st['on_image']:
            continue
        ix, iy = int(round(st['x'])), int(round(st['y']))
        if ix < m or iy < m or ix >= nx - m or iy >= ny - m:
            continue
        cut = image[iy - m:iy + m + 1, ix - m:ix + m + 1].astype('f8')
        ok = good[iy - m:iy + m + 1, ix - m:ix + m + 1]
        ann = ok & (rr > 12) & (rr < 20)
        if ann.sum() < 20 or not ok[rr < CORE_R].all():
            continue
        loc = np.median(cut[ann])
        core = (cut - loc)[rr < CORE_R].sum()
        vals.append(core / 10.0 ** (-0.4 * g))
    if len(vals) == 0:
        return np.nan
    zp = float(np.median(vals))
    print(f'    real-star core zero point: {zp:.3e} nJy per unit Gaia '
          f'flux from {len(vals)} stars G {glo}-{ghi}')
    return zp
