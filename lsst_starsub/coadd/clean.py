"""
The star-and-sky characterization of one patch image, with profiles.

Shared by lsst-starsub-cell-clean (real coadds, optionally with
injected stars) and lsst-starsub-sim (simulated coadds, every star a
known truth).
"""
import os

import numpy as np


def run_clean(vexp, gaia, tbox, state_name, gsub, nround, grow_bright,
              star_model, canonical, truth=None, inj=None,
              joint_spacing=None, joint_prior=None):
    """
    Subtract the stars and the sky of a patch and measure the profiles.

    Run handle_stars_visit on vexp (in place) and measure the
    profiles of the states around the census stars.

    Parameters
    ----------
    vexp: VisitExposure
        The patch image as a visit-like exposure, bbox set to
        the tract-frame box (tbox); left star- and sky-subtracted
    gaia: array
        The Gaia extract, injected rows included
    tbox: SimpleBox
        The tract-frame box of the image
    state_name: str
        Name of the input state in the tables
    gsub: float
        The census depth
    nround: int
        The template model's sky-and-amplitude rounds
    grow_bright: float or None
        See handle_stars_visit
    star_model: str
        'template', 'canonical' or 'joint' (handle_stars_visit)
    canonical: WingModel or (r, T)
        The canonical wing
    truth: array, optional
        The injected (or simulated) star image; adds the
        'perfect' (flat - truth) and 'model_error' (model -
        truth) states
    inj: array, optional
        Table with x, y of the injected stars; flags the rows
    joint_spacing: float, optional
        The joint fit's sky mesh node spacing
    joint_prior: float, optional
        The joint fit's amplitude prior width

    Returns
    -------
    out: dict
        res, states, seg, ambient, edges, ptable, dedges, dtable,
        star_table, sky_sigma
    """
    from ..census import field_segmentation
    from ..geom import SimpleBox
    from ..visit.profiles import ambient_levels, measure_profiles
    from ..visit.exposure import build_wide_star_mask, handle_stars_visit

    res = handle_stars_visit(
        vexp, gaia, gsub=gsub, restore='none', nround=nround,
        grow_bright=grow_bright, star_model=star_model,
        canonical=canonical, joint_spacing=joint_spacing,
        joint_prior=joint_prior,
    )
    shape = vexp.image.array.shape
    vexp.bbox = SimpleBox(0, shape[1], 0, shape[0])
    residual = vexp.image.array
    flat = res['delivered'] - res['sky']
    states = {state_name: res['delivered'], 'flat': flat,
              'residual': residual}
    if truth is not None:
        # the residual under a perfect star model (the sky
        # interpolation alone) and the model's error; the
        # residual is their difference
        states['perfect'] = flat - truth
        states['model_error'] = res['star_model'] - truth
        from ..inject import measure_core_zero_point
        measure_core_zero_point(flat, vexp.good, res['stars'],
                                vexp.sky_sigma)
    seg = field_segmentation(residual, vexp.good, vexp.sky_sigma)
    wide = build_wide_star_mask(res['stars'], seg.shape)
    ambient = ambient_levels(states, vexp, seg, wide)
    print('    ambient levels (nJy): ' + ', '.join(
        f'{k} {v:.2f}' for k, v in ambient.items()
    ))
    edges, ptable = measure_profiles(
        states, vexp, res['stars'], seg, ambient=ambient,
    )
    dedges, dtable = measure_profiles(
        states, vexp, res['stars'], seg, ambient=ambient, mode='dmask',
    )
    star_table = res['star_table']
    if inj is not None:
        from ..inject import flag_injected
        star_table = flag_injected(star_table, inj)
        ptable = flag_injected(ptable, inj)
        dtable = flag_injected(dtable, inj)
    return dict(
        res=res, states=states, seg=seg, ambient=ambient,
        edges=edges, ptable=ptable, dedges=dedges, dtable=dtable,
        star_table=star_table, truth=truth, inj=inj,
        sky_sigma=vexp.sky_sigma, vexp=vexp,
    )


def write_clean_file(stem, out, meta, no_images=False, extra_tables=None):
    """
    Write the clean output file and its summary png.

    The image states, sky, star model, star mask and truth (unless
    no_images), the census, the injected table, the profile tables
    and their edges, the meta table and any extra tables.

    Parameters
    ----------
    stem: str
        The output path stem (clean_stem); .fits and .png are added
    out: dict
        From run_clean, plus vexp, truth and inj
    meta: dict
        The run identity
    no_images: bool, optional
        Write the tables only
    extra_tables: dict, optional
        extname -> table

    Returns
    -------
    fname: str
        The FITS file written
    """
    import rustfits
    from .io import _meta_table, plot_summary

    res = out['res']
    fname = stem + '.fits'
    print('writing:', fname)
    with rustfits.FITS(fname, 'w+') as fits:
        if not no_images:
            for name, arr in out['states'].items():
                fits.write_image(
                    np.ascontiguousarray(arr, dtype='f4'),
                    extname=name, compress='gzip_2',
                )
            for name in ('sky', 'star_model'):
                fits.write_image(
                    np.ascontiguousarray(res[name], dtype='f4'),
                    extname=name, compress='gzip_2',
                )
            fits.write_image(res['starmask'].astype('u1'),
                             extname='starmask', compress='gzip_2')
            if out['truth'] is not None:
                fits.write_image(
                    np.ascontiguousarray(out['truth'], dtype='f4'),
                    extname='truth', compress='gzip_2',
                )
        fits.write_table(out['star_table'], extname='gaia_stars')
        if out['inj'] is not None:
            fits.write_table(out['inj'], extname='injected')
        fits.write_table(out['ptable'], extname='profiles')
        edges = out['edges']
        edges_t = np.zeros(1, dtype=[('edges', 'f8', edges.size)])
        edges_t['edges'][0] = edges
        fits.write_table(edges_t, extname='edges')
        fits.write_table(out['dtable'], extname='profiles_dmask')
        dedges = out['dedges']
        dedges_t = np.zeros(1, dtype=[('edges', 'f8', dedges.size)])
        dedges_t['edges'][0] = dedges
        fits.write_table(dedges_t, extname='dmask_edges')
        fits.write_table(_meta_table(meta), extname='meta')
        for name, tab in (extra_tables or {}).items():
            fits.write_table(tab, extname=name)
    try:
        plot_summary(stem + '.png', out['vexp'], res, out['states'],
                     out['edges'], out['ptable'])
    except Exception as err:
        print(f'    WARNING: summary plot failed: {err!r}')
    return fname


def clean_tag(state, star_model, inject_tag=None, joint_spacing=None):
    """
    Build the output-name tag of a clean run.

    Parameters
    ----------
    state: str
        The input state name
    star_model: str
        'template' or 'joint'
    inject_tag: str, optional
        Appended for an injection run
    joint_spacing: float, optional
        The joint fit's node spacing, appended to a joint tag

    Returns
    -------
    tag: str
    """
    tag = state if star_model == 'template' else f'{state}-{star_model}'
    if star_model == 'joint' and joint_spacing is not None:
        tag += f'{int(joint_spacing)}'
    if inject_tag:
        tag += f'-{inject_tag}'
    return tag


def clean_stem(outdir, tag, tract, patch, band, prefix='clean'):
    """
    Build the output path stem of a clean run.

    Parameters
    ----------
    outdir: str
    tag: str
        From clean_tag
    tract, patch: int
    band: str
    prefix: str, optional
        The file-name prefix; default 'clean'

    Returns
    -------
    stem: str
        {outdir}/{prefix}-{tag}-{tract}-{patch}-{band}
    """
    return os.path.join(
        outdir, f'{prefix}-{tag}-{tract:05d}-{patch:02d}-{band}',
    )
