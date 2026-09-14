"""
cli/check_datasets: report whether a butler holds everything
lsst-starsub-cell-restore needs for a patch, without running it:
the cell coadd and object background, and for every input of
the cell coadd the visit_summary and the stored visit background
"""


def get_args():
    import argparse
    from ..site import VISIT_COLLECTION, VISIT_REPO

    parser = argparse.ArgumentParser()
    parser.add_argument('--tract', type=int, required=True)
    parser.add_argument('--patch', type=int, required=True)
    parser.add_argument('--band', default='i')
    parser.add_argument('--repo', default=VISIT_REPO)
    parser.add_argument('--collection', default=VISIT_COLLECTION)
    return parser.parse_args()


def main():
    from ..coadd.cellcoadd import coadd_data_id
    from ..site import INSTRUMENT
    from ..visit.exposure import make_visit_butler

    args = get_args()
    butler = make_visit_butler(args.repo, args.collection)
    did = coadd_data_id(args.tract, args.patch, args.band)
    ok = True

    for name in ('deep_coadd_cell_predetection', 'deep_coadd',
                 'deep_coadd_background'):
        try:
            have = butler.exists(name, did)
        except Exception as err:
            have = False
            print(f'  {name}: error {err!r}')
        print(f'  {name}: {"ok" if have else "MISSING"}')
        if name != 'deep_coadd_cell_predetection':
            ok &= bool(have)

    try:
        from ..coadd.cellcoadd import load_cell_coadd
        mcoadd = load_cell_coadd(butler, did)
    except Exception as err:
        print(f'  cannot read the cell coadd ({err!r}); stopping')
        return
    keys = sorted({
        (int(ident.visit), int(ident.detector))
        for cell in mcoadd.cells.values() for ident in cell.inputs
    })
    visits = sorted({v for v, _ in keys})
    print(f'  cell coadd: {len(mcoadd.cells)} cells, {len(keys)} '
          f'inputs from {len(visits)} visits')

    missing_vs = [
        v for v in visits
        if not butler.exists(
            'visit_summary', dict(instrument=INSTRUMENT, visit=v),
        )
    ]
    missing_bg = [
        (v, d) for v, d in keys
        if not butler.exists(
            'preliminary_visit_image_background',
            dict(instrument=INSTRUMENT, visit=v, detector=d),
        )
    ]
    print(f'  visit_summary: {len(visits) - len(missing_vs)} of '
          f'{len(visits)}' + (f', missing {missing_vs[:5]}'
                             if missing_vs else ''))
    print(f'  preliminary_visit_image_background: '
          f'{len(keys) - len(missing_bg)} of {len(keys)}'
          + (f', missing {missing_bg[:5]}' if missing_bg else ''))
    ok &= not missing_vs and not missing_bg
    print('ALL PRESENT' if ok else 'INCOMPLETE')


if __name__ == '__main__':
    main()
