"""
cli/visit_wing: a visit's own wing from its pooled template.

The visit's stack inside the junction, the band's canonical wing
beyond (lsst_starsub.visit.trough.visit_wing), written in the
canonical wing file format for --canonical of lsst-starsub-visit,
whose --core-rap needs the visit's own core.

    lsst-starsub-visit-wing --template template-V-B.fits \\
        --canonical canonical-wing-B.fits --outfile wing-V-B.fits
"""


def get_args():
    """
    Parse the command line.

    Returns
    -------
    args: argparse.Namespace
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True,
                        help='the pooled per-visit template file '
                             '(lsst-starsub-visit-template)')
    parser.add_argument('--canonical', required=True,
                        help='the band\'s canonical wing file')
    parser.add_argument('--outfile', required=True)
    return parser.parse_args()


def main():
    """
    Write the visit's wing file.
    """
    from ..visit.template import read_template_file
    from ..visit.trough import visit_wing
    from ..wing import read_wing_model, write_canonical_wing

    args = get_args()
    tmpl = read_template_file(args.template)
    canonical = read_wing_model(args.canonical)
    wing, scale = visit_wing(tmpl, canonical)
    band = str(tmpl['params']['band'])
    print(f'visit {int(tmpl["params"]["visit"])} {band}: k_in '
          f'{float(tmpl["params"]["k_in"]):.3e} scaled by {scale:.3f} onto '
          f'the canonical at the junction, fwhm '
          f'{float(tmpl["params"]["fwhm"]):.2f} arcsec; writing '
          f'{args.outfile}')
    write_canonical_wing(args.outfile, wing.r, wing.T, band, 1)


if __name__ == '__main__':
    main()
