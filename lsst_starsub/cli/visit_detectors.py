"""
cli/visit_detectors: the detectors of a visit for the visit scheme.

Those with a wcs and a calibration in the visit summary, minus the
heavily vignetted ones beyond --max-radius mm from the focal-plane
center (default lsst_starsub.visit.exposure.MAX_FIELD_RADIUS; 0 keeps
all), one per line on stdout, for the job scripts, the gather and the
stacks.

    lsst-starsub-visit-detectors --visit V > detectors.txt
"""


def get_args():
    """
    Parse the command line.

    Returns
    -------
    args: argparse.Namespace
    """
    import argparse
    from . import add_butler_arguments
    from ..visit.exposure import MAX_FIELD_RADIUS

    parser = argparse.ArgumentParser()
    parser.add_argument('--visit', type=int, required=True)
    parser.add_argument('--max-radius', type=float, default=MAX_FIELD_RADIUS,
                        help='mm; 0 keeps every detector')
    add_butler_arguments(parser)
    return parser.parse_args()


def main():
    """
    Print the detectors.
    """
    import sys
    from ..visit.exposure import make_visit_butler, visit_detectors

    args = get_args()
    butler = make_visit_butler(args.repo, args.collection)
    max_radius = args.max_radius if args.max_radius > 0 else None
    dets = visit_detectors(butler, args.visit, max_radius=max_radius)
    for det in dets:
        print(det)
    print(f'{len(dets)} detectors', file=sys.stderr)


if __name__ == '__main__':
    main()
