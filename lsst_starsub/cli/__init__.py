"""
The command-line entry points; the visit-side ones share the butler
arguments below.
"""


def add_butler_arguments(parser):
    """
    Add --repo and --collection with the visit-level defaults.

    Parameters
    ----------
    parser: argparse.ArgumentParser
    """
    from ..site import VISIT_COLLECTION, VISIT_REPO

    parser.add_argument('--repo', default=VISIT_REPO)
    parser.add_argument('--collection', default=VISIT_COLLECTION)
