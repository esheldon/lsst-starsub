"""
this package imports nothing from lsst_mdet: lsst_mdet calls it, never
the other way
"""
import ast
import pathlib

import lsst_starsub


def test_no_lsst_mdet_imports():
    root = pathlib.Path(lsst_starsub.__file__).parent
    found = []
    for path in sorted(root.rglob('*.py')):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            for name in names:
                if name == 'lsst_mdet' or name.startswith('lsst_mdet.'):
                    found.append(f'{path.relative_to(root)}:{node.lineno}')
    assert not found, f'lsst_mdet imported at {found}'
