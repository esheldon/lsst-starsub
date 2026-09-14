"""
Small geometry adapters: a DM-style bounding box and a wcs wrapper.

Copies of lsst_mdet.patchfiles.SimpleBox and lsst_mdet.wcs.ButlerWcs, so
this package needs nothing from lsst_mdet.  The star code only reads
their attributes, so either package's objects can be passed in.
"""


class SimpleAxis(object):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class SimpleBox(object):
    """bbox with the DM attribute layout: .x/.y start/stop"""
    def __init__(self, x0, x1, y0, y1):
        self.x = SimpleAxis(x0, x1)
        self.y = SimpleAxis(y0, y1)


class ButlerWcs(object):
    """
    thin wrapper around the DM tract wcs adding the
    linearize_matrix method the jacobian helper uses;
    everything else passes through.  The stack import lives
    inside the method so this module imports without the LSST
    pipelines
    """

    def __init__(self, dm_wcs):
        self._wcs = dm_wcs

    def __getattr__(self, name):
        return getattr(self._wcs, name)

    def linearize_matrix(self, x, y):
        import lsst.geom

        dm_jac = self._wcs.linearizePixelToSky(
            lsst.geom.Point2D(x, y),
            lsst.geom.arcseconds,
        )
        return dm_jac.getLinear().getMatrix()
