"""
The DM mask-plane bits the star code reads.

The values of the LSST pipelines' mask planes as they appear in the
coadds and visit images.  lsst_mdet.defaults carries the same values for
the rest of the pipeline; a test there checks that they agree.
"""
DM_NO_DATA = 1
DM_SAT = 2
DM_INTRP = 4
DM_DETECTION_EDGE = 16
DM_OUT = DM_NO_DATA | DM_DETECTION_EDGE
