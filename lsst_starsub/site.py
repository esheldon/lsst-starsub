"""
The instrument, skymap and butler defaults of the visit-level tools.
"""

# the weekly reprocessing runs carry the shapelets IQ score in
# visit_detector_table; DP2 itself does not
VISIT_REPO = 'dp2_prep'
VISIT_COLLECTION = 'LSSTCam/runs/DRP/w_2026_32/DM-55677'
INSTRUMENT = 'LSSTCam'
SKYMAP = 'lsst_cells_v2'
