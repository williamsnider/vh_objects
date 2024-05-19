import numpy as np
from pathlib import Path

### Parameters ###
NUM_CP_PER_BACKBONE = 5
SEGMENT_LENGTH = 35
NUM_CS = 11
X_WIDTH = 4 # base radius off which other features are derived.
NUM_CP_PER_CROSS_SECTION = 50

# Limb sizes
TERMINATION_RADIUS = 0.75  # mm; Smaller result in too sharp a point to be safe
ac_radii = np.array([TERMINATION_RADIUS, 1 * X_WIDTH, 2 * X_WIDTH])
ac_theta_dict = {"th0": 0, "th1": np.pi / 2}
ac_junc_angles = {"ja0": 0, "ja1": np.pi / 4, "ja2": np.pi / 2}
ac_junc_rotations = {"r0": 0, "r1": np.pi}
SPHERE_ALIGNMENT_OFFSET = np.array(
    [0.25, 0.1, 0.1]
)  # Instead of perfectly aligning the left and right spheres of limb1 and limb2, shift limb2 slightly, which is necessary for the boolean union to work. Because these shifts are small (<0.25mm), this should be OK.

# Post and interface
POST_RADIUS = X_WIDTH
POST_OFFSET = 0.25  # mm - Post shifted to improve overlap

# Saving png and stl
SAVE_DIR = Path("./sample_shapes/")

# Volumetric and appendage sizes
APPENDAGE_LENGTH = SEGMENT_LENGTH / 2
SHEET_THICKNESS = 3
NUM_CP_PER_BASE_SHEET = 50
NUM_CS_PER_SHEET = 11
NUM_CP_PER_CROSS_SECTION = 50
VOLUMETRIC_RADII = np.array([1.01 * X_WIDTH, 2.01 * X_WIDTH, 1.01 * X_WIDTH])
POINT_RADII = np.array([1 * X_WIDTH, 0.75 * X_WIDTH, TERMINATION_RADIUS])
POINT_ROUNDOVER_OFFSET = SHEET_THICKNESS / 5
assert POINT_ROUNDOVER_OFFSET < POINT_RADII[-1]
LEAF_RADII = np.array([1 * X_WIDTH, 1.4 * X_WIDTH, 0.25 * X_WIDTH])
ROUND_RADIUS = 1.25 * X_WIDTH  # radius of sphere and sheet_round

# Modifying appendages to align better
SLICER_DEPTH = -1.0 * X_WIDTH
XYZ_OFFSET = 0.0  # Distance from volumetric/cylinder surface for origin of appendage (helps with boolean union)

# Use a box at the end of shapes to fair (remove bump)
BOX_EXTENTS = 3 * np.array([X_WIDTH, X_WIDTH, X_WIDTH])
BOX_TRANSLATION = np.array([SEGMENT_LENGTH + 3 * X_WIDTH / 4, 0, 0])

# Meshing
uu = 50
vv = 50

