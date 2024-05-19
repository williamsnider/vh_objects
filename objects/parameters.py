### Parameters used in generating objects
import numpy as np
from pathlib import Path

# Backbone construction
BACKBONE_LENGTH = 80  # mm length of Backbone
BACKBONE_NUM_CP = 10  # Number of controlpoints to construct backbone
NUM_INTERPOLATION_POINTS = 1000  #
NUM_SAMPLES_FOR_REPARAMETERIZATION = 10 * NUM_INTERPOLATION_POINTS
EPSILON = 1e-2  # Used to estimate b-spline derivative

# AxialComponent construction
SAMPLING_DENSITY_U = 75  # Number of samples along round axis of 'cylinder' axial component
SAMPLING_DENSITY_V = 75  # Number of samples along long axis of 'cylinder' axial component
SHRINK_FACTOR = 0.5  # Impacts surface slope near endpoints
ORDER = 3  # quadratic B-spline

# Shape construction
HARMONIC_POWER = 2  # 2-> G1 curvature fairing, 3-> G2 curvature fairing, etc
FAIRING_DISTANCE = 3  # How far (mm) from a junction should vertices in a union mesh be faired

# Interface construction
INTERFACE_PATH = Path("./assets/Interface_0038 v3.stl")
FONT_PATH = Path("./assets/consolab.ttf")
INTERFACE_WIDTH = 12.70
INTERFACE_HEIGHT_ABOVE_ORIGIN = 9.071
INTERFACE_DEPTH_FROM_ORIGIN = INTERFACE_HEIGHT_ABOVE_ORIGIN
INTERFACE_SHIFT = -25  # Length of post
LABEL_DEPTH = 1
FONT_HEIGHT_IN_MM = 4.5


# Component construction
cs_scale_backbone = BACKBONE_LENGTH / 4  # Controls thickness of cross sections around backbone
cs_scale_surface_deformation = 5  # Controls thickness of cross sections in surface deformations
SD_LENGTH = 12  # Part of this will be inside parent shape
STRAIGHT_PROPORTION = 0.25
ARC_ANGLE = np.pi / 4
ELLIPTICAL_MAJOR = 7 / 6
ELLIPTICAL_MINOR = 1 - (ELLIPTICAL_MAJOR - 1)

# Misc
SAVE_DIR = "../sample_shapes"
