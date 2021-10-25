### Parameters used in generating objects

# Backbone construction
NUM_INTERPOLATION_POINTS = 1000  #
NUM_SAMPLES_FOR_REPARAMETERIZATION = 10 * NUM_INTERPOLATION_POINTS

# AxialComponent construction
SAMPLING_DENSITY_U = 100  # Number of samples along round axis of 'cylinder' axial component
SAMPLING_DENSITY_V = 100  # Number of samples along long axis of 'cylinder' axial component
SHRINK_FACTOR = 0.5  # Impacts surface slope near endpoints
ORDER = 3  # quadratic B-spline

# Shape construction
HARMONIC_POWER = 2  # 2-> G1 curvature fairing, 3-> G2 curvature fairing, etc
FAIRING_DISTANCE = 10  # How far (mm) from a junction should vertices in a union mesh be faired

# Interface construction
CUBE_SIDE_LENGTH = 25  # mm
POST_RADIUS = 5
POST_HEIGHT = 15
POST_SECTIONS = 10  # Strangely, 8 causes the interface to be non-watertight
FINGERTIP_SLOT_SIDE_LENGTH = 5
PEG_SIDE_LENGTH = 10  # mm
PEG_CORNER_RADIUS = 2  # mm - Round the edges to fit into waterjet cut slots better
PEG_CORNER_NUM_STEPS = 3
PEG_DEPTH = 20  # mm
PEG_SPHERE_SUBDIVISIONS = 1  # Higher --> more points used to fair peg tip

# Misc
SAVE_DIR = "../sample_shapes"
