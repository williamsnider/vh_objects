### Parameters used in generating objects

# Axial component construction parameters
SAMPLING_DENSITY_U = 100  # How densely to sample along round axis of 'cylinder' axial component
SAMPLING_DENSITY_V = 100  # How densely to sample along long axis of 'cylinder' axial component
SLIDE_FACTOR = 0.3  # Impacts surface slope near endpoints
SHRINK_FACTOR = 0.3  # Impacts surface slope at cross sections adjacent to endpoint
ORDER = 3  # quadratic B-spline

# Shape construction parameters
HARMONIC_POWER = 3  # 2-> G1 curvature fairing, 3-> G2 curvature fairing, ...
FAIRING_DISTANCE = 0.1
