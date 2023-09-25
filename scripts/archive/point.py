# Make a point
import numpy as np
from scripts.sheets import plot_arr
from objects.utilities import make_surface, make_mesh, approximate_arc, calc_hemisphere_controlpoints

SHEET_THICKNESS = 3
ROUNDOVER_OFFSET = SHEET_THICKNESS / 3
NUM_CS = 11
APPENDAGE_LENGTH = 10
POINT_RADII = np.array([10, 4, 1])

# Calculate widths
x = np.linspace(0, APPENDAGE_LENGTH, 3)
y = POINT_RADII
poly = np.polyfit(x, y, 2)
xvals = np.linspace(0, APPENDAGE_LENGTH, NUM_CS)
widths = np.polyval(poly, xvals)

# Calculate z_levels
z_levels = np.linspace(0, APPENDAGE_LENGTH, NUM_CS)

# Assign controlpoints
cp = np.zeros((NUM_CS, 8, 3))
for i, width in enumerate(widths):

    if width == 0:
        inner = np.zeros((8, 2))
    else:
        inner = np.array(
            [
                [SHEET_THICKNESS / 2, 0],
                [SHEET_THICKNESS / 2, width / 2],
                [0, width / 2 + ROUNDOVER_OFFSET],
                [-SHEET_THICKNESS / 2, width / 2],
                [-SHEET_THICKNESS / 2, 0],
                [-SHEET_THICKNESS / 2, -width / 2],
                [0, -width / 2 - ROUNDOVER_OFFSET],
                [SHEET_THICKNESS / 2, -width / 2],
            ]
        )

    xyz = np.hstack([inner, z_levels[i] * np.ones((inner.shape[0], 1))])

    cp[i, :, :] = xyz

# Roundover edges

side_y = POINT_RADII + ROUNDOVER_OFFSET
side_poly = np.polyfit(x, side_y, 2)
bot = calc_hemisphere_controlpoints(
    cp[0], np.array([0, 0, 1]), cp[0].mean(axis=0), side_poly, x[0], morph_to_ellipse=True
)
top = calc_hemisphere_controlpoints(
    cp[-1], np.array([0, 0, 1]), cp[-1].mean(axis=0), side_poly, x[-1], morph_to_ellipse=True
)


comb = np.vstack([cp, top[-2::-1]])

plot_arr(comb)


# Make shape
surf = make_surface(comb)
mesh = make_mesh(surf, 100, 100)
mesh.show(smooth=False)
