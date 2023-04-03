# Construct sheets needed in experiment
import numpy as np
from objects.utilities import make_mesh, make_surface, approximate_arc
from objects.backbone import Backbone


def construct_sheet(base_sheet, scale_max, sheet_thickness, num_cs):
    """Constructs a b-spline surface in the form of a sheet.

    scale_max is that maximum amount that base_sheet is multiplied by. For example, if base_sheet is a circle with radius 1, and you ultimately want a surface with radius 11, scale_max should be 11."""

    cp = np.tile(base_sheet, (num_cs, 1, 1))

    # Scale to create progression from endpoint to middle to endpoint
    mid_idx = (num_cs - 1) // 2
    scale = np.linspace(0.0, scale_max, mid_idx - 1)
    scale = np.insert(scale, 1, scale[1] / 5)  # Controls slope at endpoint
    scale = np.concatenate(
        [scale, np.array([scale_max + sheet_thickness / 3]), scale[::-1]]
    )
    cp *= scale.reshape(-1, 1, 1)

    # Translate to create volume along x-axis
    cp[:mid_idx, :, 0] -= sheet_thickness / 2
    cp[mid_idx + 1 :, :, 0] += sheet_thickness / 2

    return cp


def bend_sheet(cp, backbone, max_scale):

    assert np.all(np.isclose(backbone.r(0.0), np.array([0.0, 0.0, 0.0])))
    assert np.isclose(backbone.r(0.0)[0, 1], 0)  # Must lie in yz-plane

    # Calc relative distance each point should travel along backbone
    mid = (cp.shape[0] - 1) // 2
    edge_radius = np.linalg.norm(
        cp[mid - 1, :, 1:], axis=1
    ).max()  # TODO: Only works for spherical
    scale = cp[:, :, 2] / edge_radius
    scale[scale < 0] = 0

    # Shift each point along backbone
    new_cp = cp.copy()
    num_rows, num_cols = new_cp.shape[:2]
    for i in range(num_rows):

        # Leave endpoints unbent, avoids tearing artifact
        if i in [0, 1, num_rows - 2, num_rows - 1]:
            continue

        for j in range(num_cols):

            point_xyz = cp[i, j]
            point_scale = np.round(scale[i, j], 4)  # Avoid divide by zero by rounding
            pos = point_scale
            if point_scale <= 0:
                new_cp[i, j] = point_xyz
            else:
                dist_to_yz_plane = point_xyz[0]

                # Handle points with scale > 1
                if point_scale > 1:
                    pos = 1.0
                    dist_from_end = (point_scale - 1.0) * max_scale
                else:
                    dist_from_end = 0.0
                new_point = (
                    backbone.r(pos)
                    - backbone.B(pos) * dist_to_yz_plane
                    + backbone.T(pos) * dist_from_end
                )

                new_point[0, 1] = point_xyz[1]  # Keep original y value
                new_cp[i, j] = new_point

    return new_cp


##################
### Parameters ###
##################
SHEET_THICKNESS = 5
NUM_CP = 16
NUM_CS = 21
SEGMENT_LENGTH = 11
X_WIDTH = 3
LEAF_RADII = np.array([X_WIDTH, 3 * X_WIDTH, X_WIDTH])

b_cp = approximate_arc(np.pi / 2, SEGMENT_LENGTH, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
backbone = Backbone(b_cp, reparameterize=True)

##############
### Sheets ###
##############

# Round sheet
t = np.linspace(0, 2 * np.pi, NUM_CP, endpoint=False).reshape(-1, 1)
round_cs_cp = np.hstack([np.zeros(t.shape), np.cos(t), np.sin(t)])
base_sheet = round_cs_cp
cp = construct_sheet(
    base_sheet, SEGMENT_LENGTH, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS
)
surf = make_surface(cp)
sheet_round = make_mesh(surf, 100, 100)

# Bend round sheet
bent_cp = bend_sheet(cp, backbone, SEGMENT_LENGTH)
surf = make_surface(bent_cp)
sheet_round_bent = make_mesh(surf, 100, 100)

# Leaf sheet
num_edge_cp = 7
num_roundover_cp = 3
x = np.linspace(0, 1, 3) * SEGMENT_LENGTH
y = LEAF_RADII
poly = np.polyfit(x, y, 2)
xx = np.linspace(0, SEGMENT_LENGTH, num_edge_cp).reshape(-1, 1)
yy = np.polyval(poly, xx)
leaf_cp = np.hstack([xx, yy])

# Opposite direction
opp = leaf_cp.copy()
opp[:, 1] *= -1

# Roundover
import scipy.interpolate

der = np.polyder(poly, 1)
chs_x = np.array([0, 1])
chs_y = np.concatenate([yy[-1], -yy[-1]])
chs_dydx = np.concatenate([np.polyval(der, xx[-1]), -np.polyval(der, xx[-1])])
chs = scipy.interpolate.CubicHermiteSpline(chs_x, chs_y, chs_dydx)
tt = np.linspace(0, 1, num_roundover_cp + 2)
roundover = chs(tt[1:-1])

leaf_cp = np.vstack([leaf_cp, roundover, opp[::-1]])
import matplotlib.pyplot as plt

ax = plt.figure().add_subplot()
ax.plot(leaf_cp[:, 0], leaf_cp[:, 1], "b-")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# import matplotlib.pyplot as plt

# ax = plt.figure().add_subplot(projection="3d")
# arr = bent_cp
# for i in range(arr.shape[0]):
#     ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")

# # Plot points
# # ax.plot(b_pts[:, 0], b_pts[:, 1], b_pts[:, 2], "r*-")

# # Set scale
# xs = arr[:, :, 0].ravel()
# ys = arr[:, :, 1].ravel()
# zs = arr[:, :, 2].ravel()
# ax.set_box_aspect(
#     (np.ptp(xs), np.ptp(ys), np.ptp(zs))
# )  # aspect ratio is 1:1:1 in data space
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.show()
