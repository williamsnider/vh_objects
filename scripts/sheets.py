# Construct sheets needed in experiment
import numpy as np
from objects.utilities import make_mesh, make_surface, approximate_arc
from objects.backbone import Backbone


def plot_arr(cp):
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection="3d")
    arr = cp
    for i in range(arr.shape[0]):
        ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")
    # Set scale
    xs = arr[:, :, 0].ravel()
    ys = arr[:, :, 1].ravel()
    zs = arr[:, :, 2].ravel()
    ax.set_box_aspect(
        (np.ptp(xs), np.ptp(ys), np.ptp(zs))
    )  # aspect ratio is 1:1:1 in data space
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def construct_sheet(base_sheet, sheet_thickness, num_cs, edge_prop):
    """Constructs a b-spline surface in the form of a sheet.

    scale_max is that maximum amount that base_sheet is multiplied by. For example, if base_sheet is a circle with radius 1, and you ultimately want a surface with radius 11, scale_max should be 11."""

    # Edges (adjacent to roundover of sheet) - shrink by factor of sheet thicknes
    edge = base_sheet * edge_prop

    # plot_arr(np.vstack([edge.reshape(1, -1, 3), base_sheet.reshape(1, -1, 3)]))

    # Tile copies of edge
    cp = np.tile(edge, (num_cs, 1, 1))
    mid_idx = (num_cs - 1) // 2
    cp[mid_idx] = base_sheet  # replace base_sheet in middle

    # Scale other profiles
    scale = np.linspace(0, 1, mid_idx - 1)
    scale = np.insert(scale, 1, scale[1] / 5)  # Controls slope at endpoint
    scale = np.concatenate([scale, np.array([1]), scale[::-1]])
    cp *= scale.reshape(-1, 1, 1)

    # Translate to create volume along x-axis
    cp[:mid_idx, :, 0] -= sheet_thickness / 2
    cp[mid_idx + 1 :, :, 0] += sheet_thickness / 2

    # # Scale to create progression from endpoint to middle to endpoint
    # mid_idx = (num_cs - 1) // 2
    # scale = np.linspace(0.0, scale_max, mid_idx - 1)
    # scale = np.insert(scale, 1, scale[1] / 5)  # Controls slope at endpoint
    # scale = np.concatenate(
    #     [scale, np.array([scale_max + sheet_thickness / 3]), scale[::-1]]
    # )
    # cp *= scale.reshape(-1, 1, 1)

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


def calc_roundover_cp(dydx, x, y, num_cp, side="left"):

    assert side in ["left", "right"]

    # Find circle that matches the given parameters
    m = dydx
    a = x + y * m  # Solve for a
    r = np.sqrt((x - a) ** 2 + y**2)  # Solve for r

    # Adjust based on left or right side of circle
    th = np.arctan2(y, (x - a))
    if side == "left":
        rt = np.linspace(2 * np.pi - th, th, num_cp + 2).reshape(-1, 1)[1:-1]
    elif side == "right":
        rt = np.linspace(th, -th, num_cp + 2).reshape(-1, 1)[1:-1]

    # Sample controlpoints along this circle
    rx = r * np.cos(rt) + a
    ry = r * np.sin(rt)
    roundover_cp = np.hstack([rx, ry])

    return roundover_cp


def make_base_cp(poly, x, num_edge_cp, num_round_cp):

    assert len(x) == 3, "x must have 3 values for quadratic evaluation."

    # Sample polynomial in forward direction
    xx = np.linspace(x[0], x[-1], num_edge_cp).reshape(-1, 1)
    yy = np.polyval(poly, xx)
    top_cp = np.hstack([xx, yy])

    # Flip top_cp to get bottom / opposite direction
    bot_cp = top_cp.copy()
    bot_cp[:, 1] *= -1

    # Right roundover
    rx = x[-1]
    der = np.polyder(poly, 1)
    ry = np.polyval(poly, rx)
    rm = np.polyval(der, rx)
    right_round_cp = calc_roundover_cp(rm, rx, ry, num_round_cp, "right")

    # Left roundover
    rx = x[0]
    der = np.polyder(poly, 1)
    ry = np.polyval(poly, rx)
    rm = np.polyval(der, rx)
    left_round_cp = calc_roundover_cp(rm, rx, ry, num_round_cp, "left")

    # Combine into aray
    base_cp = np.vstack([top_cp, right_round_cp, bot_cp[::-1], left_round_cp])

    # Reverse order to fix winding
    base_cp = base_cp[::-1]

    # Rotate 90 degrees to align with expected format of bspline surface
    th = -np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    base_cp = base_cp @ R

    # Add z-axis
    base_cp = np.hstack([np.zeros((base_cp.shape[0], 1)), base_cp])

    # plot_arr(base_cp.reshape(1, -1, 3))
    return base_cp


##################
### Parameters ###
##################
