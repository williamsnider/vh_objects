# Construct sheets needed in experiment
import numpy as np
from vh_objects.utilities import make_mesh, make_surface, approximate_arc, angle_between
from vh_objects.backbone import Backbone


def plot_arr(cp):
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection="3d")
    arr = cp

    if arr.ndim == 2:
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], "b-*")
        xs = arr[:, 0].ravel()
        ys = arr[:, 1].ravel()
        zs = arr[:, 2].ravel()
    elif arr.ndim == 3:
        for i in range(arr.shape[0]):
            ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")
        xs = arr[:, :, 0].ravel()
        ys = arr[:, :, 1].ravel()
        zs = arr[:, :, 2].ravel()

    # Set scale
    x_mid = (xs.max() + xs.min()) / 2
    y_mid = (ys.max() + ys.min()) / 2
    z_mid = (zs.max() + zs.min()) / 2
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    z_range = zs.max() - zs.min()

    dim = max(x_range, y_range, z_range)

    # for i in range(arr.shape[1]):
    #     ax.plot(arr[:, i, 0], arr[:, i, 1], arr[:, i, 2], "g-")
    ax.set_xlim([x_mid - dim / 2, x_mid + dim / 2])
    ax.set_ylim([y_mid - dim / 2, y_mid + dim / 2])
    ax.set_zlim([z_mid - dim / 2, z_mid + dim / 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def construct_sheet(base_sheet, sheet_thickness, num_cs):
    """Constructs a b-spline surface in the form of a sheet."""

    # Tile copies of base_sheet
    cp = np.tile(base_sheet, (num_cs, 1, 1))
    mid_idx = (num_cs - 1) // 2

    # Replace base sheet in middle with offset (rounds according to sheet thickness)
    arr = base_sheet[:, 1:]
    vecA = arr - np.roll(arr, -1, axis=0)
    vecA = vecA / np.linalg.norm(vecA, axis=1, keepdims=True)
    vecB = arr - np.roll(arr, 1, axis=0)
    vecB = vecB / np.linalg.norm(vecB, axis=1, keepdims=True)
    vecBisect = vecA + vecB

    # Handle straight sides - bisecting vector is zero
    vecBisect[np.all(vecBisect == 0, axis=1)] = arr[np.all(vecBisect == 0, axis=1)]  # Axis from (0,0) to point

    # Normalize bisecting vector
    vecBisect = vecBisect / np.linalg.norm(vecBisect, axis=1, keepdims=True)
    vecC = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    dotproduct = np.sum(vecBisect * vecC, axis=1)
    vecBisect[dotproduct < 0] *= -1  # Flip direction if pointing towards origin
    offset = arr + vecBisect * sheet_thickness / 3
    offset = np.hstack([np.zeros((offset.shape[0], 1)), offset])  # Add back in x-axis
    cp[mid_idx] = offset  # replace base_sheet in middle

    # Scale other profiles
    scale = np.linspace(0, 1, mid_idx - 1)
    scale = np.insert(scale, 1, scale[1] / 5)  # Controls slope at endpoint
    scale = np.concatenate([scale, np.array([1]), scale[::-1]])
    cp *= scale.reshape(-1, 1, 1)

    # Translate to create volume along x-axis
    cp[:mid_idx, :, 0] -= sheet_thickness / 2
    cp[mid_idx + 1 :, :, 0] += sheet_thickness / 2

    # plot_arr(cp)
    # import matplotlib.pyplot as plt

    # ax = plt.figure().add_subplot()
    # ax.plot(arr[:, 0], arr[:, 1], "-b*")
    # ax.plot(offset[:, 1], offset[:, 2], "-g*")

    # for i in range(arr.shape[0]):
    #     point1 = arr[i]
    #     vec = vecBisect[i]
    #     point2 = point1 + vec
    #     pts = np.vstack([point1, point2])
    #     ax.plot(pts[:, 0], pts[:, 1], "-r*")
    # # pts = np.vstack(
    # #     [arr[1] + vecB[1], arr[1], arr[1] + vecA[1], arr[1], arr[1] + vecBisect[1]]
    # # )
    # # ax.plot(pts[:, 0], pts[:, 1])
    # ax.set_aspect("equal")
    # plt.show()

    return cp


def bend_sheet(cp, backbone, max_scale, K_perpendicular=0):

    assert np.all(np.isclose(backbone.r(0.0), np.array([0.0, 0.0, 0.0])))
    assert np.isclose(backbone.r(0.0)[0, 1], 0)  # Must lie in yz-plane

    # # Calc relative distance each point should travel along backbone
    # mid = (cp.shape[0] - 1) // 2
    # edge_radius = np.linalg.norm(
    #     cp[mid - 1, :, 1:], axis=1
    # ).max()  # TODO: Only works for spherical
    # scale = cp[:, :, 2] / edge_radius
    # scale[scale < 0] = 0

    scale = cp[:, :, 2] / max_scale
    scale[scale < 0] = 0

    # Shift each point along backbone
    new_cp = cp.copy()
    num_rows, num_cols = new_cp.shape[:2]
    for i in range(num_rows):

        # # Leave endpoints unbent, avoids tearing artifact
        # if i in [0, 1, num_rows - 2, num_rows - 1]:
        #     continue

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

                new_point = backbone.r(pos) - backbone.B(pos) * dist_to_yz_plane + backbone.T(pos) * dist_from_end

                # Bend along perpendicular axis
                dist_to_xz_plane = point_xyz[1]
                if K_perpendicular == 0:
                    theta = 0
                else:
                    theta = dist_to_xz_plane * K_perpendicular

                new_point += (
                    backbone.B(pos) * np.sin(theta) * dist_to_xz_plane
                    + backbone.N(pos) * np.cos(theta) * dist_to_xz_plane
                )

                if i == 7 and j == 7:
                    print("here")

                # new_point[0, 1] = point_xyz[1]  # Keep original y value
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


def make_base_cp(poly, x, num_edge_cp, base_round_cp, top_round_cp):

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
    right_round_cp = calc_roundover_cp(rm, rx, ry, top_round_cp, "right")

    # Left roundover
    rx = x[0]
    der = np.polyder(poly, 1)
    ry = np.polyval(poly, rx)
    rm = np.polyval(der, rx)
    left_round_cp = calc_roundover_cp(rm, rx, ry, base_round_cp, "left")

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
