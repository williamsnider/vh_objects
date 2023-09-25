# Make a hemisphere out of controlpoints
import numpy as np
from objects.utilities import (
    open_uniform_knot_vector,
    BSplineBasis,
    Curve,
    approximate_arc,
    make_surface,
    make_mesh,
)
from objects.parameters import ORDER
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from objects.backbone import Backbone


def calc_sphere_controlpoints(base_cp, num_cp, tan_vec, endpoint, x):

    arc_length = np.pi
    r = 1

    # Calculate and transform the controlpoints based on which side of the axial component we are
    cp = approximate_arc(arc_length, r * arc_length, num_cp)
    cp_T = cp[:, :2] + np.array([0, r])  # Align to origin
    cp_T[:, 1] = cp_T[::-1, 1]  # Flip order of translation

    # Plot samples
    # fig, ax = plt.subplots()

    # ax.plot(cp_rot[:, 0], cp_rot[:, 1], "r-*")
    # ax.set_aspect("equal")
    # plt.show()

    # if x == 0:
    #     cp_rot[:, 0] *= -1
    # cp_T = cp_rot - np.array(
    #     [cp_rot[-1, 0] - x_shift, 0]
    # )  # Shift to align with end of quadratic

    scale_ratio = cp_T[:, 0]  # Normalize this column by the height of the quadratic

    ### Apply these transformations to base_cp ###

    # Rotate base_cp to be in yz plane
    vecA = base_cp[0] - base_cp[1]
    vecB = base_cp[0] - base_cp[2]
    N = np.cross(vecA, vecB) / np.linalg.norm(np.cross(vecA, vecB))
    assert np.all(np.isclose(N, tan_vec)) or np.all(np.isclose(N, -tan_vec))
    T = np.cross(vecA, N) / np.linalg.norm(np.cross(vecA, N))
    B = np.cross(T, N)
    curr = np.vstack([N.reshape(1, -1), T.reshape(1, -1), B.reshape(1, -1)])
    goal = np.eye(3)
    R = (goal @ np.linalg.inv(curr.T)).T
    yz_cp = (base_cp - endpoint) @ R

    # Scale base_cp
    out_cp = np.tile(yz_cp, (num_cp, 1, 1))
    cp_scale = out_cp * scale_ratio.reshape(-1, 1, 1)
    cp_scale[:, :, 0] = yz_cp[:, 0]

    # Translate base_cp
    vec_rotated = tan_vec @ R
    cp_shift = cp_scale + vec_rotated * (cp_T[:, 1].reshape(-1, 1, 1)) - np.array([x, 0, 0])
    result = (cp_shift) @ R.T + endpoint
    return result


def calc_hemisphere_controlpoints(base_cp, tan_vec, endpoint, poly, x, num_cp):
    """Calculates the controlpoints needed to approximate a hemispherical ending to an axial component.

    This works by calculating a 5-controlpoint arc that will connect a quadratic curve (poly), resulting in a hemisphere shape. This arc serves as the scale (first column) and translation (second column) that are applied to base_cp."""

    ### Calculate the controlpoint arc that approximates a hemisphere ###

    # Solve for a and r (of the fitting circle) given m and quadratic polynomial
    y = np.polyval(poly, x)
    der = np.polyder(poly, 1)
    m = np.polyval(der, x)
    a = x + y * m  # Solve for a
    r = np.sqrt((x - a) ** 2 + y**2)  # Solve for r

    # Calculate the theta and arc length of the arc
    if x == 0:
        th = np.arctan2(y, (x - a))
        arc_length = np.pi - th
        x_shift = 0
    else:
        th = np.arctan2(y, (x - a))
        arc_length = th
        x_shift = x

    # Calculate and transform the controlpoints based on which side of the axial component we are
    cp = approximate_arc(arc_length, r * arc_length, num_cp)
    cp_rot = cp[:, [1, 0]]  # Rotate 45deg
    if x == 0:
        cp_rot[:, 0] *= -1
    cp_T = cp_rot - np.array([cp_rot[-1, 0] - x_shift, 0])  # Shift to align with end of quadratic

    scale_ratio = cp_T[:, 1] / y  # Normalize this column by the height of the quadratic

    ### Apply these transformations to base_cp ###

    # Rotate base_cp to be in yz plane
    vecA = base_cp[0] - base_cp[1]
    vecB = base_cp[0] - base_cp[2]
    N = np.cross(vecA, vecB) / np.linalg.norm(np.cross(vecA, vecB))
    assert np.all(np.isclose(N, tan_vec)) or np.all(np.isclose(N, -tan_vec))
    T = np.cross(vecA, N) / np.linalg.norm(np.cross(vecA, N))
    B = np.cross(T, N)
    curr = np.vstack([N.reshape(1, -1), T.reshape(1, -1), B.reshape(1, -1)])
    goal = np.eye(3)
    R = (goal @ np.linalg.inv(curr.T)).T
    yz_cp = (base_cp - endpoint) @ R

    # Scale base_cp
    out_cp = np.tile(yz_cp, (num_cp, 1, 1))
    cp_scale = out_cp * scale_ratio.reshape(-1, 1, 1)
    cp_scale[:, :, 0] = yz_cp[:, 0]

    # Translate base_cp
    vec_rotated = tan_vec @ R
    cp_shift = cp_scale + vec_rotated * (cp_T[:, 0].reshape(-1, 1, 1)) - np.array([x, 0, 0])
    result = (cp_shift) @ R.T + endpoint

    assert np.all(np.isclose(result[-1], base_cp)), "base_cp not aligned with result[-1]"

    # # Plot everything
    # fig, ax = plt.subplots()
    # t = np.linspace(0, 1, 100)
    # vals = np.polyval(poly, t)
    # ax.plot(t, vals)
    # # ax.plot(xx, yy, "-k")  # Arc
    # ax.plot(x, y, "*r")  # Intersection point
    # # ax.plot(new_cp[:, 0], new_cp[:, 1], "g-")
    # ax.set_aspect("equal")
    # plt.show()

    return result


if __name__ == "__main__":

    num_cp = 9

    t = np.linspace(0, 2 * np.pi, 8, endpoint=False).reshape(-1, 1)
    round_cp = np.hstack([np.cos(t), np.sin(t)])
    base_cp = np.hstack([np.zeros((round_cp.shape[0], 1)), round_cp])

    # Sphere
    cp = calc_sphere_controlpoints(
        base_cp,
        num_cp,
        np.array([1, 0, 0]),
        np.array([0, 0, 0]),
        x=0,
    )

    thickness = 0.5

    # Squash sphere along z-dimension
    mid = (cp.shape[0] - 1) // 2
    offset = 1
    cp[:mid, :, 0] = -thickness / 2
    cp[mid + offset :, :, 0] = thickness / 2

    # Match radius at center
    cp[mid - 1, :, 1:] = cp[mid, :, 1:]
    cp[mid + 1, :, 1:] = cp[mid, :, 1:]

    # Expand radius at center to round out the transition
    center_radius = np.linalg.norm(cp[mid, :, 1:], axis=1).mean()
    goal_radius = center_radius + thickness / 3
    cp[mid, :, 1:] = cp[mid, :, 1:] / center_radius * goal_radius

    # Bend according to backbone
    edge_radius = np.linalg.norm(cp[mid - 1, :, 1:], axis=1).mean()
    scale = cp[:, :, 2] / edge_radius
    scale[scale < 0] = 0
    b_cp = approximate_arc(np.pi / 2, edge_radius, 5)
    b_cp = b_cp[:, [1, 2, 0]]  # Reorder
    b_cp[:, 0] *= -1  # Flip direction across yz axis
    b = Backbone(b_cp, reparameterize=True)
    b_pts = b.r(np.linspace(0, 1))

    new_cp = cp.copy()
    num_rows, num_cols = new_cp.shape[:2]
    for i in range(num_rows):
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
                    dist_from_end = point_scale - 1.0
                else:
                    dist_from_end = 0.0
                new_point = b.r(pos) - b.B(pos) * dist_to_yz_plane + b.T(pos) * dist_from_end
                if np.any(np.isnan(new_point)):
                    print("here")
                new_point[0, 1] = point_xyz[1]  # Keep original y value
                new_cp[i, j] = new_point

    # Remove artifact from endpoints
    new_cp[[0, 1, -2, -1], :, :] = cp[[0, 1, -2, -1], :, :]

    # Scale up
    new_cp = new_cp * 5

    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection="3d")
    arr = new_cp
    for i in range(arr.shape[0]):
        ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")

    # Plot points
    ax.plot(b_pts[:, 0], b_pts[:, 1], b_pts[:, 2], "r*-")

    # Set scale
    xs = arr[:, :, 0].ravel()
    ys = arr[:, :, 1].ravel()
    zs = arr[:, :, 2].ravel()
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))  # aspect ratio is 1:1:1 in data space
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    surf = make_surface(new_cp)
    mesh = make_mesh(surf, 100, 100)
    mesh.show(smooth=False)

    # # Hemisphere
    tan_vec = np.array([1, 0, 0])
    endpoint = np.array([0, 0, 0])
    xx = np.array([0, 0.5, 1])
    yy = np.array([1, 1, 1])
    poly = np.polyfit(xx, yy, 2)
    x = 0
    cp = calc_hemisphere_controlpoints(base_cp, tan_vec, endpoint, poly, x, num_cp)

    # Make mesh
    hemi = np.vstack([cp, cp[-1:] * 0])
    surf = make_surface(hemi)
    mesh = make_mesh(surf, 100, 100)
    mesh.show(smooth=False)

    pass
    # # Calculate an arc that makes a hemisphere
    # angle = np.pi / 2
    # radius = 3
    # num_cp = 5
    # arc_array = approximate_arc(angle, angle * radius, num_cp)

    # # Sample and plot arc
    # knot = open_uniform_knot_vector(arc_array.shape[0], ORDER)
    # basis = BSplineBasis(order=ORDER, knots=knot, periodic=-1)
    # curve = Curve(basis=basis, controlpoints=arc_array, rational=False)

    # t = np.linspace(0, 1, 100)
    # r = curve(t)

    # # Plot comparison arc
    # tt = np.linspace(0, angle)
    # xx = radius * np.sin(tt)
    # yy = radius * np.cos(tt) - radius  # Shift downward

    # # Plot samples
    # fig, ax = plt.subplots()
    # ax.plot(r[:, 0], r[:, 1], "k-")
    # ax.plot(xx, yy, "b-")
    # ax.plot(arc_array[:, 0], arc_array[:, 1], "r*")
    # ax.set_aspect("equal")
    # plt.show()
