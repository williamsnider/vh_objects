# General functions used by different classes for objects project

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

##########
# B-Spline Functions


def open_uniform_knot_vector(num_cps, order):

    num_knots = num_cps + order
    knots = np.zeros(num_knots)
    knots[order:-order] = range(1, num_knots - 2 * order + 1)
    knots[-order:] = knots[-order - 1] + 1
    knots = knots / knots.max()  # Scale from 0 to 1
    return knots


def approximate_arc(MAX_ANGLE, arc_length):
    """Construct a B-Spline curve that approximates a circular arc."""

    radius = arc_length / (2 * np.pi) * (2 * np.pi / MAX_ANGLE)

    # if NUM_CP_PER_SEGMENT != 5:
    #     raise NotImplementedError("NUM_CP_PER_SEGMENT must be 5 for this funtion to work.")

    def make_arc_array(a, b, c):

        # We can think of the second to last arc controlpoint as lying along a vector from the last controlpoint. The vector's slope can be determined from the tangent line of the circle (which is negated). We then can use a single parameter (d) as a measure of how far along this vector we are travelling. This reduces the number of parameters we need, and also ensures that the tangent of the resulting arc at the end will match that of the circle

        def tan_vec(MAX_ANGLE):
            tangent_vec = np.array(
                [
                    -radius * np.sin(MAX_ANGLE),
                    radius * np.cos(MAX_ANGLE),
                ]
            )
            return tangent_vec

        start_tan_vec = tan_vec(0)
        end_tan_vec = -tan_vec(MAX_ANGLE)  # Negate so this points toward start

        arc_array = np.array(
            [
                [radius, 0, 0],
                [
                    radius * np.cos(0) + a * start_tan_vec[0],
                    radius * np.sin(0) + a * start_tan_vec[1],
                    0,
                ],  # Tangent line from start
                [b, c, 0],
                [
                    radius * np.cos(MAX_ANGLE) + a * end_tan_vec[0],
                    radius * np.sin(MAX_ANGLE) + a * end_tan_vec[1],
                    0,
                ],  # Tangent line from end
                [radius * np.cos(MAX_ANGLE), radius * np.sin(MAX_ANGLE), 0],
            ]
        )
        return arc_array

    def radius_error(vars):

        # Make the arc array
        [a, b, c] = vars
        arc_array = make_arc_array(a, b, c)

        # Make backbone
        backbone = Backbone(arc_array, reparameterize=False)

        # Sample points along the backbone
        t = np.linspace(0, 1, 10)
        r = backbone.r(t)

        # distance from origin should be close to radius if points are well_aligned
        dist = np.linalg.norm(r, axis=1)
        return ((dist - radius) ** 2).sum()

    fun = radius_error
    x0 = [0.1, radius, radius]
    bounds = [
        [0.0, 100 * radius],
        [radius * np.cos(MAX_ANGLE / 2), 100 * radius],  # Convex hull property of B-Splines
        [radius * np.sin(MAX_ANGLE / 2), 100 * radius],  # Convex hull property of B-Splines
    ]
    result = minimize(fun=fun, x0=x0, bounds=bounds)
    [a, b, c] = result.x

    arc_array = make_arc_array(a, b, c)

    # Shift so that the curve begins at the origin
    arc_array[:, 0] -= radius
    arc_array[:, [0, 1]] = arc_array[:, [1, 0]]  # Flip x and y-axis so long portion points in +X direction
    arc_array[:, 1] = -arc_array[:, 1]  # Negate y axis so curves upward (towards +Y)

    return arc_array


##########
# Vector functions
def unit_vector(vector):
    """Returns the unit vector of the vector."""
    a = vector.ndim - 1
    return vector / np.linalg.norm(vector, axis=a, keepdims=True)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    Args:
        v1 (array): Vector 1.
        v2 (array): Vector 2
    Returns:
        angle (float): Radians
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    a = np.max([v1_u.ndim, v2_u.ndim]) - 1
    return np.arccos(np.clip((v1_u * v2_u).sum(axis=a), -1.0, 1.0))


def calc_R_euler_angles(euler_angles):
    s = np.sin
    c = np.cos

    a1, a2, a3 = euler_angles
    R_x = np.array(
        [
            [1, 0, 0],
            [0, c(a1), -s(a1)],
            [0, s(a1), c(a1)],
        ]
    )

    R_y = np.array(
        [
            [c(a2), 0, s(a2)],
            [0, 1, 0],
            [-s(a2), 0, c(a2)],
        ]
    )

    R_z = np.array(
        [
            [c(a3), -s(a3), 0],
            [s(a3), c(a3), 0],
            [0, 0, 1],
        ]
    )

    R_euler_angles = R_z @ (R_y @ R_x)
    return R_euler_angles


##########
# Mesh Functions
def calc_face_normals(verts, faces):
    p0 = verts[faces[:, 0]]
    p1 = verts[faces[:, 1]]
    p2 = verts[faces[:, 2]]

    # Subtract to form vectors
    vec0 = p1 - p0
    vec1 = p2 - p0

    # Calculate cross product
    cross = np.cross(vec0, vec1)
    cross /= np.linalg.norm(cross, axis=1, keepdims=True)
    face_norms = cross

    return face_norms


##########
# Misc Functions
def flatten(groups):
    flattened = []
    for sublist in groups:
        for i in sublist:
            flattened.append(i)
    return flattened


def sliding_window_mean(arr, window_size, axis):

    assert window_size % 2 == 1, "window_size must be odd."

    big_arr = np.zeros(arr.shape + (window_size,))  # Add extra dimension along which we will average
    for idx in range(window_size):

        shift = (window_size - 1) // 2 - idx
        shifted = np.roll(arr, shift=shift, axis=axis)
        big_arr[..., idx] = shifted  # Ellipsis in python --> get last column. COOL!
        # big_arr[[slice(None)] * (big_arr.ndim - 1) + [idx]] = shifted  # More general solution

    return big_arr.mean(axis=-1)  # Average along the axis we added (the last one)


##########
# Plotting helper functions


def plot_mesh_and_specific_indices(
    mesh,
    specific_indices,
    spacing=1,
):

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("blue = specific indices")
    ax.view_init(elev=-90, azim=90)

    # Entire mesh
    x, y, z = mesh.vertices[::spacing].T
    ax.plot(x, y, z, ".", color="green")

    # Vertices in the group
    x, y, z = mesh.vertices[specific_indices].T
    ax.plot(x, y, z, ".", color="blue")

    plt.show()


def plot_controlpoints(ac):

    # Controlpoints
    cp = ac.controlpoints
    x = cp[:, :, 0].ravel()
    y = cp[:, :, 1].ravel()
    z = cp[:, :, 2].ravel()

    # Backbone
    v = np.linspace(0, 1, 51)
    r = ac.r(v)

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax_min = cp.min()
    ax_max = cp.max()
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.set_zlim([ax_min, ax_max])
    ax.view_init(elev=-90, azim=90)
    ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "k.")
    ax.plot3D(x, y, z, "g-")
    plt.show()
