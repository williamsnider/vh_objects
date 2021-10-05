# General functions used by different classes for objects project

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.linalg import norm

##########
# B-Spline Functions


def open_uniform_knot_vector(num_cps, order):

    num_knots = num_cps + order
    knots = np.zeros(num_knots)
    knots[order:-order] = range(1, num_knots - 2 * order + 1)
    knots[-order:] = knots[-order - 1] + 1
    knots = knots / knots.max()  # Scale from 0 to 1
    return knots


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
    ax.view_init(elev=-90, azim=90)

    # Entire mesh
    x, y, z = mesh.vertices[::spacing].T
    ax.plot(x, y, z, ".", color="green")

    # Vertices in the group
    x, y, z = mesh.vertices[specific_indices].T
    ax.plot(x, y, z, ".", color="blue")

    plt.show()
