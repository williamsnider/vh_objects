# Distribute controlpoints so that the ending is hemispherical

from scripts import approximate_arc
from scipy.optimize import minimize
import numpy as np
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent


def plot_array(arr):
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection="3d")
    for i in range(arr.shape[0]):
        pts = arr[i]

        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-b*")
    ax.set_aspect("auto")
    plt.show()


def make_arc(radius):
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
        end_tan_vec = -tan_vec(np.pi / 2)  # Negate so this points toward start

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
                    radius * np.cos(np.pi / 2) + a * end_tan_vec[0],
                    radius * np.sin(np.pi / 2) + a * end_tan_vec[1],
                    0,
                ],  # Tangent line from end
                [radius * np.cos(np.pi / 2), radius * np.sin(np.pi / 2), 0],
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
        [radius * np.cos(np.pi / 2 / 2), 100 * radius],  # Convex hull property of B-Splines
        [radius * np.sin(np.pi / 2 / 2), 100 * radius],  # Convex hull property of B-Splines
    ]
    result = minimize(fun=fun, x0=x0, bounds=bounds)
    [a, b, c] = result.x

    arc_array = make_arc_array(a, b, c)

    # Shift so that the curve begins at the origin
    arc_array[:, 0] -= radius
    arc_array[:, [0, 1]] = arc_array[:, [1, 0]]  # Flip x and y-axis so long portion points in +X direction
    arc_array[:, 1] = -arc_array[:, 1]  # Negate y axis so curves upward (towards +Y)

    return arc_array


# Calculate arc
out1 = make_arc(1)
vec_frac = out1[:, 0].reshape(-1, 1, 1)


def calc_cp_hemisphere(base_cp, endpoint, tan_vec, radius):

    # Rotate base_cp to line in yz plane
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

    cp = np.tile(yz_cp, (5, 1, 1))
    cp_scale = cp * out1[::-1, 0].reshape(-1, 1, 1)
    cp_scale[:, :, 0] = yz_cp[:, 0]

    vec_rotated = tan_vec @ R
    cp_shift = cp_scale + vec_rotated * vec_frac * radius

    # Transform back to original plane
    result = (cp_shift) @ R.T + endpoint

    return result


if __name__ == "__main__":

    # Calculate controlpoints
    c = np.cos
    s = np.sin

    t = np.linspace(0, 2 * np.pi, 8, endpoint=False).reshape(-1, 1)
    base_cp = np.hstack([c(t), s(t), np.zeros(t.shape)])

    # Create test axial component
    b_cp = make_arc(15)
    b = Backbone(b_cp)
    NUM_CS = 11
    scale = np.geomspace(0.25, 1.5, NUM_CS)
    pos = np.linspace(0, 1, NUM_CS)
    cs_list = [CrossSection(scale[i] * base_cp[:, :2], pos[i]) for i in range(NUM_CS)]
    ac = AxialComponent(b, cs_list)
    # ac.mesh.show()

    # Adjust controlpoints
    orig_cp = ac.controlpoints
    center_cp = orig_cp[2:-2]

    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection="3d")

    arr = center_cp
    for i in range(arr.shape[0]):
        pts = arr[i]
        # # b = Backbone(out)
        # t = np.linspace(0, 1, 100)
        # pts = b.r(t)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-b*")
    ax.set_aspect("auto")
    plt.show()

    cp_hemisphere_right = calc_cp_hemisphere(center_cp[-1], ac.r(1.0), ac.T(1.0), scale[-1])
    cp_hemisphere_left = calc_cp_hemisphere(center_cp[0], ac.r(0.0), -ac.T(0.0), scale[0])

    new_cp = np.vstack([cp_hemisphere_left[:0:-1], center_cp, cp_hemisphere_right[1:]])

    import copy

    new_ac = copy.deepcopy(ac)
    new_ac.controlpoints = new_cp
    new_ac.num_rows = new_cp.shape[0]
    new_ac.make_surface()
    new_ac.make_mesh()
    new_ac.mesh.show(smooth=True)
