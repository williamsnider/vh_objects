# Find the 5 CP that most closely approximate a circular arc, using radius/distance from origin as the error.
from scipy.optimize import minimize
from objects.backbone import Backbone
import numpy as np


def approximate_arc(MAX_ANGLE, GOAL_LENGTH_SEGMENT, NUM_CP_PER_SEGMENT=5):
    """Construct a B-Spline curve that approximates a circular arc."""

    radius = GOAL_LENGTH_SEGMENT / (2 * np.pi) * (2 * np.pi / MAX_ANGLE)

    if NUM_CP_PER_SEGMENT != 5:
        raise NotImplementedError("NUM_CP_PER_SEGMENT must be 5 for this funtion to work.")

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


if __name__ == "__main__":
    cp_arc_1_4 = approximate_arc(np.pi / 4)
    cp_arc_1_8 = approximate_arc(np.pi / 8)

# # Make backbone and plot
# backbone = Backbone(cp_arc_1_4, reparameterize=False)
# MAX_ANGLE = np.pi / 4
# radius = GOAL_LENGTH_SEGMENT / (2 * np.pi) * (2 * np.pi / MAX_ANGLE)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# cp = backbone.controlpoints
# maxcp = cp.max()
# ax.set_xlim([-0, maxcp])
# ax.set_ylim([-0, maxcp])
# ax.set_zlim([-0, maxcp])

# t = np.linspace(0, 1, 1000)
# x, y, z = backbone.r(t).T
# ax.plot(x, y, z, "b-")
# x, y, z = backbone.controlpoints.T
# ax.plot(x, y, z, "g-")

# u = np.linspace(0, MAX_ANGLE, 100)
# x = radius * np.sin(u)
# y = radius * (1 - np.cos(u))
# z = np.zeros(u.shape)
# ax.plot(x, y, z, "r.")

# plt.show()
