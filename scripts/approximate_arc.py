# Find the 5 CP that most closely approximate a circular arc, using radius/distance from origin as the error.
from objects.parameters import NUM_CP_PER_SEGMENT, GOAL_LENGTH_SEGMENT
from scipy.optimize import minimize
from objects.backbone import Backbone
import numpy as np
import matplotlib.pyplot as plt

# """
#
#
# "

MAX_ANGLE = np.pi
radius = GOAL_LENGTH_SEGMENT / (2 * np.pi) * (2 * np.pi / MAX_ANGLE)


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

print(arc_array)
# Make backbone and plot
backbone = Backbone(arc_array, reparameterize=False)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
cp = backbone.controlpoints
maxcp = cp.max()
ax.set_xlim([-0, maxcp])
ax.set_ylim([-0, maxcp])
ax.set_zlim([-0, maxcp])

t = np.linspace(0, 1, 1000)
x, y, z = backbone.r(t).T
ax.plot(x, y, z, "b-")
x, y, z = backbone.controlpoints.T
ax.plot(x, y, z, "g-")

u = np.linspace(0, MAX_ANGLE, 100)
x = radius * np.cos(u)
y = radius * np.sin(u)
z = np.zeros(u.shape)
ax.plot(x, y, z, "r.")

plt.show()
