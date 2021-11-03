# Contains the digit segments and cross sections we will use to build our shapes.

import numpy as np
from objects.backbone import Backbone
from objects.parameters import NUM_CP_PER_SEGMENT, GOAL_LENGTH_SEGMENT
from scipy.optimize import minimize


def approximate_arc(MAX_ANGLE):
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


####################
### Digit Segments
####################

### Flat
cp_flat = np.array(
    [
        np.linspace(0, GOAL_LENGTH_SEGMENT, NUM_CP_PER_SEGMENT),
        np.zeros(5),
        np.zeros(5),
    ]
).T
segment_flat = Backbone(cp_flat, reparameterize=False)
assert np.isclose(segment_flat.length(), GOAL_LENGTH_SEGMENT), "Arc segment not close to length GOAL_LENGTH_SEGMENT."

### 1/8 circle
angle = np.pi / 4
t = np.linspace(0, angle, NUM_CP_PER_SEGMENT)
cp_arc_1_4 = approximate_arc(angle)
segment_arc_1_4 = Backbone(cp_arc_1_4, reparameterize=False)
assert np.isclose(segment_arc_1_4.length(), GOAL_LENGTH_SEGMENT), "Arc segment not close to length GOAL_LENGTH_SEGMENT."

### 1/16 circle
angle = np.pi / 8
t = np.linspace(0, angle, NUM_CP_PER_SEGMENT)
cp_arc_1_8 = approximate_arc(angle)
segment_arc_1_8 = Backbone(cp_arc_1_8, reparameterize=False)
assert np.isclose(segment_arc_1_8.length(), GOAL_LENGTH_SEGMENT), "Arc segment not close to length GOAL_LENGTH_SEGMENT."


####################
### Cross Sections
####################
c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
    ]
)

# concave_high
cp_concave_high = base_cp.copy()
cp_concave_high[0, :] = [0.001, 0.001]
cp_concave_high = cp_concave_high * GOAL_LENGTH_SEGMENT

# concave_low
cp_concave_low = base_cp.copy()
cp_concave_low[0, :] = [0.5, 0]
cp_concave_low = cp_concave_low * GOAL_LENGTH_SEGMENT

# round
cp_round = base_cp.copy()
cp_round = cp_round * GOAL_LENGTH_SEGMENT

# convex_low
cp_convex_low = base_cp.copy()
cp_convex_low[0, :] = [1.33, 0]
cp_convex_low = cp_convex_low * GOAL_LENGTH_SEGMENT

# convex_med
cp_convex_med = base_cp.copy()
cp_convex_med[0, :] = [1.66, 0]
cp_convex_med = cp_convex_med * GOAL_LENGTH_SEGMENT

# convex_high
cp_convex_high = base_cp.copy()
cp_convex_high[0, :] = [2, 0]
cp_convex_high = cp_convex_high * GOAL_LENGTH_SEGMENT

# concave_point
cp_concave_point = base_cp.copy()
cp_concave_point[:2, :] = [c(1 / 12 * 2 * np.pi) * 0.001, s(1 / 12 * 2 * np.pi) * 0.001]
cp_concave_point = cp_concave_point * GOAL_LENGTH_SEGMENT

# convex_point_low
cp_convex_point_low = base_cp.copy()
cp_convex_point_low[:2, :] = [c(1 / 12 * 2 * np.pi) * 2 / 3, s(1 / 12 * 2 * np.pi) * 2 / 3]
cp_convex_point_low = cp_convex_point_low * GOAL_LENGTH_SEGMENT

# convex_point_high
cp_convex_point_high = base_cp.copy()
cp_convex_point_high[:2, :] = [c(1 / 12 * 2 * np.pi) * 5 / 4, s(1 / 12 * 2 * np.pi) * 5 / 4]
cp_convex_point_high = cp_convex_point_high * GOAL_LENGTH_SEGMENT
