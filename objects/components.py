# Contains the digit segments and cross sections we will use to build our shapes.

import numpy as np
from objects.backbone import Backbone
from objects.parameters import BACKBONE_LENGTH, BACKBONE_NUM_CP
from scipy.optimize import minimize


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


####################
### Backbones
####################

### Flat
NUM_CP = 3
cp_flat = np.array(
    [
        np.linspace(0, BACKBONE_LENGTH, NUM_CP),
        np.zeros(NUM_CP),
        np.zeros(NUM_CP),
    ]
).T
segment_flat = Backbone(cp_flat, reparameterize=True)

### Weak curve
NUM_CP = 3
CURVE_HEIGHT = BACKBONE_LENGTH / 4
cp_weak_curve = np.array(
    [
        np.linspace(0, BACKBONE_LENGTH, NUM_CP),
        [0, CURVE_HEIGHT, 0],
        np.zeros(NUM_CP),
    ]
).T
segment_weak_curve = Backbone(cp_weak_curve, reparameterize=True)

### Strong curve
NUM_CP = 3
CURVE_HEIGHT = BACKBONE_LENGTH / 2
cp_strong_curve = np.array(
    [
        np.linspace(0, BACKBONE_LENGTH, NUM_CP),
        [0, CURVE_HEIGHT, 0],
        np.zeros(NUM_CP),
    ]
).T
segment_strong_curve = Backbone(cp_strong_curve, reparameterize=True)

### Sharp bend
NUM_CP = 4
BEND_HEIGHT = BACKBONE_LENGTH / 4
BEND_SHARPNESS = 0.05  # Smaller -> sharper
cp_sharp_bend = np.array(
    [
        [0, BACKBONE_LENGTH * (0.5 - BEND_SHARPNESS), BACKBONE_LENGTH * (0.5 + BEND_SHARPNESS), BACKBONE_LENGTH],
        [0, BEND_HEIGHT, BEND_HEIGHT, 0],
        np.zeros(NUM_CP),
    ]
).T
segment_sharp_bend = Backbone(cp_sharp_bend, reparameterize=True)


### Hook
NUM_CP = 4
HOOK_HEIGHT = BACKBONE_LENGTH / 4
cp_hook = np.array(
    [
        [0, BACKBONE_LENGTH - HOOK_HEIGHT * 3, BACKBONE_LENGTH - HOOK_HEIGHT, BACKBONE_LENGTH],
        [0, 0, 0, HOOK_HEIGHT],
        np.zeros(NUM_CP),
    ]
).T
segment_hook = Backbone(cp_hook, reparameterize=True)

### Slight S
NUM_CP = 4
S_PROPORTION = 0.1  # How far up/down the middle controlpoints are. Higher --> sharper curves
cp_weak_s = np.array(
    [
        [0, BACKBONE_LENGTH * 0.25, BACKBONE_LENGTH * 0.75, BACKBONE_LENGTH],
        [0, BACKBONE_LENGTH * S_PROPORTION, -BACKBONE_LENGTH * S_PROPORTION, 0],
        np.zeros(NUM_CP),
    ]
).T
segment_weak_s = Backbone(cp_weak_s, reparameterize=True)

### Strong S
NUM_CP = 4
S_PROPORTION = 0.2  # How far up/down the middle controlpoints are. Higher --> sharper curves
cp_strong_s = np.array(
    [
        [0, BACKBONE_LENGTH * 0.25, BACKBONE_LENGTH * 0.75, BACKBONE_LENGTH],
        [0, BACKBONE_LENGTH * S_PROPORTION, -BACKBONE_LENGTH * S_PROPORTION, 0],
        np.zeros(NUM_CP),
    ]
).T
segment_strong_s = Backbone(cp_strong_s, reparameterize=True)

# ####################
# ### Digit Segments
# ####################

# ### Flat
# cp_flat = np.array(
#     [
#         np.linspace(0, GOAL_LENGTH_SEGMENT, NUM_CP_PER_SEGMENT),
#         np.zeros(5),
#         np.zeros(5),
#     ]
# ).T
# segment_flat = Backbone(cp_flat, reparameterize=False)
# assert np.isclose(segment_flat.length(), GOAL_LENGTH_SEGMENT), "Arc segment not close to length GOAL_LENGTH_SEGMENT."

# ### 1/8 circle
# angle = np.pi / 4
# t = np.linspace(0, angle, NUM_CP_PER_SEGMENT)
# cp_arc_1_4 = approximate_arc(angle)
# segment_arc_1_4 = Backbone(cp_arc_1_4, reparameterize=False)
# assert np.isclose(segment_arc_1_4.length(), GOAL_LENGTH_SEGMENT), "Arc segment not close to length GOAL_LENGTH_SEGMENT."

# ### 1/16 circle
# angle = np.pi / 8
# t = np.linspace(0, angle, NUM_CP_PER_SEGMENT)
# cp_arc_1_8 = approximate_arc(angle)
# segment_arc_1_8 = Backbone(cp_arc_1_8, reparameterize=False)
# assert np.isclose(segment_arc_1_8.length(), GOAL_LENGTH_SEGMENT), "Arc segment not close to length GOAL_LENGTH_SEGMENT."


####################
### Cross Sections
####################
c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 8 * 2 * np.pi), s(0 / 8 * 2 * np.pi)],
        [c(1 / 8 * 2 * np.pi), s(1 / 8 * 2 * np.pi)],
        [c(2 / 8 * 2 * np.pi), s(2 / 8 * 2 * np.pi)],
        [c(3 / 8 * 2 * np.pi), s(3 / 8 * 2 * np.pi)],
        [c(4 / 8 * 2 * np.pi), s(4 / 8 * 2 * np.pi)],
        [c(5 / 8 * 2 * np.pi), s(5 / 8 * 2 * np.pi)],
        [c(6 / 8 * 2 * np.pi), s(6 / 8 * 2 * np.pi)],
        [c(7 / 8 * 2 * np.pi), s(7 / 8 * 2 * np.pi)],
    ]
)

# concave_high
cp_concave_high = base_cp.copy()
cp_concave_high[0, :] = [0.001, 0.001]
cp_concave_high = cp_concave_high * BACKBONE_LENGTH / 2 / 2

# concave_low
cp_concave_low = base_cp.copy()
cp_concave_low[0, :] = [0.25, 0]
cp_concave_low = cp_concave_low * BACKBONE_LENGTH / 2 / 2

# round_low
cp_round_low = base_cp.copy()
cp_round_low = cp_round_low * BACKBONE_LENGTH / 2 * 5 / 8 / 2

# round_high
cp_round_high = base_cp.copy()
cp_round_high = cp_round_high * BACKBONE_LENGTH / 2 / 2

# convex_low
cp_convex_low = base_cp.copy()
cp_convex_low[0, :] = [1.2, 0]
cp_convex_low = cp_convex_low * BACKBONE_LENGTH / 2 / 2

# convex_med
cp_convex_med = base_cp.copy()
cp_convex_med[0, :] = [1.4, 0]
cp_convex_med = cp_convex_med * BACKBONE_LENGTH / 2 / 2

# convex_high
cp_convex_high = base_cp.copy()
cp_convex_high[0, :] = [1.6, 0]
cp_convex_high = cp_convex_high * BACKBONE_LENGTH / 2 / 2

# plane
cp_plane = base_cp.copy()
cp_plane[0, :] = cp_plane[[1, -1], :].mean(axis=0)
cp_plane = cp_plane * BACKBONE_LENGTH / 2 / 2

# convex_point_low
cp_convex_point_low = base_cp.copy()
cp_convex_point_low[[-1, 0, 1], :] = [c(0 / 8 * 2 * np.pi) * 0.1, s(0 / 8 * 2 * np.pi) * 0.1]
cp_convex_point_low = cp_convex_point_low * BACKBONE_LENGTH / 2 / 2

# convex_point_med
cp_convex_point_med = base_cp.copy()
cp_convex_point_med[[-1, 0, 1], :] = [c(0 / 8 * 2 * np.pi) * 1 / 2, s(0 / 8 * 2 * np.pi) * 1 / 2]
cp_convex_point_med = cp_convex_point_med * BACKBONE_LENGTH / 2 / 2

# convex_point_high
cp_convex_point_high = base_cp.copy()
cp_convex_point_high[[-1, 0, 1], :] = [c(0 / 8 * 2 * np.pi), s(0 / 8 * 2 * np.pi)]
cp_convex_point_high = cp_convex_point_high * BACKBONE_LENGTH / 2 / 2
