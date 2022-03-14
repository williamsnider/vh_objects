# Contains the digit segments and cross sections we will use to build our shapes.

import numpy as np
from objects.backbone import Backbone
from objects.parameters import BACKBONE_LENGTH, BACKBONE_NUM_CP
from scipy.optimize import minimize
from objects.utilities import sliding_window_mean

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
backbone_flat = Backbone(cp_flat, reparameterize=True, name="Flat")

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
backbone_weak_curve = Backbone(cp_weak_curve, reparameterize=True, name="Weak Curve")

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
backbone_strong_curve = Backbone(cp_strong_curve, reparameterize=True, name="Strong Curve")

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
backbone_sharp_bend = Backbone(cp_sharp_bend, reparameterize=True, name="Sharp Bend")

###  S
NUM_CP = 4
S_PROPORTION = 0.15  # How far up/down the middle controlpoints are. Higher --> sharper curves
cp_s = np.array(
    [
        [0, BACKBONE_LENGTH * 0.25, BACKBONE_LENGTH * 0.75, BACKBONE_LENGTH],
        [0, BACKBONE_LENGTH * S_PROPORTION, -BACKBONE_LENGTH * S_PROPORTION, 0],
        np.zeros(NUM_CP),
    ]
).T
backbone_s = Backbone(cp_s, reparameterize=True, name="S")

### Hook forward
# To match curvature of S, sample many controlpoints along the S, and then convert 1 side to linear.
NUM_CP = 100
t = np.linspace(0, 1, NUM_CP)
cp_hook_f = backbone_s.r(t)
cp_hook_f[NUM_CP // 2 :, 1] = 0  # Make straight by clamping Y-values to 0
window = NUM_CP // 10 - 1  # Smooth 10% of backbone to avoid harsh junction
start = NUM_CP // 2 - (window - 1) // 2
stop = NUM_CP // 2 + (window + 1) // 2
cp_hook_f[start:stop] = sliding_window_mean(cp_hook_f, window, axis=0)[start:stop]  # Smooth junction
backbone_hook_f = Backbone(cp_hook_f, reparameterize=True, name="Hook F")

### Hook reverse
# To match curvature of S, sample many controlpoints along the S, and then convert 1 side to linear.
NUM_CP = 100
cp_hook_r = cp_hook_f.copy()
cp_hook_r = np.flip(cp_hook_r, axis=0)  # Reorder x-values
cp_hook_r[:, 0] *= -1  # Flip across y-axis
cp_hook_r -= cp_hook_r[0]  # Shift leftmost point to origin
window = NUM_CP // 10 - 1  # Smooth 10% of backbone to avoid harsh junction
start = NUM_CP // 2 + (window + 1) // 2
stop = NUM_CP // 2 - (window - 1) // 2
cp_hook_r[start:stop] = sliding_window_mean(cp_hook_r, window, axis=0)[start:stop]  # Smooth junction
backbone_hook_r = Backbone(cp_hook_r, reparameterize=True, name="Hook R")
backbone = backbone_hook_r


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

# For clarity, define variables used in the following transformations
cp_x = base_cp[0, 0]  # x coordinate of point we will manipulate
cp_x_prev_next = base_cp[1, 0]  # x coordinate of points on either side of the point we will manipulate
concave_convex_shift = cp_x_prev_next - 0.001  # How much to shift the point for concave/convex cross sections

# concave_high
cp_concave_high = base_cp.copy()
cp_concave_high[0, 0] = cp_x_prev_next - concave_convex_shift
cp_concave_high = cp_concave_high * BACKBONE_LENGTH / 2 / 2

# # concave_low
# # We want the concave low's curvature to be the opposite of the round_high. To do this, we will reflect the controlpoint across the line connecting the current and next controlpoints. In other words, the x-value of this controlpoint will be the same distance away from the previous/next controlpoint's x values, however, it will be closer to the origin.
# cp_concave_low = base_cp.copy()
# cp_x_flipped = cp_x_prev_next - (cp_x - cp_x_prev_next)  # shift to left of the line
# cp_concave_low[0, 0] = cp_x_flipped
# cp_concave_low = cp_concave_low * BACKBONE_LENGTH / 2 / 2

# elliptical
cp_elliptical = base_cp.copy()
cp_elliptical[:, 0] *= 2 / 3
cp_elliptical[:, 1] *= 4 / 3
cp_elliptical *= BACKBONE_LENGTH / 2 / 2

# # round_low
# cp_round_low = base_cp.copy()
# cp_round_low = cp_round_low * BACKBONE_LENGTH / 2 * 5 / 8 / 2

# round_high
cp_round = base_cp.copy()
cp_round = cp_round * BACKBONE_LENGTH / 2 / 2

# convex - inverse of concave_high
cp_convex = base_cp.copy()
cp_convex[0, 0] = cp_x_prev_next + concave_convex_shift
cp_convex *= BACKBONE_LENGTH / 2 / 2

# plane
cp_plane = base_cp.copy()
cp_plane[[-1, 0, 1], 0] = 0.001  # controlpoint cannot be at (0,0) for cross sections
cp_plane = cp_plane * BACKBONE_LENGTH / 2 / 2


# def approximate_arc(MAX_ANGLE, arc_length):
#     """Construct a B-Spline curve that approximates a circular arc."""

#     radius = arc_length / (2 * np.pi) * (2 * np.pi / MAX_ANGLE)

#     # if NUM_CP_PER_SEGMENT != 5:
#     #     raise NotImplementedError("NUM_CP_PER_SEGMENT must be 5 for this funtion to work.")

#     def make_arc_array(a, b, c):

#         # We can think of the second to last arc controlpoint as lying along a vector from the last controlpoint. The vector's slope can be determined from the tangent line of the circle (which is negated). We then can use a single parameter (d) as a measure of how far along this vector we are travelling. This reduces the number of parameters we need, and also ensures that the tangent of the resulting arc at the end will match that of the circle

#         def tan_vec(MAX_ANGLE):
#             tangent_vec = np.array(
#                 [
#                     -radius * np.sin(MAX_ANGLE),
#                     radius * np.cos(MAX_ANGLE),
#                 ]
#             )
#             return tangent_vec

#         start_tan_vec = tan_vec(0)
#         end_tan_vec = -tan_vec(MAX_ANGLE)  # Negate so this points toward start

#         arc_array = np.array(
#             [
#                 [radius, 0, 0],
#                 [
#                     radius * np.cos(0) + a * start_tan_vec[0],
#                     radius * np.sin(0) + a * start_tan_vec[1],
#                     0,
#                 ],  # Tangent line from start
#                 [b, c, 0],
#                 [
#                     radius * np.cos(MAX_ANGLE) + a * end_tan_vec[0],
#                     radius * np.sin(MAX_ANGLE) + a * end_tan_vec[1],
#                     0,
#                 ],  # Tangent line from end
#                 [radius * np.cos(MAX_ANGLE), radius * np.sin(MAX_ANGLE), 0],
#             ]
#         )
#         return arc_array

#     def radius_error(vars):

#         # Make the arc array
#         [a, b, c] = vars
#         arc_array = make_arc_array(a, b, c)

#         # Make backbone
#         backbone = Backbone(arc_array, reparameterize=False)

#         # Sample points along the backbone
#         t = np.linspace(0, 1, 10)
#         r = backbone.r(t)

#         # distance from origin should be close to radius if points are well_aligned
#         dist = np.linalg.norm(r, axis=1)
#         return ((dist - radius) ** 2).sum()

#     fun = radius_error
#     x0 = [0.1, radius, radius]
#     bounds = [
#         [0.0, 100 * radius],
#         [radius * np.cos(MAX_ANGLE / 2), 100 * radius],  # Convex hull property of B-Splines
#         [radius * np.sin(MAX_ANGLE / 2), 100 * radius],  # Convex hull property of B-Splines
#     ]
#     result = minimize(fun=fun, x0=x0, bounds=bounds)
#     [a, b, c] = result.x

#     arc_array = make_arc_array(a, b, c)

#     # Shift so that the curve begins at the origin
#     arc_array[:, 0] -= radius
#     arc_array[:, [0, 1]] = arc_array[:, [1, 0]]  # Flip x and y-axis so long portion points in +X direction
#     arc_array[:, 1] = -arc_array[:, 1]  # Negate y axis so curves upward (towards +Y)

#     return arc_array

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
