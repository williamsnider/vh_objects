# Contains the digit segments and cross sections we will use to build our shapes.

import numpy as np
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
import trimesh
from objects.parameters import (
    BACKBONE_LENGTH,
    BACKBONE_NUM_CP,
    cs_scale_backbone,
    cs_scale_surface_deformation,
    SD_LENGTH,
    STRAIGHT_PROPORTION,
    ARC_ANGLE,
    ELLIPTICAL_MAJOR,
    ELLIPTICAL_MINOR,
)
from scipy.optimize import minimize
from objects.utilities import sliding_window_mean, approximate_arc, fuse_meshes

components = {}


##################
### Parameters ###
##################
MAX_BACKBONE_CURVATURE = np.pi
BACKBONE_BASE_RADIUS = 20  # mm
BACKBONE_STEP_RADIUS = 5  # mm
NUM_BACKBONE_STEPS = 5
NUM_CROSS_SECTION_CP = 8
NUM_CROSS_SECTION_VARIATIONS = 7  # Concave --> convex :: 0,1,2,3...
CP_PLANE_INDEX_SHIFT = 2  # Shift index of which cross section variation is planar so that it is index 0


def calc_scale_for_thickness(radius, base_cp):
    """Calculates the optimal scale factor that results in a B-spline surface with the requested radius."""

    def radius_error(scale_factor):

        # Create axial Component
        backbone = Backbone(approximate_arc(0, BACKBONE_LENGTH), reparameterize=True)
        cross_sections = [CrossSection(base_cp * scale_factor, i) for i in np.linspace(0.1, 0.9, 5)]
        ac = AxialComponent(backbone, cross_sections)

        # Sample points on axial component
        (us, vs) = ac.surface.start()
        (ue, ve) = ac.surface.end()
        u = np.linspace(us, ue, 100)  # Sample around ac
        v = np.linspace(0.3, 0.7, 101)  # Sample middle region of ac
        surface_points = ac.surface(u, v)

        # Calculate distance to center axis
        surface_points_to_center = surface_points - surface_points.mean(axis=0)
        dist = np.linalg.norm(surface_points_to_center, axis=2)
        error = ((radius - dist) ** 2).sum()
        return error

    fun = radius_error
    x0 = [radius]
    bounds = [
        [0.0, 10 * radius],
    ]
    result = minimize(fun=fun, x0=x0, bounds=bounds)
    return result.x


######################
### Cross Sections ###
######################

# Define base_cp that forms a round cross section
c = np.cos
s = np.sin
argument = np.linspace(0, 2 * np.pi, NUM_CROSS_SECTION_CP, endpoint=False).reshape(-1, 1)
base_cp = np.hstack([c(argument), s(argument)])

# Calculate the scale factor that gives the desired b-spline surface thickness (since controlpoints are not necessarily on the surface)
radii = (np.arange(NUM_BACKBONE_STEPS) - (NUM_BACKBONE_STEPS - 1) // 2) * BACKBONE_STEP_RADIUS + BACKBONE_BASE_RADIUS
for radius in radii:
    cp_scale_name = "cp_scale_for_radius_{}".format(radius)
    components[cp_scale_name] = calc_scale_for_thickness(radius, base_cp)


# Define plane_cp that forms a cross section with a planar portion; used to determine spacing for cross sections
# Use these spacings to generate range of cross section concavities/convexities
# cp_variation_0 = planar
# cp_variation_2 = normal round (cylinder)
plane_cp = base_cp.copy()
vec_between_neighbors = plane_cp[1] - plane_cp[-1]
plane_cp[0] = plane_cp[-1] + 0.5 * vec_between_neighbors  # Move first cp to midpoint between neighboring cp's
vec_between_plane_and_base = (base_cp[0] - plane_cp[0]) / 2  # Divide by two so that spacing is half this distance
steps = np.arange(NUM_CROSS_SECTION_VARIATIONS) - CP_PLANE_INDEX_SHIFT
for step in steps:
    cp = base_cp.copy()
    cp[0] = plane_cp[0] + step * vec_between_plane_and_base
    cp_name = "cp_concave_convex_{}".format(step)
    components[cp_name] = cp


#################
### Backbones ###
#################

# Backbones with different curvatures
for max_angle in np.linspace(0, MAX_BACKBONE_CURVATURE, 5):
    cp = approximate_arc(max_angle, BACKBONE_LENGTH)
    b_name = "b_curvature_" + str(np.round(max_angle, 2)).replace(".", "p")
    components[b_name] = Backbone(cp, reparameterize=True, name=b_name)


# For clarity, define variables used in the following transformations
cp_x = base_cp[0, 0]  # x coordinate of point we will manipulate
cp_x_prev_next = base_cp[1, 0]  # x coordinate of points on either side of the point we will manipulate
concave_convex_shift = cp_x_prev_next - 0.001  # How much to shift the point for concave/convex cross sections

# concave_high
cp_concave = base_cp.copy()
cp_concave[0, 0] = cp_x_prev_next - concave_convex_shift
cp_concave = cp_concave * cs_scale_backbone

# # concave_low
# # We want the concave low's curvature to be the opposite of the round_high. To do this, we will reflect the controlpoint across the line connecting the current and next controlpoints. In other words, the x-value of this controlpoint will be the same distance away from the previous/next controlpoint's x values, however, it will be closer to the origin.
# cp_concave_low = base_cp.copy()
# cp_x_flipped = cp_x_prev_next - (cp_x - cp_x_prev_next)  # shift to left of the line
# cp_concave_low[0, 0] = cp_x_flipped
# cp_concave_low = cp_concave_low * cs_scale_backbone

# elliptical
cp_elliptical = base_cp.copy()
cp_elliptical[:, 0] *= 2 / 3
cp_elliptical[:, 1] *= 4 / 3
cp_elliptical *= cs_scale_backbone

# # round_low
# cp_round_low = base_cp.copy()
# cp_round_low = cp_round_low * BACKBONE_LENGTH / 2 * 5 / 8 / 2

# round_high
cp_round = base_cp.copy()
cp_round = cp_round * cs_scale_backbone

# convex - inverse of concave_high
cp_convex = base_cp.copy()
cp_convex[0, 0] = cp_x_prev_next + concave_convex_shift
cp_convex *= cs_scale_backbone

# plane
cp_plane = base_cp.copy()
cp_plane[[-1, 0, 1], 0] = 0.001  # controlpoint cannot be at (0,0) for cross sections
cp_plane = cp_plane * cs_scale_backbone


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
cp_concave = base_cp.copy()
cp_concave[0, 0] = cp_x_prev_next - concave_convex_shift
cp_concave = cp_concave * cs_scale_backbone

# # concave_low
# # We want the concave low's curvature to be the opposite of the round_high. To do this, we will reflect the controlpoint across the line connecting the current and next controlpoints. In other words, the x-value of this controlpoint will be the same distance away from the previous/next controlpoint's x values, however, it will be closer to the origin.
# cp_concave_low = base_cp.copy()
# cp_x_flipped = cp_x_prev_next - (cp_x - cp_x_prev_next)  # shift to left of the line
# cp_concave_low[0, 0] = cp_x_flipped
# cp_concave_low = cp_concave_low * cs_scale_backbone

# elliptical
cp_elliptical = base_cp.copy()
cp_elliptical[:, 0] *= 2 / 3
cp_elliptical[:, 1] *= 4 / 3
cp_elliptical *= cs_scale_backbone

# # round_low
# cp_round_low = base_cp.copy()
# cp_round_low = cp_round_low * BACKBONE_LENGTH / 2 * 5 / 8 / 2

# round_high
cp_round = base_cp.copy()
cp_round = cp_round * cs_scale_backbone

# convex - inverse of concave_high
cp_convex = base_cp.copy()
cp_convex[0, 0] = cp_x_prev_next + concave_convex_shift
cp_convex *= cs_scale_backbone

# plane
cp_plane = base_cp.copy()
cp_plane[[-1, 0, 1], 0] = 0.001  # controlpoint cannot be at (0,0) for cross sections
cp_plane = cp_plane * cs_scale_backbone

##########################################
### Backbones for surface deformations ###
##########################################


# SD Flat Backbone
sd_flat_cp = np.array([np.linspace(0, SD_LENGTH, 3), np.zeros(3), np.zeros(3)]).T
sd_flat = Backbone(sd_flat_cp, reparameterize=True, name="sd_flat")

# SD Weak Curve Backbone
curve_arc = approximate_arc(ARC_ANGLE, SD_LENGTH * (1 - STRAIGHT_PROPORTION))
straight = np.array([np.linspace(0, STRAIGHT_PROPORTION * SD_LENGTH, 2, endpoint=False), np.zeros(2), np.zeros(2)]).T
curve_arc_shifted = curve_arc + np.array([[STRAIGHT_PROPORTION * SD_LENGTH, 0, 0]])
sd_weak_curve_cp = np.concatenate([straight, curve_arc_shifted], axis=0)
sd_weak_curve = Backbone(sd_weak_curve_cp, reparameterize=True, name="sd_weak_curve")

###############################################
### Cross sections for surface deformations ###
###############################################

# SD Round
sd_cp_round = base_cp.copy()
sd_cp_round *= cs_scale_surface_deformation

# SD Elliptical
sd_cp_elliptical = base_cp.copy()
sd_cp_elliptical *= cs_scale_surface_deformation
sd_cp_elliptical *= np.array([ELLIPTICAL_MAJOR, ELLIPTICAL_MINOR])

#####################################
### Surface deformations (meshes) ###
#####################################


# Cylinder
cp = sd_cp_round
rotation = 0
cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 5)]
ac = AxialComponent(backbone=sd_flat, cross_sections=cs_list)
origin = ac.backbone.r(STRAIGHT_PROPORTION)[0]
sd_cylinder = (ac.mesh, origin)

# Cylinder - spherical top
vs, us = ac.surface.start()
ve, ue = ac.surface.end()
opposite_points = np.squeeze(ac.surface([vs, (vs + ve) / 2], 0.5))
sphere_radius = np.linalg.norm(opposite_points[1] - opposite_points[0]) / 2
join_position = (SD_LENGTH - sphere_radius) / SD_LENGTH
join_point = ac.backbone.r(join_position)[0]  #
sphere_top = trimesh.creation.icosphere(subdivisions=3, radius=sphere_radius)
sphere_top.apply_translation(join_point)  # Align center of sphere to end of cylinder
cp = sd_cp_round
rotation = 0
cs_list = [
    *[CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, join_position, 5)],
    *[CrossSection(cp * 0.1, i, rotation=rotation) for i in np.linspace(join_position, 0.9, 5)],
]
ac_cylinder_spherical_top = AxialComponent(backbone=sd_flat, cross_sections=cs_list)
origin = ac_cylinder_spherical_top.backbone.r(STRAIGHT_PROPORTION)[0]
fused = fuse_meshes(ac_cylinder_spherical_top.mesh, sphere_top, 1, "union")
sd_cylinder_spherical_top = (fused, origin)

# Sphere
sphere = trimesh.creation.icosphere(subdivisions=3, radius=sphere_radius)
origin = sphere.centroid
sd_sphere = (sphere, origin)

# Ellipsoid
ellipsoid = sphere.copy()
ellipsoid.apply_scale([1, ELLIPTICAL_MAJOR, ELLIPTICAL_MINOR])
origin = ellipsoid.centroid
sd_ellipsoid = (ellipsoid, origin)

# Elliptical Cylinder
cp = sd_cp_elliptical
rotation = 0
cs_list = [
    CrossSection(cp * 0.1, 0.0, rotation=rotation),
    *[CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 5)],
]
ac = AxialComponent(backbone=sd_flat, cross_sections=cs_list)
origin = ac.backbone.r(STRAIGHT_PROPORTION)[0]
sd_elliptical_cylinder = (ac.mesh, origin)

# Cone forward (pointy)
cp = sd_cp_round
rotation = 0
cs_list = [
    CrossSection(cp, 0.0, rotation=rotation),
    CrossSection(cp * 0.2, 1.0, rotation=rotation),
]
ac = AxialComponent(backbone=sd_flat, cross_sections=cs_list)
origin = ac.backbone.r(STRAIGHT_PROPORTION)[0]
sd_cone = (ac.mesh, origin)

# # Cone reverse (widening)
# cp = sd_cp_round
# rotation = 0
# cs_list = [
#     CrossSection(cp * 0.1, 0.0, rotation=rotation),
#     CrossSection(cp * 0.2, 0.25, rotation=rotation),
#     CrossSection(cp * 0.2, 0.25, rotation=rotation),
#     CrossSection(cp, 1.0, rotation=rotation),
#     CrossSection(cp, 1.0, rotation=rotation),
# ]
# ac = AxialComponent(backbone=sd_flat, cross_sections=cs_list)
# origin = ac.backbone.r(STRAIGHT_PROPORTION)[0]
# sd_cone_reverse = (ac.mesh, origin)

# Curved Cylinder
cp = sd_cp_round
rotation = 0
cs_list = [
    CrossSection(cp * 0.1, 0.0, rotation=rotation),
    *[CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 5)],
]
ac = AxialComponent(backbone=sd_weak_curve, cross_sections=cs_list)
origin = ac.backbone.r(STRAIGHT_PROPORTION)[0]
sd_curved_cylinder = (ac.mesh, origin)

# Curved Elliptical cylinder
cp = sd_cp_elliptical
rotation = 0
cs_list = [
    CrossSection(cp * 0.1, 0.0, rotation=rotation),
    *[CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 5)],
]
ac = AxialComponent(backbone=sd_weak_curve, cross_sections=cs_list)
origin = ac.backbone.r(STRAIGHT_PROPORTION)[0]
sd_curved_elliptical_cylinder = (ac.mesh, origin)

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
