from objects.components import (
    segment_flat,
    segment_arc_1_4,
    segment_arc_1_8,
    cp_concave_high,
    cp_concave_low,
    cp_round_low,
    cp_round_high,
    cp_convex_low,
    cp_convex_med,
    cp_convex_high,
    cp_plane,
    cp_concave_point,
    cp_convex_point_low,
    cp_convex_point_high,
)
from objects.backbone_from_digits import BackboneFromDigits
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.parameters import GOAL_LENGTH_SEGMENT
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pathlib import Path
from joblib import Parallel, delayed
import time

base_dir = Path(Path.cwd(), "sample_shapes", "stimulus_set", "iteration1")

# ######################################
# ### Make backbone combinations
# ######################################

# Segments
valid_segments = [segment_flat, segment_arc_1_4]
segment_combinations = [list(comb) for comb in itertools.product(valid_segments, repeat=4)]

# Angles
T_euler = np.linspace(0, 2 * np.pi, 4, endpoint=False)  # These rotations introduce kinks
N_euler = np.linspace(-np.pi / 4, np.pi / 4, 3, endpoint=True)
B_euler = np.linspace(-np.pi / 4, np.pi / 4, 3, endpoint=True)
valid_angles = [[t, n, b] for t in T_euler for n in N_euler for b in B_euler]


# Allow T at all positions
# Allow N at only middle position (1)
# Allow B at only middle position (1)
angle_combinations = []
for [pos0_T, pos0_N, pos0_B] in valid_angles:

    if pos0_N != 0 or pos0_B != 0:
        continue

    for [pos1_T, pos1_N, pos1_B] in valid_angles:
        for [pos2_T, pos2_N, pos2_B] in valid_angles:

            if pos2_N != 0 or pos2_B != 0:
                continue

            angle_combinations.append(
                [
                    [pos0_T, pos0_N, pos0_B],
                    [pos1_T, pos1_N, pos1_B],
                    [pos2_T, pos2_N, pos2_B],
                ]
            )
angle_combinations = np.array(angle_combinations)


# # Backbone combinations
def make_backbone(digit_segments, angles_between_segments):
    bfd = BackboneFromDigits(digit_segments, angles_between_segments)
    backbone = Backbone(bfd.controlpoints, reparameterize=True)
    return backbone


argument_list = []
for digit_segments in segment_combinations:
    for angles_between_segments in angle_combinations:
        args = [digit_segments, angles_between_segments]
        argument_list.append(args)

print("Generating backbone combinations...")
start = time.time()
backbone_list = Parallel(n_jobs=-1)(delayed(make_backbone)(*args) for args in argument_list)
end = time.time()
print("Generating backbone combinations DONE.")

# Cull backbones whose first and last contrrolpoints are too close
backbone_list_long_enough = []
for backbone in backbone_list:
    start = backbone.controlpoints[0]
    end = backbone.controlpoints[-1]
    dist = np.linalg.norm(end - start)
    if dist > 2 * GOAL_LENGTH_SEGMENT:
        backbone_list_long_enough.append(backbone)

# Detect redundant
# This method considers each backbone as a list of distances (from each controlpoint to the origin, which is the same as the first controlpoint). If all the distances between two backbones are identical, it is considered a duplicate. Thus, this excludes backbones that are simple rotations of each other. Consequently, when applying cross sections, we should rotate those so that they are not all applied to the same part of a backbone. However, this method is bad because it will have a false positive for chiral trajectories
def round_to_5_or_0(x, base=5):
    return base * np.round(x / base)


dist_from_each_cp_to_origin = np.zeros((len(backbone_list_long_enough), (4 * 4 + 1)))
for i, backbone in enumerate(backbone_list_long_enough):
    dist_from_each_cp_to_origin[i, :] = np.linalg.norm(backbone.controlpoints, axis=1)
dist_from_each_cp_to_origin = round_to_5_or_0(dist_from_each_cp_to_origin)
_, indices = np.unique(dist_from_each_cp_to_origin, axis=0, return_index=True)
backbone_list_pruned = [backbone_list_long_enough[i] for i in np.sort(indices)]
print("Num backbones: ", len(backbone_list_pruned))


# TODO: I'm not convinced that a chain of arcs rotate properly about the tangent axis.


######################################
### Make cross section combinations
######################################
cs_combinations = []

# # Cylinder
# c = np.cos
# s = np.sin
# cs_cp = np.array(
#     [
#         [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
#         [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
#         [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
#         [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
#         [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
#         [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
#     ]
# )
# cs0 = CrossSection(cs_cp * 15, 0.0500)
# cs1 = CrossSection(cs_cp * 15, 0.1625)
# cs2 = CrossSection(cs_cp * 15, 0.2750)
# cs3 = CrossSection(cs_cp * 15, 0.3875)
# cs4 = CrossSection(cs_cp * 15, 0.5000)
# cs5 = CrossSection(cs_cp * 15, 0.6125)
# cs6 = CrossSection(cs_cp * 15, 0.7250)
# cs7 = CrossSection(cs_cp * 15, 0.8375)
# cs8 = CrossSection(cs_cp * 15, 0.9500)

# cs_combinations.append([cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8])


######################################
### Define functions that will be paralleled
######################################
def make_shape(backbone, cross_sections, label, png_save_dir, stl_save_dir):
    ac = AxialComponent(
        backbone,
        cross_sections,
    )
    s = Shape([ac], align_OBB=False, fuse_to_interface=True, label=label)
    print(s.label)
    # s.mesh.show()
    s.save_mesh_as_png(png_save_dir)
    s.export_stl(stl_save_dir)


######################################
### Render all backbones with circular cross section
######################################

# # Construct argument list for parallelization
# argument_list = []

# # Define save_dir
# png_save_dir = Path(base_dir, "png", "medial_axis")

# stl_save_dir = Path(base_dir, "stl", "medial_axis")

# # Populate argument list
# count = 0
# for backbone in backbone_list_pruned:

#     for cross_sections in circular_cs:

#         label = "ac_{}".format(count)
#         args = [backbone, cross_sections, label, png_save_dir, stl_save_dir]
#         count += 1
#         argument_list.append(args)

# start = time.time()
# Parallel(n_jobs=-1)(delayed(make_shape)(*args) for args in argument_list)
# end = time.time()
# print("Execution time: ", end - start)


######################################
### Render all cross sections with straight backbone
######################################
cs_combinations = []

# Same cross section across shape
for cp in [
    cp_concave_high,
    cp_concave_low,
    cp_round_low,
    cp_round_high,
    cp_convex_low,
    cp_convex_med,
    cp_convex_high,
    cp_plane,
    cp_concave_point,
    cp_convex_point_low,
    cp_convex_point_high,
]:

    # Shift point a little further so it's easier to see
    if cp is cp_concave_point or cp is cp_convex_point_low or cp is cp_convex_point_high:
        rotation = 3 * np.pi / 4
    else:
        rotation = np.pi / 2

    cs0 = CrossSection(cp, 0.0500, rotation=rotation)
    cs1 = CrossSection(cp, 0.1625, rotation=rotation)
    cs2 = CrossSection(cp, 0.2750, rotation=rotation)
    cs3 = CrossSection(cp, 0.3875, rotation=rotation)
    cs4 = CrossSection(cp, 0.5000, rotation=rotation)
    cs5 = CrossSection(cp, 0.6125, rotation=rotation)
    cs6 = CrossSection(cp, 0.7250, rotation=rotation)
    cs7 = CrossSection(cp, 0.8375, rotation=rotation)
    cs8 = CrossSection(cp, 0.9500, rotation=rotation)
    cs_combinations.append([cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8])

# Cross section at each digit zone, with rest of shape round
for digit in [2, 3, 4, 5]:

    for cp in [
        cp_concave_high,
        cp_concave_low,
        cp_convex_low,
        cp_convex_med,
        cp_convex_high,
        cp_round_low,
        cp_plane,
        cp_concave_point,
        cp_convex_point_low,
        cp_convex_point_high,
    ]:  # Omit cp_round_high

        # Shift point a little further so it's easier to see
        if cp is cp_concave_point or cp is cp_convex_point_low or cp is cp_convex_point_high:
            rotation = 3 * np.pi / 4
        else:
            rotation = np.pi / 2

        if digit == 2:
            cs0 = CrossSection(cp, 0.0500, rotation=rotation)
            cs1 = CrossSection(cp, 0.1625, rotation=rotation)
            cs2 = CrossSection(cp, 0.2750, rotation=rotation)
            cs3 = CrossSection(cp_round_high, 0.3875, rotation=rotation)
            cs4 = CrossSection(cp_round_high, 0.5000, rotation=rotation)
            cs5 = CrossSection(cp_round_high, 0.6125, rotation=rotation)
            cs6 = CrossSection(cp_round_high, 0.7250, rotation=rotation)
            cs7 = CrossSection(cp_round_high, 0.8375, rotation=rotation)
            cs8 = CrossSection(cp_round_high, 0.9500, rotation=rotation)
            cs_combinations.append([cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8])
        elif digit == 3:
            cs0 = CrossSection(cp_round_high, 0.0500, rotation=rotation)
            cs1 = CrossSection(cp_round_high, 0.1625, rotation=rotation)
            cs2 = CrossSection(cp, 0.2750, rotation=rotation)
            cs3 = CrossSection(cp, 0.3875, rotation=rotation)
            cs4 = CrossSection(cp, 0.5000, rotation=rotation)
            cs5 = CrossSection(cp_round_high, 0.6125, rotation=rotation)
            cs6 = CrossSection(cp_round_high, 0.7250, rotation=rotation)
            cs7 = CrossSection(cp_round_high, 0.8375, rotation=rotation)
            cs8 = CrossSection(cp_round_high, 0.9500, rotation=rotation)
            cs_combinations.append([cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8])
        elif digit == 4:
            cs0 = CrossSection(cp_round_high, 0.0500, rotation=rotation)
            cs1 = CrossSection(cp_round_high, 0.1625, rotation=rotation)
            cs2 = CrossSection(cp_round_high, 0.2750, rotation=rotation)
            cs3 = CrossSection(cp_round_high, 0.3875, rotation=rotation)
            cs4 = CrossSection(cp, 0.5000, rotation=rotation)
            cs5 = CrossSection(cp, 0.6125, rotation=rotation)
            cs6 = CrossSection(cp, 0.7250, rotation=rotation)
            cs7 = CrossSection(cp_round_high, 0.8375, rotation=rotation)
            cs8 = CrossSection(cp_round_high, 0.9500, rotation=rotation)
            cs_combinations.append([cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8])
        elif digit == 5:
            cs0 = CrossSection(cp_round_high, 0.0500, rotation=rotation)
            cs1 = CrossSection(cp_round_high, 0.1625, rotation=rotation)
            cs2 = CrossSection(cp_round_high, 0.2750, rotation=rotation)
            cs3 = CrossSection(cp_round_high, 0.3875, rotation=rotation)
            cs4 = CrossSection(cp_round_high, 0.5000, rotation=rotation)
            cs5 = CrossSection(cp_round_high, 0.6125, rotation=rotation)
            cs6 = CrossSection(cp, 0.7250, rotation=rotation)
            cs7 = CrossSection(cp, 0.8375, rotation=rotation)
            cs8 = CrossSection(cp, 0.9500, rotation=rotation)
            cs_combinations.append([cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8])

# A-B-A pattern
for cpA in [
    cp_concave_high,
    cp_concave_low,
    cp_round_low,
    cp_round_high,
    cp_convex_low,
    cp_convex_med,
    cp_convex_high,
    cp_plane,
    cp_concave_point,
    cp_convex_point_low,
    cp_convex_point_high,
]:

    for cpB in [
        cp_concave_high,
        cp_concave_low,
        cp_round_low,
        cp_round_high,
        cp_convex_low,
        cp_convex_med,
        cp_convex_high,
        cp_plane,
        cp_concave_point,
        cp_convex_point_low,
        cp_convex_point_high,
    ]:

        # Skip all same since that was done above
        if cpA is cpB:
            continue

        # Shift point a little further so it's easier to see
        if cpA is cp_concave_point or cpA is cp_convex_point_low or cpA is cp_convex_point_high:
            rotationA = np.pi / 2
        else:
            rotationA = np.pi / 2

        # Shift point a little further so it's easier to see
        if cpB is cp_concave_point or cpB is cp_convex_point_low or cpB is cp_convex_point_high:
            rotationB = np.pi / 2
        else:
            rotationB = np.pi / 2

        cs0 = CrossSection(cpA, 0.0500, rotation=rotationA)
        cs1 = CrossSection(cpA, 0.1625, rotation=rotationA)
        cs2 = CrossSection(cpA, 0.2750, rotation=rotationA)
        cs3 = CrossSection(cpB, 0.3875, rotation=rotationB)
        cs4 = CrossSection(cpB, 0.5000, rotation=rotationB)
        cs5 = CrossSection(cpB, 0.6125, rotation=rotationB)
        cs6 = CrossSection(cpA, 0.7250, rotation=rotationA)
        cs7 = CrossSection(cpA, 0.8375, rotation=rotationA)
        cs8 = CrossSection(cpA, 0.9500, rotation=rotationA)
        cs_combinations.append([cs0, cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8])

# # Construct argument list for parallelization
# argument_list = []

# # Define save_dir
# png_save_dir = Path(base_dir, "png", "cross_section")

# stl_save_dir = Path(base_dir, "stl", "cross_section")

# digit_segments = [segment_flat, segment_flat, segment_flat, segment_flat]
# angles_between_segments = np.zeros([3, 3])
# backbone = make_backbone(digit_segments, angles_between_segments)

# # Populate argument list
# count = 0
# for cross_sections in cs_combinations:

#     label = "cs_{}".format(count)
#     args = [backbone, cross_sections, label, png_save_dir, stl_save_dir]
#     count += 1
#     argument_list.append(args)

# start = time.time()
# Parallel(n_jobs=-1)(delayed(make_shape)(*args) for args in argument_list)
# end = time.time()
# print("Execution time: ", end - start)

######################################
### Render 1000 shapes (shuffled backbones and cross section)
######################################

num_shapes = 1000

# Extend backbone by repeating entire length
num_backbone_repeats = np.ceil(num_shapes / len(backbone_list_pruned)).astype("int")
backbone_list_extended = np.tile(backbone_list_pruned, num_backbone_repeats)
backbone_list_extended = backbone_list_extended[:num_shapes]

# Extend cross sections by duplicating each item
cs_list_extended = np.zeros((num_shapes), dtype="object")
num_cs_repeats = num_shapes / len(cs_combinations)
for i, comb in enumerate(cs_combinations):
    start = np.round(i * num_cs_repeats).astype("int")
    end = np.round((i + 1) * num_cs_repeats).astype("int")
    cs_list_extended[start:end] = [comb]

# Construct argument list for parallelization
argument_list = []

# Define save_dir
png_save_dir = Path(base_dir, "png", "combined")

stl_save_dir = Path(base_dir, "stl", "combined")

# Populate argument list
count = 0
for i in range(num_shapes):

    label = "shape_{}".format(count)
    args = [backbone_list_extended[i], cs_list_extended[i], label, png_save_dir, stl_save_dir]
    count += 1
    argument_list.append(args)

start = time.time()
Parallel(n_jobs=-1)(delayed(make_shape)(*args) for args in argument_list)
end = time.time()
print("Execution time: ", end - start)
