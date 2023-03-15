import numpy as np
from objects.utilities import (
    find_cp_for_desired_radius,
    approximate_arc,
    get_deformation_vertex,
    fuse_meshes,
    fair_mesh,
    angle_between,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.backbone import Backbone
from objects.shape import Shape
import trimesh
import scipy
import scipy.spatial
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import copy
import itertools

#########################
### Common parameters ###
#########################
NUM_CP_PER_CROSS_SECTION = 8
NUM_CP_PER_BACKBONE = 5
NUM_CP_PER_BACKBONE_3SEG = 7
NUM_CROSS_SECTION_2SEG = 10
NUM_CROSS_SECTION_3SEG = int(round(NUM_CROSS_SECTION_2SEG * 1.5))
SEGMENT_LENGTH = 20
RADIUS = 2

#################
### Backbones ###
#################

# One segment
b_straight1_cp = np.hstack(
    [
        np.linspace(0, 1 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_straight1 = Backbone(b_straight1_cp, reparameterize=True)

th = np.pi / 2
D = SEGMENT_LENGTH
r = D / 2 / np.cos(th / 2)
arclength = r * th
b_arc1_cp = approximate_arc(th, arclength)
b_arc1_cp[:, 1] *= -1  # Flip across y-axis
b_arc1 = Backbone(b_arc1_cp, reparameterize=True)

# Two segment
b_straight2_cp = np.hstack(
    [
        np.linspace(0, 2 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_straight2 = Backbone(b_straight2_cp, reparameterize=True)

b_bend45_cp = np.hstack(
    [
        np.linspace(0, 2 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
shift = b_bend45_cp - b_bend45_cp[NUM_CP_PER_BACKBONE // 2, :]
R = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([0, 0, 3 * np.pi / 4])).as_matrix()
shift_R = shift @ R
transformed = shift_R + b_bend45_cp[NUM_CP_PER_BACKBONE // 2, :]
b_bend45_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[NUM_CP_PER_BACKBONE // 2 :: -1, :]
b_bend45 = Backbone(b_bend45_cp, reparameterize=True)

b_bend90_cp = np.hstack(
    [
        np.linspace(0, 2 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
shift = b_bend90_cp - b_bend90_cp[NUM_CP_PER_BACKBONE // 2, :]
R = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([0, 0, 2 * np.pi / 4])).as_matrix()
shift_R = shift @ R
transformed = shift_R + b_bend90_cp[NUM_CP_PER_BACKBONE // 2, :]
b_bend90_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[NUM_CP_PER_BACKBONE // 2 :: -1, :]
b_bend90 = Backbone(b_bend90_cp, reparameterize=True)

b_bend135_cp = np.hstack(
    [
        np.linspace(0, 2 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
shift = b_bend135_cp - b_bend135_cp[NUM_CP_PER_BACKBONE // 2, :]
R = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([0, 0, 1 * np.pi / 4])).as_matrix()
shift_R = shift @ R
transformed = shift_R + b_bend135_cp[NUM_CP_PER_BACKBONE // 2, :]
b_bend135_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[NUM_CP_PER_BACKBONE // 2 :: -1, :]
b_bend135 = Backbone(b_bend135_cp, reparameterize=True)

# Two-segment continuous arc (rotate so endpoints align with b_bend45)
th = np.pi / 2
D = np.linalg.norm(b_bend45_cp[-1] - b_bend45_cp[0])
r = D / 2 / np.cos(th / 2)
arclength = r * th
arc_cp = approximate_arc(th, arclength)
arc_cp[:, 1] *= -1  # Flip across y-axis
th_R = np.arccos(
    np.dot(arc_cp[-1], b_bend45_cp[-1]) / np.linalg.norm(arc_cp[-1]) / np.linalg.norm(b_bend45_cp[-1])
)  # Angle between two vectors
R = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([0, 0, th_R])).as_matrix()
b_arc2_cp = arc_cp @ R
b_arc2 = Backbone(b_arc2_cp, reparameterize=True)

# Hook out (points outward)
segment1 = b_straight1_cp.copy()
segment2 = b_arc1_cp.copy()
segment2[:, 0] += SEGMENT_LENGTH
b_hook_out_cp = np.vstack([segment1, segment2[1:]])
b_hook_out = Backbone(b_hook_out_cp, reparameterize=True)

# Hook up (points upward)
segment1 = b_straight1_cp.copy()
segment2 = b_arc1_cp.copy()
R = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([0, 0, np.pi / 2])).as_matrix()
segment2 = segment2 @ R  # Rotate 90deg
segment2[:, 1] *= -1  # Flip across y-axis
segment2[:, 0] += SEGMENT_LENGTH  # Shift up
b_hook_up_cp = np.vstack([segment1, segment2[1:]])
b_hook_up = Backbone(b_hook_up_cp, reparameterize=True)

# Three segement
b_3seg_bend90_cp = np.hstack(
    [
        np.linspace(0, 3 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE_3SEG).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE_3SEG, 1)),
        np.zeros((NUM_CP_PER_BACKBONE_3SEG, 1)),
    ]
)
shift = b_3seg_bend90_cp - b_3seg_bend90_cp[NUM_CP_PER_BACKBONE_3SEG - 3, :]
R = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([0, 0, -2 * np.pi / 4])).as_matrix()
shift_R = shift @ R
transformed = shift_R + b_3seg_bend90_cp[NUM_CP_PER_BACKBONE_3SEG - 3, :]
b_3seg_bend90_cp[NUM_CP_PER_BACKBONE_3SEG - 3 :, :] = transformed[NUM_CP_PER_BACKBONE_3SEG - 3 :, :]
b_3seg_bend90 = Backbone(b_3seg_bend90_cp, reparameterize=True)

b_3seg_bend135_cp = np.hstack(
    [
        np.linspace(0, 3 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE_3SEG).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE_3SEG, 1)),
        np.zeros((NUM_CP_PER_BACKBONE_3SEG, 1)),
    ]
)
shift = b_3seg_bend135_cp - b_3seg_bend135_cp[NUM_CP_PER_BACKBONE_3SEG - 3, :]
R = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([0, 0, -3 * np.pi / 4])).as_matrix()
shift_R = shift @ R
transformed = shift_R + b_3seg_bend135_cp[NUM_CP_PER_BACKBONE_3SEG - 3, :]
b_3seg_bend135_cp[NUM_CP_PER_BACKBONE_3SEG - 3 :, :] = transformed[NUM_CP_PER_BACKBONE_3SEG - 3 :, :]
b_3seg_bend135 = Backbone(b_3seg_bend135_cp, reparameterize=True)

######################
### Cross sections ###
######################
cp_2 = find_cp_for_desired_radius(2, NUM_CP_PER_CROSS_SECTION)
# cp_5 = find_cp_for_desired_radius(5, NUM_CP_PER_CROSS_SECTION)
# cp_0 = 0

cs_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
cs_2 = np.hstack((cp_2 * np.cos(cs_th), cp_2 * np.sin(cs_th)))
# cs_5 = np.hstack((cp_5 * np.cos(cs_th), cp_5 * np.sin(cs_th)))
# cs_0 = np.hstack((cp_0 * np.cos(cs_th), cp_0 * np.sin(cs_th)))

# 2 seg

# Cylinder
cylinder_2seg = [
    CrossSection(controlpoints=cs_2, position=position) for position in np.linspace(0.0, 0.95, NUM_CROSS_SECTION_2SEG)
]

# Point
scale = np.ones((NUM_CROSS_SECTION_2SEG, 1))
scale[-5:] *= np.linspace(1, 0.1, 5).reshape(-1, 1)
position = np.linspace(0.0, 0.98, NUM_CROSS_SECTION_2SEG)
point_2seg = [CrossSection(controlpoints=scale[i] * cs_2, position=position[i]) for i in range(NUM_CROSS_SECTION_2SEG)]

# Football
scale = np.geomspace(1, 3, NUM_CROSS_SECTION_2SEG // 2)
scale = np.concatenate([scale, scale[-1::-1]])
position = np.linspace(0.0, 0.95, NUM_CROSS_SECTION_2SEG)
football_2seg = [
    CrossSection(controlpoints=scale[i] * cs_2, position=position[i]) for i in range(NUM_CROSS_SECTION_2SEG)
]

# Hourglass
scale = np.geomspace(1, 3, NUM_CROSS_SECTION_2SEG // 2)
scale = np.concatenate([scale[-1::-1], scale])
position = np.linspace(0.0, 0.95, NUM_CROSS_SECTION_2SEG)
hourglass_2seg = [
    CrossSection(controlpoints=scale[i] * cs_2, position=position[i]) for i in range(NUM_CROSS_SECTION_2SEG)
]

# 3 seg

# Cylinder
cylinder_3seg = [
    CrossSection(controlpoints=cs_2, position=position) for position in np.linspace(0.0, 0.95, NUM_CROSS_SECTION_3SEG)
]

# Point
scale = np.ones((NUM_CROSS_SECTION_3SEG, 1))
scale[-5:] *= np.linspace(1, 0.1, 5).reshape(-1, 1)
position = np.linspace(0.0, 0.98, NUM_CROSS_SECTION_3SEG)
point_3seg = [CrossSection(controlpoints=scale[i] * cs_2, position=position[i]) for i in range(NUM_CROSS_SECTION_3SEG)]

# Football
scale = np.geomspace(1, 3, NUM_CROSS_SECTION_2SEG // 2)
scale = np.concatenate([scale, scale[-1::-1]])
scale = np.concatenate([np.ones(NUM_CROSS_SECTION_3SEG - NUM_CROSS_SECTION_2SEG), scale])
position = np.linspace(0.0, 0.95, NUM_CROSS_SECTION_3SEG)
football_3seg = [
    CrossSection(controlpoints=scale[i] * cs_2, position=position[i]) for i in range(NUM_CROSS_SECTION_3SEG)
]

# Hourglass
scale = np.geomspace(1, 3, NUM_CROSS_SECTION_2SEG // 2)
scale = np.concatenate([scale[-1::-1], scale])
scale = np.concatenate([np.ones(NUM_CROSS_SECTION_3SEG - NUM_CROSS_SECTION_2SEG), scale])
position = np.linspace(0.0, 0.95, NUM_CROSS_SECTION_3SEG)
hourglass_3seg = [
    CrossSection(controlpoints=scale[i] * cs_2, position=position[i]) for i in range(NUM_CROSS_SECTION_3SEG)
]
# cs_list = [cylinder, point, football, hourglass]

###################
### Projections ###
###################

cs_2seg_dict = {
    "cylinder": cylinder_2seg,
    "point": point_2seg,
    "football": football_2seg,
    "hourglass": hourglass_2seg,
}

cs_3seg_dict = {
    "cylinder": cylinder_3seg,
    "point": point_3seg,
    "football": football_3seg,
    "hourglass": hourglass_3seg,
}

# cs_list = [cylinder]  # , point, football, hourglass]
# one_segment_backbone_list = [b_straight1, b_arc1]
two_segment_backbone_dict = {
    "b_straight2": b_straight2,
    "b_bend45": b_bend45,
    "b_bend90": b_bend90,
    "b_bend135": b_bend135,
    "b_arc2": b_arc2,
    "b_hook_out": b_hook_out,
    "b_hook_up": b_hook_up,
}

three_segment_backbone_dict = {
    "b_3seg_bend90": b_3seg_bend90,
    "b_3seg_bend135": b_3seg_bend135,
}
# one_segment_projections = {}
# for b in one_segment_backbone_list:
#     for cs in cs_list:
#         one_segment_projections.append(AxialComponent(b, cs, smooth_with_post=False))
#         one_segment_projections[-1].mesh.show(smooth=False)

combined_projection_dict = {"None": None}
# two_segment_projection_dict = {"None": None}
for b_k, b_v in two_segment_backbone_dict.items():
    for cs_k, cs_v in cs_2seg_dict.items():
        projection_key = b_k + "_" + cs_k
        projection_value = AxialComponent(b_v, cs_v, smooth_with_post=False)
        combined_projection_dict[projection_key] = projection_value

# three_segment_projection_dict = {"None": None}
for b_k, b_v in three_segment_backbone_dict.items():
    for cs_k, cs_v in cs_3seg_dict.items():
        projection_key = b_k + "_" + cs_k
        projection_value = AxialComponent(b_v, cs_v, smooth_with_post=False)
        combined_projection_dict[projection_key] = projection_value

##############
### Shapes ###
##############
# combined_projections = dict(two_segment_projection_dict.update(three_segment_projection_dict))
pairs = [list(tup) for tup in itertools.combinations_with_replacement(combined_projection_dict.keys(), 2)]

rotations = np.linspace(0, 2 * np.pi, 4, endpoint=False)
combs = []
for p1_key, p2_key in pairs:
    for T_rot in rotations:

        # Skip none-none pair
        if p1_key == p2_key == "None":
            continue

        # Skip two identical straight projections (would perfectly overlap)
        if p1_key == p2_key and "straight2" in p2_key:
            continue

        # Skip two identical non-straight projections if T_rot = 0 (would perfectly overlap) or T_rot = 3*np.pi/2 (would perfectly overlap np.pi/2)
        if (p1_key == p2_key) and (T_rot == rotations[0] or T_rot == rotations[-1]):
            continue

        # Skip two straight/non-straight pair if T_rot > 0 (rotations give equivalent shape)
        if (("straight2" in p1_key) or ("straight2" in p2_key)) and T_rot != 0:
            continue

        # Skip straight projction with 3seg projection (perfectly overlap)
        if (("straight2" in p1_key) and ("3seg" in p2_key)) or (("straight2" in p2_key) and ("3seg" in p1_key)):
            continue

        # Skip rotations around straight projection
        if (("straight2" in p1_key) or ("straight2" in p2_key)) and T_rot != 0:
            continue

        # Adjust angle of arc to align
        if "arc2" in p1_key and p2_key != "None":
            B_rot = -np.pi / 8
        elif "arc2" in p2_key and p1_key != "None":
            B_rot = np.pi / 8
        else:
            B_rot = 0.0

        combs.append([p1_key, p2_key, T_rot, B_rot])

        # Only 1 T_rot for single projection shapes
        if p1_key == "None" or p2_key == "None":
            break


shapes = []
for i, (p1_key, p2_key, T_rot, B_rot) in enumerate(combs[:]):

    print(i, p1_key, p2_key, T_rot, B_rot)

    p1 = copy.deepcopy(combined_projection_dict[p1_key])
    p2 = copy.deepcopy(combined_projection_dict[p2_key])

    # Assign parent axial component
    p2.parent_axial_component = p1
    p2.position_along_parent = 0.001  # Slight shift so ends not perfectly aligned; helps union
    # Shift p2 slightly so that p1 and p2 not perfectly aligned; helps mesh union
    for cs in p2.cross_sections:
        cs.controlpoints *= 0.99

    # Rotate arc to align with other projections correctly
    p2.euler_angles = np.array([T_rot, 0, B_rot])

    # Recalculate p2 with these updates
    p2.calc_points()

    # Make shape
    try:
        if p1_key == "None" and p2_key != "None":
            shapes.append(Shape([p2]))
        elif p1_key != "None" and p2_key == "None":
            shapes.append(Shape([p1]))
        else:
            shapes.append(Shape([p1, p2]))
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        print("EXCEPTION")
        continue

    COM = shapes[-1].ac_list[0].backbone.r(0.5)[0]
    # Rotate shape for easier viewing
    shape_euler = np.array([7 * np.pi / 8, 0, np.pi])
    R = np.eye(4)
    R[:3, :3] = scipy.spatial.transform.Rotation.from_euler("xyz", shape_euler).as_matrix()
    COM_rot = R[:3, :3] @ COM  # Rotate center of mass as well
    R[:3, 3] = -COM_rot  # Add COM to transformation matrix

    base_dir = "/home/williamsnider/Code/objects/sample_shapes/stimulus_set_B"
    shapes[-1].label = "test_" + str(i)
    shapes[-1].save_mesh_as_png(
        base_dir + "/png",
        return_img=False,
        rotation=R,
        interface=False,
    )
    # shapes[-1].mesh.show()

# shapes[9].mesh.show()
# shapes[9].export_stl(".")

# # two_segment_projection_dict["b_arc2_cylinder"].T(0.001)

# # T1 = two_segment_projection_dict["b_arc2_cylinder"].r(1)
# # T2 = two_segment_projection_dict["b_bend45_cylinder"].r(1)

# # angle_between(T1, T2)
# # One projection shapes
# for p in two_segment_projections:
#     shapes.append(Shape([p]))
#     shapes[-1].mesh.show()

# # Two projection shapes
# for p1 in two_segment_projections:
#     for p2 in [two_segment_projections]:

#         p2_copy = copy.deepcopy(p2)  # Copy p2 so that changes not applied to original
#         p2_copy.parent = p1  # Assign parent axial component

#         # Assign distance along parent
#         if p2 in one_segment_projections:
#             p2_copy.distance_along_parent = 0.5
#         elif p2 in two_segment_projections:
#             p2_copy.distance_along_parent = 0.0
#         else:
#             raise NotImplementedError

#         # Shift p2 slightly so that p1 and p2 not perfectly aligned; helps mesh union
#         for cs in p2_copy.cross_sections:
#             cs.controlpoints *= 0.999

#         # Recalculate p2 with these updates
#         p2_copy.calc_points()

#         # Make shape
#         shapes.append(Shape([p1, p2]))
#         shapes[-1].mesh.show()


# for cs in cs_list:
#     # p_straight2 = AxialComponent(
#     #     b_straight2,
#     #     cs,
#     #     smooth_with_post=False,
#     # )
#     # p_straight2.mesh.show()
#     p_hook_out = AxialComponent(
#         b_hook_up,
#         cs,
#         smooth_with_post=False,
#     )
#     p_hook_out.mesh.show()

# # Straight
# p_straight2_cylinder = AxialComponent(
#     b_straight2,
#     cylinder,
#     smooth_with_post=False,
# )
# # p_straight2_cylinder.mesh.show(smooth=False)

# p_straight2_point = AxialComponent(
#     b_straight2,
#     point,
#     smooth_with_post=False,
# )
# # p_straight2_point.mesh.show(smooth=False)

# p_straight2_football = AxialComponent(
#     b_straight2,
#     football,
#     smooth_with_post=False,
# )
# p_straight2_football.mesh.show(smooth=False)

# p_straight2_hourglass = AxialComponent(
#     b_straight2,
#     hourglass,
#     smooth_with_post=False,
# )
# p_straight2_hourglass.mesh.show(smooth=False)

# ### Construct projections ###
# NUM_CP_PER_CROSS_SECTION = 8
# NUM_CP_PER_BACKBONE = 5
# LENGTH = 20
# RADIUS = 2

# # Base
# backbone_cp = np.hstack(
#     [
#         np.linspace(0, 10, NUM_CP_PER_BACKBONE).reshape(-1, 1),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#     ]
# )
# backbone = Backbone(backbone_cp, reparameterize=True)
# cp_radius = find_cp_for_desired_radius(RADIUS, NUM_CP_PER_CROSS_SECTION)
# cs_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(
#     -1, 1
# )
# cs_cp = np.hstack((cp_radius * np.cos(cs_th), cp_radius * np.sin(cs_th)))

# cs_list = [
#     CrossSection(controlpoints=1.01 * cs_cp, position=position)
#     for position in np.linspace(0.0, 0.9, 6)
# ]
# cs_list.append(CrossSection(controlpoints=1.01 * cs_cp, position=1.0, tilt=np.pi / 100))
# p_base = AxialComponent(backbone, cs_list, smooth_with_post=False)
# # p_base.mesh.show(smooth=False)

# # Straight
# backbone_cp = np.hstack(
#     [
#         np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#     ]
# )
# backbone = Backbone(backbone_cp, reparameterize=True)
# cp_radius = find_cp_for_desired_radius(RADIUS, NUM_CP_PER_CROSS_SECTION)
# cs_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(
#     -1, 1
# )
# cs_cp = np.hstack((cp_radius * np.cos(cs_th), cp_radius * np.sin(cs_th)))

# cs_list = [
#     CrossSection(controlpoints=1.000 * cs_cp, position=position)
#     for position in np.linspace(0.0, 0.9, 6)
# ]
# p_straight = AxialComponent(
#     backbone,
#     cs_list,
#     smooth_with_post=False,
#     parent_axial_component=p_base,
#     position_along_parent=0.9,
# )
# # p_straight.mesh.show(smooth=False)

# # Sharp bend
# backbone_cp = np.hstack(
#     [
#         np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#     ]
# )
# shift = backbone_cp - backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
# R = scipy.spatial.transform.Rotation.from_euler(
#     "xyz", np.array([0, 0, np.pi / 4])
# ).as_matrix()
# shift_R = shift @ R
# transformed = shift_R + backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
# backbone_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[
#     NUM_CP_PER_BACKBONE // 2 :: -1, :
# ]
# backbone = Backbone(backbone_cp, reparameterize=True)
# cs_list = []
# cs_list.extend(
#     [
#         CrossSection(controlpoints=1.001 * cs_cp, position=position)
#         for position in [0.0, 0.2, 0.25]
#     ]
# )
# # Scale middle cross section to prevent tearing
# # Alternative would be to place middle cross section where vectors of other controlpoints meet. Must handle how to center along backbone.
# S = np.eye(2)
# S[0, 0] = 2
# cs_list.extend(
#     [
#         CrossSection(controlpoints=1.001 * cs_cp @ S, position=position)
#         for position in [0.5]
#     ]
# )
# cs_list.extend(
#     [
#         CrossSection(controlpoints=1.001 * cs_cp, position=position)
#         for position in [0.75, 0.8, 0.9]
#     ]
# )
# p_bend_sharp = AxialComponent(
#     backbone, cs_list, parent_axial_component=p_base, position_along_parent=0.9
# )
# # p_bend_sharp.mesh.show(smooth=False)

# # Half bend
# backbone_cp = np.hstack(
#     [
#         np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#     ]
# )
# shift = backbone_cp - backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
# R = scipy.spatial.transform.Rotation.from_euler(
#     "xyz", np.array([0, 0, np.pi / 2])
# ).as_matrix()
# shift_R = shift @ R
# transformed = shift_R + backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
# backbone_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[
#     NUM_CP_PER_BACKBONE // 2 :: -1, :
# ]
# backbone = Backbone(backbone_cp, reparameterize=True)
# cs_list = [
#     CrossSection(controlpoints=1.002 * cs_cp, position=position)
#     for position in [0.0, 0.3, 0.40, 0.5, 0.60, 0.7, 0.9]
# ]
# p_bend_half = AxialComponent(
#     backbone, cs_list, parent_axial_component=p_base, position_along_parent=0.9
# )
# # p_bend_half.mesh.show(smooth=False)

# # Weak bend
# backbone_cp = np.hstack(
#     [
#         np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#     ]
# )
# shift = backbone_cp - backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
# R = scipy.spatial.transform.Rotation.from_euler(
#     "xyz", np.array([0, 0, 3 * np.pi / 4])
# ).as_matrix()
# shift_R = shift @ R
# transformed = shift_R + backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
# backbone_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[
#     NUM_CP_PER_BACKBONE // 2 :: -1, :
# ]
# backbone = Backbone(backbone_cp, reparameterize=True)
# cs_list = [
#     CrossSection(controlpoints=1.003 * cs_cp, position=position)
#     for position in [0.0, 0.3, 0.40, 0.5, 0.60, 0.7, 0.9]
# ]
# p_bend_weak = AxialComponent(
#     backbone, cs_list, parent_axial_component=p_base, position_along_parent=0.9
# )
# # p_bend_weak.mesh.show(smooth=False)


# # Arc
# backbone_cp = approximate_arc(np.pi / 2, LENGTH)
# backbone_cp *= np.array([1, -1, 1])  # Flip about y axis
# backbone = Backbone(backbone_cp, reparameterize=True)
# cs_list = [
#     CrossSection(controlpoints=1.004 * cs_cp, position=position)
#     for position in np.linspace(0.0, 0.9, 10)
# ]
# p_arc = AxialComponent(
#     backbone,
#     cs_list,
#     smooth_with_post=False,
#     parent_axial_component=p_base,
#     position_along_parent=0.9,
# )
# # p_arc.mesh.show(smooth=False)


# ### Shapes ###

# # s1 = Shape([p_base, p_arc, p_bend_weak])
# # s2 = Shape([p_base, p_bend_weak])
# s3 = Shape([p_base, p_straight, p_arc])
# # s3.mesh.show()


# p_list = [None, p_straight, p_bend_sharp, p_bend_half, p_bend_weak, p_arc]
# import itertools

# p_dict = {
#     "None": None,
#     "p_straight": p_straight,
#     "p_bend_sharp": p_bend_sharp,
#     "p_bend_half": p_bend_half,
#     "p_bend_weak": p_bend_weak,
#     "p_arc": p_arc,
# }
# combs = list(itertools.combinations_with_replacement(p_dict.keys(), 2))
# angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
# for p1_name, p2_name in combs:

#     p1 = p_dict[p1_name]
#     p2 = p_dict[p2_name]

#     if (p1 == None) and (p2 == None):
#         continue
#     elif p1 == None:
#         print(p1_name, p2_name)
#         s = Shape([p_base, p2])
#     elif p2 == None:
#         print(p1_name, p2_name)
#         s = Shape([p_base, p1])
#     else:

#         # Rotate projections (exclude straight projection)
#         if (p1_name == "p_straight") and (p2_name == "p_straight"):
#             continue
#         elif (p1_name == "p_straight") or (p2_name == "p_straight"):
#             print(p1_name, p2_name)
#             s = Shape([p_base, p1, p2])
#         else:
#             for ang in angles:

#                 # Skip duplicate of same projection at same angle
#                 if (p1_name == p2_name) and (ang == 0 or ang == angles[-1]):
#                     continue

#                 # Apply rotation
#                 p1_rot = copy.deepcopy(p1)
#                 p1_rot.euler_angles = np.array([ang, 0, 0])

#                 # Recalculate axial component
#                 p1_rot.calc_points()

#                 # Apply small shift (to help with boolean union)
#                 p1_rot.mesh.vertices *= 1.001

#                 print(p1_name, p2_name, ang)
#                 s = Shape([p_base, p1_rot, p2])
#                 s.mesh.show()

#     s.mesh.show()

# # # Plot cross sections of shape
# # fig = plt.figure()
# # ax = plt.axes(projection="3d")
# # cp = p_arc.controlpoints
# # for i in range(8):
# #     xyz = cp[:, i, :]
# #     ax.plot(
# #         xyz[:, 0],
# #         xyz[:, 1],
# #         xyz[:, 2],
# #         "k-*",
# #     )
# # plt.show()


# # T = np.eye(4)
# # T[:3, 3] = [6.5, 1, 0]
# # box = trimesh.primitives.Box(extents=[5, 5, 5], transform=T)

# # scene = trimesh.Scene()
# # scene.add_geometry(box)
# # scene.add_geometry(p_bend_sharp.mesh)
# # scene.show()

# # # Fair mesh
# # neighbors = box.contains(p_bend_sharp.mesh.vertices)
# # neighbors = np.arange(0, len(neighbors))[neighbors]
# # faired_mesh = fair_mesh(p_bend_sharp.mesh, neighbors, 2)
# # faired_mesh.show(smooth=False)


# # fig = plt.figure()
# # ax = plt.axes(projection="3d")
# # t = np.linspace(0, 1, 100)
# # xyz = backbone.r(t)
# # ax.plot(
# #     xyz[:, 0],
# #     xyz[:, 1],
# #     xyz[:, 2],
# #     "k.",
# # )
# # plt.show()
