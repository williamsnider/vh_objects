# Axial components
from vh_objects.axial_component import AxialComponent
from vh_objects.shaft import Shaft
from vh_objects.utilities import make_mesh, make_surface, approximate_arc
from vh_objects.shape import Shape
import numpy as np
from scripts.stim_set_common import (
    create_scene,
    num_edge_cp,
    base_round_cp,
    top_round_cp,
    UU,
    VV,
    NUM_CS_PER_SHEET,
    load_cap,
    export_shape,
    slightly_deform_mesh,
    STL_DIR,
)
from scripts.sheets import make_base_cp, construct_sheet, bend_sheet, plot_arr
from vh_objects.backbone import Backbone
from scipy.spatial.transform import Rotation
import copy
from trimesh.transformations import rotation_matrix as rotvec2T
from pathlib import Path
from scripts.label_quartet import label_mesh
from scripts.make_gif import calc_dist_from_z_axis

AC_LENGTH = 25.5
AC_DIAMETER = 10
AC_FLAT_THICKNESS = 5
NUM_CS = 11
NUM_CP_PER_CROSS_SECTION = 50
AC_K_PERPENDICULAR = 1 / 20
LONG_FACTOR = 1.1265
base_cs_ellipse_factors = [2 / 3, 5 / 4]
mesh_fairing_distance = 1

# ac_round_K0
ac_round_K0_shape = Shaft(
    AC_LENGTH,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
ac_round_K0 = ac_round_K0_shape.mesh
T = np.eye(4)
R = Rotation.from_rotvec(np.array([0, -np.pi / 2, 0])).as_matrix()
T[:3, :3] = R
T[2, 3] -= AC_DIAMETER / 2  # Align spherical end to origin
ac_round_K0.apply_transform(T)


# ac_round_K0_long
ac_round_K0_long_shape = Shaft(
    AC_LENGTH * LONG_FACTOR,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
ac_round_K0_long = ac_round_K0_long_shape.mesh
T = np.eye(4)
R = Rotation.from_rotvec(np.array([0, -np.pi / 2, 0])).as_matrix()
T[:3, :3] = R
T[2, 3] -= AC_DIAMETER / 2  # Align spherical end to origin
TY45 = rotvec2T(np.pi / 4, [0, 1, 0])  # Align to 45 degrees
ac_round_K0_long.apply_transform(TY45 @ T)

# ac_round_K1
ac_round_K1_shape = Shaft(
    LONG_FACTOR * AC_LENGTH,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    theta=np.pi / 2,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
ac_round_K1 = ac_round_K1_shape.mesh
T = np.eye(4)
R = (
    Rotation.from_rotvec(np.array([0, -np.pi / 2, 0])).as_matrix()
    @ Rotation.from_rotvec(np.array([-np.pi / 2, 0, 0])).as_matrix()
)
T[:3, :3] = R
T[2, 3] -= AC_DIAMETER / 2  # Align spherical end to origin
ac_round_K1.apply_transform(T)


ac_round_K2 = ac_round_K1.copy()
ac_round_K2.apply_transform(rotvec2T(np.pi, [1, 0, 1]))


# ac_ellipse_K0
ac_ellipse_K0_shape = Shaft(
    AC_LENGTH,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
    base_cs_ellipse_factors=base_cs_ellipse_factors,
)
ac_ellipse_K0 = ac_ellipse_K0_shape.mesh
T = np.eye(4)
T[:3, :3] = R
T[2, 3] -= AC_DIAMETER / 2 * base_cs_ellipse_factors[0]  # Align spherical end to origin
ac_ellipse_K0.apply_transform(T)


# ac_ellipse_K1
ac_ellipse_K1_shape = Shaft(
    LONG_FACTOR * AC_LENGTH,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    theta=np.pi / 2,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
    base_cs_ellipse_factors=base_cs_ellipse_factors,
)
ac_ellipse_K1 = ac_ellipse_K1_shape.mesh
T = np.eye(4)
T[:3, :3] = R
T[2, 3] -= AC_DIAMETER / 2 * base_cs_ellipse_factors[0]  # Align spherical end to origin
ac_ellipse_K1.apply_transform(T)

ac_ellipse_K2 = ac_ellipse_K1.copy()
ac_ellipse_K2.apply_transform(rotvec2T(np.pi, [1, 0, 1]))

# ac_ellipse_K0_long
ac_ellipse_K0_long_shape = Shaft(
    LONG_FACTOR * AC_LENGTH,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    AC_DIAMETER / 2,
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
    base_cs_ellipse_factors=base_cs_ellipse_factors,
)
ac_ellipse_K0_long = ac_ellipse_K0_long_shape.mesh
T = np.eye(4)
T[:3, :3] = R
T[2, 3] -= AC_DIAMETER / 2 * base_cs_ellipse_factors[0]
TY45 = rotvec2T(np.pi / 4, [0, 1, 0])  # Align to 45 degrees
ac_ellipse_K0_long.apply_transform(TY45 @ T)


# ac_flat_K0
flat_length = 10
flat_x = np.linspace(0, 1, 3) * flat_length
flat_y = np.array([5, 5, 5])
flat_poly = np.polyfit(flat_x, flat_y, 2)
flat_cp = make_base_cp(flat_poly, flat_x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = flat_cp.mean(axis=0)
flat_cp = flat_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(flat_cp, sheet_thickness=AC_FLAT_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)
cp[:, :, 2] -= cp[:, :, 2].min()
surf = make_surface(cp)
ac_flat_K0 = make_mesh(surf, UU, VV)

# ac_p_K0
ac_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(0.000001, ac_height, 5)  # TODO: Fix this, very ugly
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K0 = Backbone(b_cp, reparameterize=True)

bent_cp = bend_sheet(cp, b_appendage_K0, ac_height, AC_K_PERPENDICULAR)
surf = make_surface(bent_cp)
ac_p_K0 = make_mesh(surf, UU, VV)

# ac_flat_K1
b_cp = approximate_arc(np.pi / 2, ac_height * 1.4, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)
bent_cp = bend_sheet(cp, b_appendage_K1, ac_height, 0)
surf = make_surface(bent_cp)
ac_flat_K1 = make_mesh(surf, UU, VV)

# ac_p_K1
bent_cp = bend_sheet(cp, b_appendage_K1, ac_height, AC_K_PERPENDICULAR)
surf = make_surface(bent_cp)
ac_p_K1 = make_mesh(surf, UU, VV)

# ac_n_K1
bent_cp = bend_sheet(cp, b_appendage_K1, ac_height, -AC_K_PERPENDICULAR)
surf = make_surface(bent_cp)
ac_n_K1 = make_mesh(surf, UU, VV)


mesh_dict = {
    "cap": load_cap(),
    "ac_round_K0": ac_round_K0,
    "ac_round_K1": ac_round_K1,
    "ac_ellipse_K0": ac_ellipse_K0,
    "ac_ellipse_K1": ac_ellipse_K1,
    "ac_ellipse_K2": ac_ellipse_K2,
    "ac_ellipse_K0_long": ac_ellipse_K0_long,
    # "ac_flat_K0": ac_flat_K0,
    # "ac_p_K0": ac_p_K0,
    # "ac_flat_K1": ac_flat_K1,
    # "ac_p_K1": ac_p_K1,
    # "ac_n_K1": ac_n_K1,
    "ac_round_K0_long": ac_round_K0_long,
    "ac_round_K2": ac_round_K2,
}

create_scene(mesh_dict)


s_list = []
################
## 2 SEGMENT ###
################
# for cs_type in ["round", "ellipse"]:
#     ellipse_T23_shift = AC_LENGTH - AC_DIAMETER / 2 - AC_DIAMETER / 2 + 0.5
#     ellipse_T23_shift_double = ellipse_T23_shift - 3
#     # Straight-Straight
#     mesh_names = ["ac_" + cs_type + "_K0", "ac_" + cs_type + "_K0"]

#     for j in range(3):

#         mesh_list = [mesh_dict[mesh_names[i]] for i in range(len(mesh_names))]
#         mesh_list = slightly_deform_mesh(mesh_list)
#         # for i in range(len(mesh_names)):
#         #     mesh_copy = copy.deepcopy(mesh_dict[mesh_names[i]])
#         #     mesh_copy.apply_scale(1 + 0.003 * i)
#         #     mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
#         #     mesh_list.append(mesh_copy)

#         T_list = [np.eye(4)]  # Limb 1
#         T2 = rotvec2T(np.linspace(np.pi / 2, 0, 3)[j], [0, 1, 0])

#         if cs_type == "ellipse":
#             T2[2, 3] = AC_LENGTH - AC_DIAMETER * base_cs_ellipse_factors[0]
#         elif cs_type == "round":
#             T2[2, 3] = AC_LENGTH - AC_DIAMETER
#         else:
#             raise ValueError("Invalid cross-section type")

#         T_list.append(T2)  # Limb 2

#         op_list = ["union"] * len(mesh_list)
#         description = "straight"
#         label = "D006"

#         # Add in cap
#         mesh_list.append(mesh_dict["cap"])
#         T = np.eye(4)
#         # T[2, 3] = -AC_DIAMETER / 3
#         T_list.append(T)
#         op_list.append("union")

#         claw6 = [
#             mesh_list,
#             T_list,
#             op_list,
#             label,
#             description,
#             "test",
#             np.eye(4),
#             mesh_fairing_distance,
#         ]

#         s = Shape(*claw6)

#         s_list.append(s)
#         # s.mesh.show(smooth=False)

#     # Straight - Curved
#     mesh_names = ["ac_" + cs_type + "_K0", "ac_" + cs_type + "_K1"]
#     T = np.eye(4)
#     T[2, 3] = AC_LENGTH - AC_DIAMETER
#     T_overall = [T]

#     TY90 = rotvec2T(np.pi / 2, [0, 1, 0])
#     for th, i in enumerate(np.linspace(0, 2 * np.pi, 2, endpoint=False)):
#         T = rotvec2T(i, [1, 0, 0]) @ TY90
#         if cs_type == "ellipse":
#             T[2, 3] = AC_LENGTH - AC_DIAMETER * base_cs_ellipse_factors[0]
#         elif cs_type == "round":
#             T[2, 3] = AC_LENGTH - AC_DIAMETER
#         else:
#             raise ValueError("Invalid cross-section type")

#         T_overall.append(T)

#     TX90 = rotvec2T(np.pi / 2, [1, 0, 0])
#     for th, i in enumerate(np.linspace(0, 2 * np.pi, 2, endpoint=False)):
#         T = rotvec2T(i, [0, 1, 0]) @ TX90
#         if cs_type == "ellipse":
#             T[2, 3] = ellipse_T23_shift
#         elif cs_type == "round":
#             T[2, 3] = AC_LENGTH - AC_DIAMETER
#         else:
#             raise ValueError("Invalid cross-section type")
#         T_overall.append(T)

#     for T in T_overall:

#         mesh_list = [mesh_dict[mesh_names[i]] for i in range(len(mesh_names))]
#         mesh_list = slightly_deform_mesh(mesh_list)
#         # mesh_list = []
#         # for i in range(len(mesh_names)):
#         #     mesh_copy = copy.deepcopy(mesh_dict[mesh_names[i]])
#         #     mesh_copy.apply_scale(1 + 0.003 * i)
#         #     mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
#         #     mesh_list.append(mesh_copy)

#         T_list = [np.eye(4)]  # Limb 1
#         T_list.append(T)  # Limb 2

#         op_list = ["union"] * len(mesh_list)
#         description = "straight"
#         label = "D006"

#         # Add in cap
#         mesh_list.append(mesh_dict["cap"])
#         T = np.eye(4)
#         # T[2, 3] = -AC_DIAMETER / 3
#         T_list.append(T)
#         op_list.append("union")

#         claw6 = [
#             mesh_list,
#             T_list,
#             op_list,
#             label,
#             description,
#             "test",
#             np.eye(4),
#             mesh_fairing_distance,
#         ]

#         s = Shape(*claw6)
#         s_list.append(s)
#         # s.mesh.show(smooth=False)

#     # Curved-Straight
#     mesh_names = ["ac_" + cs_type + "_K1", "ac_" + cs_type + "_K0"]
#     T = np.eye(4)
#     T[2, 3] = AC_LENGTH - AC_DIAMETER
#     T_overall = [T]
#     for i in range(2):
#         T = rotvec2T(np.linspace(0, 2 * np.pi, 2, endpoint=False)[i], [0, 0, 1]) @ rotvec2T(np.pi / 2, [0, 1, 0])
#         if cs_type == "ellipse":
#             # T[2, 3] = AC_LENGTH - AC_DIAMETER * base_cs_ellipse_factors[0]
#             T[2, 3] = ellipse_T23_shift
#         elif cs_type == "round":
#             T[2, 3] = AC_LENGTH - AC_DIAMETER
#         else:
#             raise ValueError("Invalid cross-section type")
#         T_overall.append(T)

#     for i in range(2):
#         T = rotvec2T(np.linspace(0, 2 * np.pi, 2, endpoint=False)[i], [0, 0, 1]) @ rotvec2T(np.pi / 2, [1, 0, 0])
#         if cs_type == "ellipse":
#             T[2, 3] = ellipse_T23_shift_double
#             pass
#         elif cs_type == "round":
#             T[2, 3] = AC_LENGTH - AC_DIAMETER
#             pass
#         else:
#             raise ValueError("Invalid cross-section type")
#         T_overall.append(T)

#     for T in T_overall:

#         mesh_list = [mesh_dict[mesh_names[i]] for i in range(len(mesh_names))]
#         mesh_list = slightly_deform_mesh(mesh_list)
#         # mesh_list = []
#         # for i in range(len(mesh_names)):
#         #     mesh_copy = copy.deepcopy(mesh_dict[mesh_names[i]])
#         #     mesh_copy.apply_scale(1 + 0.003 * i)
#         #     mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
#         #     mesh_list.append(mesh_copy)

#         T1 = rotvec2T(np.pi, [0, 1, 0])  # Limb 1
#         T1[2, 3] = AC_LENGTH - AC_DIAMETER
#         T_list = [T1]
#         T_list.append(T)  # Limb 2

#         op_list = ["union"] * len(mesh_list)
#         description = "straight"
#         label = "D006"

#         # Add in cap
#         mesh_list.append(mesh_dict["cap"])
#         T = np.eye(4)
#         # T[2, 3] = -AC_DIAMETER / 3
#         T_list.append(T)
#         op_list.append("union")

#         claw6 = [
#             mesh_list,
#             T_list,
#             op_list,
#             label,
#             description,
#             "test",
#             np.eye(4),
#             mesh_fairing_distance,
#         ]

#         s = Shape(*claw6)
#         s_list.append(s)
#         # s.mesh.show(smooth=False)

#     # Curved-Curved
#     mesh_names = ["ac_" + cs_type + "_K1", "ac_" + cs_type + "_K1"]
#     T_overall = []

#     # Rotations about Z axis
#     for i in range(4):

#         # Skip rotations that result in misaligned ellipses
#         if cs_type == "ellipse" and i % 2 == 1:
#             continue

#         T = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [0, 0, 1])
#         T[2, 3] = AC_LENGTH - AC_DIAMETER
#         T_overall.append(T)

#     # Rotations about +X axis
#     for i in range(4):

#         # Skip rotations that result in misaligned ellipses
#         if cs_type == "ellipse" and i % 2 == 1:
#             continue

#         T = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [1, 0, 0]) @ rotvec2T(np.pi / 2, [0, 1, 0])

#         if cs_type == "ellipse":
#             T[2, 3] = ellipse_T23_shift
#         elif cs_type == "round":
#             T[2, 3] = AC_LENGTH - AC_DIAMETER
#         else:
#             raise ValueError("Invalid cross-section type")

#         T_overall.append(T)

#     # Rotations about +Y axis
#     for i in range(4):

#         # Skip rotations that result in misaligned ellipses
#         if cs_type == "ellipse" and i % 2 == 1:
#             continue

#         T = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [0, 1, 0]) @ rotvec2T(np.pi / 2, [1, 0, 0])
#         if cs_type == "ellipse":
#             T[2, 3] = ellipse_T23_shift_double
#         elif cs_type == "round":
#             T[2, 3] = AC_LENGTH - AC_DIAMETER
#         else:
#             raise ValueError("Invalid cross-section type")
#         T_overall.append(T)

#     # Rotatoins about -X axis
#     for i in range(4):

#         # Skip rotations that result in misaligned ellipses
#         if cs_type == "ellipse" and i % 2 == 1:
#             continue

#         T = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [1, 0, 0]) @ rotvec2T(-np.pi / 2, [0, 1, 0])
#         T[2, 3] = AC_LENGTH - AC_DIAMETER

#         T_overall.append(T)

#     # Rotations about -Y axis
#     for i in range(4):

#         # Skip rotations that result in misaligned ellipses
#         if cs_type == "ellipse" and i % 2 == 1:
#             continue

#         T = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [0, 1, 0]) @ rotvec2T(-np.pi / 2, [1, 0, 0])
#         if cs_type == "ellipse":
#             T[2, 3] = ellipse_T23_shift_double
#         elif cs_type == "round":
#             T[2, 3] = AC_LENGTH - AC_DIAMETER
#         else:
#             raise ValueError("Invalid cross-section type")
#         T_overall.append(T)

#     for T in T_overall:

#         mesh_list = [mesh_dict[mesh_names[i]] for i in range(len(mesh_names))]
#         mesh_list = slightly_deform_mesh(mesh_list)
#         # mesh_list = []
#         # for i in range(len(mesh_names)):
#         #     mesh_copy = copy.deepcopy(mesh_dict[mesh_names[i]])
#         #     mesh_copy.apply_scale(1 + 0.003 * i)
#         #     mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
#         #     mesh_list.append(mesh_copy)

#         T1 = rotvec2T(np.pi, [0, 1, 0])  # Limb 1
#         T1[2, 3] = AC_LENGTH - AC_DIAMETER
#         T_list = [T1]
#         T_list.append(T)  # Limb 2

#         op_list = ["union"] * len(mesh_list)
#         description = "straight"
#         label = "D006"

#         # Add in cap
#         mesh_list.append(mesh_dict["cap"])
#         T = np.eye(4)
#         # T[2, 3] = -AC_DIAMETER / 3
#         T_list.append(T)
#         op_list.append("union")

#         claw6 = [
#             mesh_list,
#             T_list,
#             op_list,
#             label,
#             description,
#             "test",
#             np.eye(4),
#             mesh_fairing_distance,
#         ]

#         s = Shape(*claw6)
#         s_list.append(s)

#         # s.mesh.show(smooth=False)

#################
### 3 SEGMENT ###
#################

configs = np.array(
    [
        [0, 120, 120],
        [0, 60, 240],
        # [0, 240, 60],  # Symmetric to [0,60,240]
        [0, 60, 60],
        [0, 60, 180],
        [0, 90, 180],
        [0, 180, 90],
        # [0, 90, 90], # Symmetric to [0,180,90]
    ]
).astype("float")

configs /= 180 / np.pi


for config in configs:
    for cs_type in ["ellipse", "round"]:

        for curve_direction in ["flat", "up", "down", "updown", "downup"]:

            if curve_direction in ["up", "down"]:
                mesh_names = ["ac_" + cs_type + "_K0", "ac_" + cs_type + "_K1", "ac_" + cs_type + "_K1"]
            elif curve_direction == "updown":
                mesh_names = ["ac_" + cs_type + "_K0", "ac_" + cs_type + "_K1", "ac_" + cs_type + "_K2"]
            elif curve_direction == "downup":
                mesh_names = ["ac_" + cs_type + "_K0", "ac_" + cs_type + "_K2", "ac_" + cs_type + "_K1"]
            elif curve_direction == "flat":
                mesh_names = ["ac_" + cs_type + "_K0", "ac_" + cs_type + "_K0", "ac_" + cs_type + "_K0"]

            T_list = [rotvec2T(config[0] + np.pi, [0, 1, 0])]

            if curve_direction == "down":
                down_angle2 = np.pi
                down_angle3 = np.pi
            elif curve_direction == "updown":
                down_angle2 = np.pi
                down_angle3 = np.pi
            elif curve_direction == "downup":
                down_angle2 = 0
                down_angle3 = 0
            else:
                down_angle2 = 0
                down_angle3 = 0

            # Second limb
            idx = 1
            T = rotvec2T(config[idx], [0, 1, 0]) @ T_list[0] @ rotvec2T(down_angle2, [0, 0, 1])  # Rotate to next angle
            T_list.append(T)

            # Third Limb
            idx = 2
            T = (
                rotvec2T(np.sum(config[:]), [0, 1, 0]) @ T_list[0] @ rotvec2T(down_angle3 + np.pi, [0, 0, 1])
            )  # Rotate to next angle
            T_list.append(T)

            # Shift all T up
            for i in range(3):
                if cs_type == "round":
                    T_list[i][2, 3] += AC_LENGTH - AC_DIAMETER * 7 / 8
                elif cs_type == "ellipse":
                    T_list[i][2, 3] += AC_LENGTH - AC_DIAMETER / 2
                else:
                    raise ValueError("Invalid cross-section type")

            # Transform meshes slightly
            mesh_list = [mesh_dict[mesh_names[i]] for i in range(len(mesh_names))]
            mesh_list = slightly_deform_mesh(mesh_list)
            # mesh_list = []
            # for i in range(3):
            #     mesh_copy = copy.deepcopy(mesh_dict[mesh_names[i]])
            #     mesh_copy.apply_scale(1 + 0.003 * i)
            #     mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
            #     mesh_list.append(mesh_copy)

            op_list = ["union"] * len(mesh_list)
            description = "straight"
            label = "D006"

            # Add in cap
            mesh_list.append(mesh_dict["cap"])
            T = np.eye(4)
            # T[2, 3] = -AC_DIAMETER / 3
            T_list.append(T)
            op_list.append("union")

            claw6 = [
                mesh_list,
                T_list,
                op_list,
                label,
                description,
                "test",
                np.eye(4),
                mesh_fairing_distance,
            ]

            s = Shape(*claw6)
            s_list.append(s)
            # s.mesh.show(smooth=False)

            if calc_dist_from_z_axis(s.mesh) > 22.6:
                pass


#################################
### 8 SEGMENT COPLANAR / QUAD ###
#################################
num_components = 8
for cs_type in ["ellipse", "round"]:

    for curve_direction in ["flat", "up", "down"]:

        if curve_direction == "up":
            mesh_names = ["ac_" + cs_type + "_K1" for _ in range(num_components)]
        elif curve_direction == "flat":
            mesh_names = ["ac_" + cs_type + "_K0_long" for _ in range(num_components)]
        elif curve_direction == "down":
            mesh_names = ["ac_" + cs_type + "_K2" for _ in range(num_components)]
        else:
            raise ValueError("Invalid curve direction")

        T_list = []
        for i in range(4):

            # Translate such that the mesh forms 1/4 of the final shape
            TA = np.eye(4)
            TA[0, 3] = -(mesh_dict[mesh_names[i]].bounds[1, 0] + mesh_dict[mesh_names[i]].bounds[0, 0])

            # Rotate about y-axis
            offset = np.pi
            TB = rotvec2T(np.linspace(0 + offset, 2 * np.pi + offset, 4, endpoint=False)[i], [0, 1, 0])

            T = TB @ TA

            if curve_direction in ["up", "down"]:
                T[2, 3] += AC_LENGTH - AC_DIAMETER
            elif curve_direction == "flat":
                T[2, 3] += AC_LENGTH - AC_DIAMETER

            T_list.append(T)

        # Add T's for quad
        T_list.append(rotvec2T(np.pi / 2, [0, 0, 1]) @ T_list[0])
        T_list.append(rotvec2T(np.pi / 2, [0, 0, 1]) @ T_list[1])
        T_list.append(rotvec2T(np.pi / 2, [0, 0, 1]) @ T_list[2])
        T_list.append(rotvec2T(np.pi / 2, [0, 0, 1]) @ T_list[3])
        # # Test scene
        # import trimesh
        # scene = trimesh.Scene()
        # for i in range(4):
        #     new_mesh = mesh_dict[mesh_names[3]].copy()
        #     new_mesh.apply_transform(T_list[i])
        #     scene.add_geometry(new_mesh)
        # scene.show()

        # Apply transformations
        mesh_list = [mesh_dict[mesh_names[i]] for i in range(len(mesh_names))]
        mesh_list = slightly_deform_mesh(mesh_list)

        # # Test scene
        # import trimesh
        # scene = trimesh.Scene()
        # for i in range(4):
        #     new_mesh = mesh_list[i].copy()
        #     new_mesh.apply_transform(T_list[i])
        #     scene.add_geometry(new_mesh)
        # scene.show()

        # mesh_list = []
        # for i in range(len(mesh_names)):
        #     mesh_copy = copy.deepcopy(mesh_dict[mesh_names[i]])
        #     mesh_copy.apply_scale(1 + 0.003 * i)
        #     mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
        #     # mesh_copy.apply_transform(T_list[i])
        #     mesh_list.append(mesh_copy)

        op_list = ["union"] * len(mesh_list)
        description = "straight"
        label = "D006"

        # Loop around
        idx_list = ["0", "01", "012", "0123", "014", "0145", "012347", "01234567"]
        mesh_fairing_distance = 1
        for idx_string in idx_list:

            idx = [int(i) for i in idx_string]

            mesh_list_sub = []
            T_list_sub = []
            op_list_sub = []
            for i in idx:
                mesh_list_sub.append(mesh_list[i].copy())
                T_list_sub.append(T_list[i].copy())
                op_list_sub.append(op_list[i])

            # Shift all up slightly to keep tracking point visible
            for i in range(len(idx)):
                T_list_sub[i][2, 3] += AC_DIAMETER / 4

            # Add in cap
            mesh_list_sub.append(mesh_dict["cap"])
            T = np.eye(4)
            # T[2, 3] = -AC_DIAMETER / 3
            T_list_sub.append(T)
            op_list_sub.append("union")

            claw6 = [
                mesh_list_sub,
                T_list_sub,
                op_list_sub,
                label,
                description,
                "test",
                np.eye(4),
                mesh_fairing_distance,
            ]

            s = Shape(*claw6)
            s_list.append(s)

            # s.mesh.show(smooth=False)

        # Open
        TA = np.eye(4)
        TA[0, 3] = -(mesh_dict[mesh_names[i]].bounds[1, 0] + mesh_dict[mesh_names[i]].bounds[0, 0])

        TZ = rotvec2T(np.pi, [0, 0, 1])

        T0 = TA
        T1 = TZ @ T0
        T2 = rotvec2T(-np.pi / 2, [0, 1, 0])

        if cs_type == "ellipse":
            T2[2, 3] = AC_LENGTH - AC_DIAMETER * base_cs_ellipse_factors[0]
        else:
            T2[2, 3] = AC_LENGTH - AC_DIAMETER
        T3 = TZ @ T2
        T_list = [T0, T1, T2, T3]

        # Expand T_list for quad
        T_list.append(rotvec2T(np.pi / 2, [0, 0, 1]) @ T_list[0])
        T_list.append(rotvec2T(np.pi / 2, [0, 0, 1]) @ T_list[1])
        T_list.append(rotvec2T(np.pi / 2, [0, 0, 1]) @ T_list[2])
        T_list.append(rotvec2T(np.pi / 2, [0, 0, 1]) @ T_list[3])

        idx_list = ["0", "01", "012", "0123", "014", "0145", "0167", "012346", "01234567"]
        for idx_string in idx_list:
            idx = [int(i) for i in idx_string]

            mesh_list_sub = []
            T_list_sub = []
            op_list_sub = []
            for i in idx:
                mesh_list_sub.append(mesh_list[i])
                T_list_sub.append(T_list[i])
                op_list_sub.append(op_list[i])

            # # Add in extra segment so the post connects with the shape
            mesh_list_sub.append(mesh_dict["ac_round_K0"])
            T_extra = np.eye(4)
            T_extra[2, 3] = AC_DIAMETER / 8
            T_list_sub.append(T_extra)
            op_list_sub.append("union")

            # Add in cap
            mesh_list_sub.append(mesh_dict["cap"])
            T = np.eye(4)
            # T[2, 3] = -AC_DIAMETER / 3 + 0.1
            # T[:2, 3] = np.array([0.1, 0.1])  # TODO: FIX THIS - ugly
            T_list_sub.append(T)
            op_list_sub.append("union")

            claw6 = [
                mesh_list_sub,
                T_list_sub,
                op_list_sub,
                label,
                description,
                "test",
                np.eye(4),
                mesh_fairing_distance,
            ]
            s = Shape(*claw6)
            s_list.append(s)

            # s.mesh.show(smooth=False)

###########
### Fan ###
###########

for cs_type in ["round", "ellipse"]:

    mesh_names = [
        "ac_" + "round" + "_K0",
        "ac_" + cs_type + "_K1",
        "ac_" + cs_type + "_K1",
        "ac_" + cs_type + "_K1",
        "ac_" + cs_type + "_K1",
    ]

    # Component 1 straight up
    T_list = [np.eye(4)]

    # Components 2,3,4,5 (90deg rotations about y-axis then z-axis)
    for i in range(4):
        T1 = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [0, 0, 1])
        T2 = rotvec2T(np.pi / 2, [1, 0, 0])
        T = T1 @ T2
        if cs_type == "ellipse":
            T[2, 3] = AC_LENGTH - AC_DIAMETER * base_cs_ellipse_factors[0]
        else:
            T[2, 3] = AC_LENGTH - AC_DIAMETER
        T_list.append(T)

    # Apply transformations
    mesh_list = [mesh_dict[mesh_names[i]] for i in range(len(mesh_names))]
    mesh_list = slightly_deform_mesh(mesh_list)
    mesh_list = slightly_deform_mesh(mesh_list)

    op_list = ["union"] * len(mesh_list)
    description = "straight"
    label = "D006"
    idx_list = ["01", "012", "013", "0123", "01234"]
    mesh_fairing_distance = 1

    for idx_string in idx_list:

        idx = [int(i) for i in idx_string]

        mesh_list_sub = []
        T_list_sub = []
        op_list_sub = []
        for i in idx:
            mesh_list_sub.append(mesh_list[i])
            T_list_sub.append(T_list[i])
            op_list_sub.append(op_list[i])

        # Add in cap
        mesh_list_sub.append(mesh_dict["cap"])
        T = np.eye(4)
        # T[2, 3] = -AC_DIAMETER / 3 + 0.1
        T_list_sub.append(T)
        op_list_sub.append("union")

        claw6 = [
            mesh_list_sub,
            T_list_sub,
            op_list_sub,
            label,
            description,
            "test",
            np.eye(4),
            mesh_fairing_distance,
        ]

        s = Shape(*claw6)
        s_list.append(s)

        # s.mesh.show(smooth=False)
        if calc_dist_from_z_axis(s.mesh) > 22.6:
            pass
            print(calc_dist_from_z_axis(s.mesh))

# ###############
# ### DREIDLE ###
# ###############

# for cs_type in [
#     # "ellipse",
#     "round",
# ]:
#     for curve_direction in [
#         "flat",
#         # "up",
#         # "down",
#     ]:

#         if curve_direction == "up":
#             mesh_names = [
#                 "ac_round_K0",
#                 "ac_" + cs_type + "_K1",
#                 "ac_" + cs_type + "_K1",
#                 "ac_" + cs_type + "_K1",
#                 "ac_" + cs_type + "_K1",
#                 "ac_round_K0",
#             ]
#         elif curve_direction == "flat":
#             mesh_names = ["ac_round_K0" for _ in range(6)]
#         elif curve_direction == "down":
#             mesh_names = [
#                 "ac_round_K0",
#                 "ac_" + cs_type + "_K1",
#                 "ac_" + cs_type + "_K1",
#                 "ac_" + cs_type + "_K1",
#                 "ac_" + cs_type + "_K1",
#                 "ac_round_K0",
#             ]
#         else:
#             raise ValueError("Invalid curve direction")

#         # Component 1 straight up
#         T_list = [np.eye(4)]

#         # Components 2,3,4,5 (90deg rotations about z-axis)
#         for i in range(4):
#             T1 = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [0, 0, 1])
#             if curve_direction == "up":
#                 T2 = rotvec2T(-np.pi / 2, [0, 1, 0])
#             elif curve_direction == "flat":
#                 T2 = rotvec2T(np.pi / 2, [1, 0, 0])
#             elif curve_direction == "down":
#                 T2 = rotvec2T(np.pi / 2, [0, 1, 0])
#             else:
#                 raise ValueError("Invalid curve direction")

#             T = T1 @ T2

#             if cs_type == "ellipse":
#                 T[2, 3] += AC_LENGTH - AC_DIAMETER / 2
#                 pass
#             elif cs_type == "round":
#                 T[2, 3] += AC_LENGTH - AC_DIAMETER
#                 pass
#             else:
#                 raise ValueError("Invalid cross-section type")
#             T_list.append(T)

#         # Component 6 (straight up)
#         T = np.eye(4)
#         T[2, 3] += AC_LENGTH - AC_DIAMETER
#         T_list.append(T)

#         # Apply transformations
#         mesh_list = [mesh_dict[mesh_names[i]] for i in range(len(mesh_names))]
#         mesh_list = slightly_deform_mesh(mesh_list)
#         # mesh_list = []
#         # for i in range(len(mesh_names)):
#         #     mesh_copy = copy.deepcopy(mesh_dict[mesh_names[i]])
#         #     mesh_copy.apply_scale(1 + 0.003 * i)
#         #     mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
#         #     # mesh_copy.apply_transform(T_list[i])
#         #     mesh_list.append(mesh_copy)

#         op_list = ["union"] * len(mesh_list)
#         description = "straight"
#         label = "D006"

#         # 90deg all around

#         idx_list = ["01", "012", "013", "0123", "01234", "015", "0125", "0135", "01235", "012345"]
#         mesh_fairing_distance = 3
#         for idx_string in idx_list:

#             idx = [int(i) for i in idx_string]

#             mesh_list_sub = []
#             T_list_sub = []
#             op_list_sub = []
#             for i in idx:
#                 mesh_list_sub.append(mesh_list[i])
#                 T_list_sub.append(T_list[i])
#                 op_list_sub.append(op_list[i])

#             # Add in cap
#             mesh_list_sub.append(mesh_dict["cap"])
#             T = np.eye(4)
#             # T[2, 3] = -AC_DIAMETER / 3 + 0.1
#             T_list_sub.append(T)
#             op_list_sub.append("union")

#             claw6 = [
#                 mesh_list_sub,
#                 T_list_sub,
#                 op_list_sub,
#                 label,
#                 description,
#                 "test",
#                 np.eye(4),
#                 mesh_fairing_distance,
#             ]

#             s = Shape(*claw6)
#             s_list.append(s)

#             # s.mesh.show(smooth=False)


# Label parameters
font_height = 3.5
LABEL_DEPTH = 1.5
T_label = np.eye(4)
T_label[:3, :3] = Rotation.from_euler("yzx", np.array([np.pi, -np.pi / 2, 0])).as_matrix()
T_label[0, 3] = 9
T_label[2, 3] = mesh_dict["cap"].bounds[0][2] + LABEL_DEPTH


save_dir = Path(STL_DIR, "axial_component")
for i, s in enumerate(s_list):
    label = "F" + str(i).zfill(3)
    try:
        s.mesh = label_mesh(s.mesh, label, T_label, LABEL_DEPTH, font_height, "difference")
        export_shape(s, save_dir, label)
    except:
        print(f"Failed to save {label}")
        continue
