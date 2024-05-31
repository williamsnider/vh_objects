import trimesh
import numpy as np


# Script to generate stimulus set C (contains multi-joint stimuli, sheets)


# Linear segment
from pathlib import Path
import copy
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from objects.backbone import Backbone
from objects.shape import Shape
from objects.utilities import (
    approximate_arc,
    make_mesh,
    make_surface,
    calc_mesh_boolean_and_edges,
)
from scripts.sheets import construct_sheet, bend_sheet, make_base_cp, plot_arr
import trimesh
from scipy.spatial.transform.rotation import Rotation
from objects.shaft import Shaft
from scripts.stimulus_set_params import (
    NUM_CP_PER_BASE_SHEET,
    NUM_CP_PER_CROSS_SECTION,
    NUM_CS,
    NUM_CS_PER_SHEET,
    SEGMENT_LENGTH,
    X_WIDTH,
    VOLUMETRIC_RADII,
    SHEET_THICKNESS,
    POINT_RADII,
    POINT_ROUNDOVER_OFFSET,
    LEAF_RADII,
    APPENDAGE_LENGTH,
    SAVE_DIR,
    XYZ_OFFSET,
    ROUND_RADIUS,
    BOX_EXTENTS,
    BOX_TRANSLATION,
    TERMINATION_RADIUS,
    uu,
    vv,
)


def plot_cp(cp):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(cp.shape[0]):
        ax.plot(cp[i, :, 0], cp[i, :, 1], cp[i, :, 2], "b-*")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Set axis limits
    ax_min = np.min(cp)
    ax_max = np.max(cp)
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_zlim(ax_min, ax_max)
    plt.show()


def clip_along_axis(cp, axis, thickness):

    cp_clipped = np.zeros(cp.shape)

    for cs_num in range(cp.shape[0]):

        # Skip ends
        if cs_num == 0 or cs_num == cp.shape[0] - 1:
            cp_clipped[cs_num] = cp[cs_num]
            continue

        # Calculate local coordinate system
        cs_cp = cp[cs_num, :, :]
        P0 = np.mean(cs_cp, axis=0)
        P1 = cs_cp[0]
        P2 = cs_cp[1]
        vecT = P1 - P0
        vecT /= np.linalg.norm(vecT)
        vecN = np.cross(vecT, (P2 - P0) / np.linalg.norm(P2 - P0))
        vecN /= np.linalg.norm(vecN)
        vecB = np.cross(vecT, vecN)

        # Translate to origin
        cs_cp_t = cs_cp - P0

        # Rotate to align with xz plane
        Ra = np.hstack([vecT.reshape(3, 1), vecN.reshape(3, 1), vecB.reshape(3, 1)])
        goal = np.eye(3)
        R = np.dot(goal, np.linalg.inv(Ra))
        cs_cp_R = np.dot(R, cs_cp_t.T).T

        # Shrink along axis
        cs_cp_R[:, axis] = np.clip(cs_cp_R[:, axis], -thickness / 2, thickness / 2)

        # Undo transformations
        cs_cp_n = np.dot(R.T, cs_cp_R.T).T
        cs_cp_n += P0
        cp_clipped[cs_num, :, :] = cs_cp_n
    return cp_clipped


export_dir = "scripts/samples/"
shape_count = 0
##################
### Components ###
##################
c_dict = {}


# Base cylinder
base_cylinder_radius = 20
base_cylinder_height = 5
base_cylinder = trimesh.creation.cylinder(radius=base_cylinder_radius, height=base_cylinder_height, sections=20)
base_cylinder.apply_translation([0, 0, -base_cylinder_height])
c_dict["base_cylinder"] = base_cylinder

# Capsule
capsule_diameter = 5
capsule_length = 20
capsule_K0 = Shaft(
    capsule_length,
    1.0 * capsule_diameter,
    1.0 * capsule_diameter,
    1.0 * capsule_diameter,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)

# Tranform to be pointing upwards
T_point_z = np.eye(4)
T_point_z[:3, :3] = Rotation.from_euler("xyz", np.array([0, -np.pi / 2, 0])).as_matrix()

thickness = 4

# Capsule_K0 flattened
capsule_K0_F1_cp = capsule_K0.cp.copy()
capsule_K0_F1_cp[:, :, 2] = np.clip(capsule_K0_F1_cp[:, :, 2], -thickness / 2, thickness / 2)
capsule_K0_F1_surf = make_surface(capsule_K0_F1_cp)
capsule_K0_F1_mesh = make_mesh(capsule_K0_F1_surf, uu, vv)
capsule_K0_F1_mesh.apply_transform(T_point_z)

# Capsule_K0 flattened
capsule_K0_F2_cp = capsule_K0.cp.copy()
capsule_K0_F2_cp[:, :, 1] = np.clip(capsule_K0_F2_cp[:, :, 1], -thickness / 2, thickness / 2)
capsule_K0_F2_surf = make_surface(capsule_K0_F2_cp)
capsule_K0_F2_mesh = make_mesh(capsule_K0_F2_surf, uu, vv)
capsule_K0_F2_mesh.apply_transform(T_point_z)

# TODO: Fix the capsule length
print("TODO: Fix the capsule length.")
capsule_K1_length = capsule_length * 1.7
capsule_K1 = Shaft(
    capsule_K1_length,
    1.0 * capsule_diameter,
    1.0 * capsule_diameter,
    1.0 * capsule_diameter,
    theta=np.pi / 2,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)


capsule_K0.mesh.apply_transform(T_point_z)
capsule_K1.mesh.apply_transform(T_point_z)

c_dict["capsule_K0"] = capsule_K0.mesh
c_dict["capsule_K0_F1"] = capsule_K0_F1_mesh
c_dict["capsule_K0_F2"] = capsule_K0_F2_mesh
c_dict["capsule_K1"] = capsule_K1.mesh


# Capsule_K1 flattened
capsule_K1_F1_cp = capsule_K1.cp.copy()
capsule_K1_F1_cp = clip_along_axis(capsule_K1_F1_cp, 0, thickness)
capsule_K1_F1_surf = make_surface(capsule_K1_F1_cp)
capsule_K1_F1_mesh = make_mesh(capsule_K1_F1_surf, uu, vv)
capsule_K1_F1_mesh.apply_transform(T_point_z)
c_dict["capsule_K1_F1"] = capsule_K1_F1_mesh

capsule_K1_F2_cp = capsule_K1.cp.copy()
capsule_K1_F2_cp = clip_along_axis(capsule_K1_F2_cp, 2, thickness)
capsule_K1_F2_surf = make_surface(capsule_K1_F2_cp)
capsule_K1_F2_mesh = make_mesh(capsule_K1_F2_surf, uu, vv)
capsule_K1_F2_mesh.apply_transform(T_point_z)
c_dict["capsule_K1_F2"] = capsule_K1_F2_mesh


# Show everything in c_dict
scene = trimesh.Scene()
translation = 0
for key in c_dict:
    mesh_copy = copy.deepcopy(c_dict[key])
    mesh_copy.apply_translation([translation, 0, 0])
    translation += mesh_copy.bounding_box.extents[0]
    scene.add_geometry(mesh_copy)
scene.show(smooth=False)

s_components = []

mesh_fairing_distance = 0.25
from trimesh.transformations import rotation_matrix as rotvec2T

# for mesh_name in ["capsule_K1", "capsule_K1_F1", "capsule_K1_F2"]:

#     # Spikes
#     for num_meshes in [6, 8]:

#         if num_meshes == 6:
#             idx_list = ["0", "01", "012", "04", "0145", "012345"]
#             offset = np.pi / 3
#         elif num_meshes == 8:
#             idx_list = ["01", "02", "012", "0123", "0145", "0246", "012456", "01234567"]
#             offset = np.pi
#         else:
#             raise ValueError("num_meshes must be 6 or 8")

#         mesh_names = [mesh_name] * num_meshes
#         mesh_list = [base_cylinder]
#         T_list = [np.eye(4)]
#         for i in range(num_meshes):
#             mesh = copy.deepcopy(c_dict[mesh_names[i]])
#             mesh.apply_scale(1 + 0.001 * i)
#             mesh_list.append(mesh)

#             if i < (num_meshes // 2):
#                 T = rotvec2T(np.linspace(0, 2 * np.pi, num_meshes // 2, endpoint=False)[i], [0, 0, 1])

#             else:
#                 T1 = rotvec2T(
#                     np.linspace(0, 2 * np.pi, num_meshes // 2, endpoint=False)[i - num_meshes // 2] + offset, [0, 0, 1]
#                 )
#                 T2 = rotvec2T(np.pi, [1, 0, 0])
#                 T = T1 @ T2
#                 T[2, 3] += (capsule_length - capsule_diameter) * 2

#             T_list.append(T)

#         op_list = ["union"] * len(mesh_list)
#         description = "claw"
#         label = "D006"

#         for idx_string in idx_list:

#             idx = [int(i) for i in idx_string]
#             mesh_list_sub = [mesh_list[0]]
#             T_list_sub = [T_list[0]]
#             op_list_sub = [op_list[0]]
#             for i in idx:
#                 mesh_list_sub.append(mesh_list[i + 1])
#                 T_list_sub.append(T_list[i + 1])
#                 op_list_sub.append(op_list[i + 1])

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
#             s.mesh.export(Path(export_dir, str(shape_count).zfill(4) + ".stl"))
#             shape_count += 1

#             # s.mesh.show(smooth=False)

#     # Claw 8
#     for num_meshes in [6, 8]:

#         if num_meshes == 6:
#             idx_list = ["0", "01", "012", "04", "0145", "012345"]
#             offset = np.pi / 3
#         elif num_meshes == 8:
#             idx_list = ["01", "02", "012", "0123", "0145", "0246", "012456", "01234567"]
#             offset = np.pi
#         else:
#             raise ValueError("num_meshes must be 6 or 8")

#         mesh_names = [mesh_name] * num_meshes
#         mesh_list = [base_cylinder]
#         T_list = [np.eye(4)]
#         for i in range(num_meshes):
#             mesh = copy.deepcopy(c_dict[mesh_names[i]])
#             mesh.apply_scale(1 + 0.001 * i)
#             mesh_list.append(mesh)

#             if i < (num_meshes // 2):
#                 T1 = rotvec2T(np.linspace(0, 2 * np.pi, num_meshes // 2, endpoint=False)[i], [0, 0, 1])
#                 T2 = rotvec2T(np.pi / 2, [1, 0, 0])
#                 T = T1 @ T2
#                 T[2, 3] += 0

#             else:
#                 if num_meshes == 8:
#                     offset = np.pi
#                 elif num_meshes == 6:
#                     offset = np.pi / 3
#                 else:
#                     raise ValueError("num_meshes must be 6 or 8")
#                 T1 = rotvec2T(
#                     np.linspace(0, 2 * np.pi, num_meshes // 2, endpoint=False)[i - num_meshes // 2] + offset, [0, 0, 1]
#                 )
#                 T2 = rotvec2T(3 * np.pi / 2, [1, 0, 0])
#                 T = T1 @ T2
#                 T[2, 3] += (capsule_length - capsule_diameter) * 2

#             T_list.append(T)

#         op_list = ["union"] * len(mesh_list)
#         description = "claw"
#         label = "D006"

#         for idx_string in idx_list:

#             idx = [int(i) for i in idx_string]
#             mesh_list_sub = [mesh_list[0]]
#             T_list_sub = [T_list[0]]
#             op_list_sub = [op_list[0]]
#             for i in idx:
#                 mesh_list_sub.append(mesh_list[i + 1])
#                 T_list_sub.append(T_list[i + 1])
#                 op_list_sub.append(op_list[i + 1])

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
#             s.mesh.export(Path(export_dir, str(shape_count).zfill(4) + ".stl"))
#             shape_count += 1

################
### Straight ###
################
shape_count = 500
for mesh_name in ["capsule_K0", "capsule_K0_F1", "capsule_K0_F2"]:

    # Orthogonal
    mesh_names = ["base_cylinder"]
    mesh_names.extend([mesh_name] * 6)
    T_list = [np.eye(4)]

    # Component 1
    T = np.eye(4)
    T_list.append(T)

    # Component 2
    T = np.eye(4)
    T[2, 3] += capsule_length - capsule_diameter / 2  # shift up
    T_list.append(T)

    # Components 3,4,5,6 (rotations about Z-axis)
    for i in range(4):
        T1 = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [0, 0, 1])
        T2 = rotvec2T(np.pi / 2, [1, 0, 0])
        T = T1 @ T2
        T[2, 3] += capsule_length - capsule_diameter / 2
        T_list.append(T)

    mesh_list = []
    for i in range(len(mesh_names)):
        mesh_copy = copy.deepcopy(c_dict[mesh_names[i]])
        mesh_copy.apply_scale(1 + 0.001 * i)
        mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
        mesh_list.append(mesh_copy)

    op_list = ["union"] * len(mesh_list)
    description = "straight"
    label = "D006"

    idx_list = ["0", "02", "023", "024", "0234", "02345", "01", "012", "0123", "0124", "01234", "012345"]
    for idx_string in idx_list:

        idx = [int(i) for i in idx_string]
        mesh_list_sub = [mesh_list[0]]
        T_list_sub = [T_list[0]]
        op_list_sub = [op_list[0]]
        for i in idx:
            mesh_list_sub.append(mesh_list[i + 1])
            T_list_sub.append(T_list[i + 1])
            op_list_sub.append(op_list[i + 1])

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
        # s.mesh.show(smooth=False)

        s.mesh.export(Path(export_dir, str(shape_count).zfill(4) + ".stl"))
        shape_count += 1

    # 45up 90around
    mesh_names = ["base_cylinder"]
    mesh_names.extend([mesh_name] * 5)
    T_list = [np.eye(4)]

    # Component 1
    T = np.eye(4)
    T_list.append(T)

    # Components 2,3,4,5 (90deg rotations about z-axis, 45deg rotation about x-axis)
    for i in range(4):
        T1 = rotvec2T(np.linspace(0, 2 * np.pi, 4, endpoint=False)[i], [0, 0, 1])
        T2 = rotvec2T(np.pi / 4, [1, 0, 0])
        T = T1 @ T2
        T[2, 3] += capsule_length - capsule_diameter / 2
        T_list.append(T)

    mesh_list = []
    for i in range(len(mesh_names)):
        mesh_copy = copy.deepcopy(c_dict[mesh_names[i]])
        mesh_copy.apply_scale(1 + 0.001 * i)
        mesh_copy.apply_translation(np.array([0.001, 0.001, 0.001]) * i)
        mesh_list.append(mesh_copy)

    op_list = ["union"] * len(mesh_list)
    description = "straight"
    label = "D006"

    idx_list = ["0", "01", "012", "013", "0123", "01234"]
    for idx_string in idx_list:

        idx = [int(i) for i in idx_string]
        mesh_list_sub = [mesh_list[0]]
        T_list_sub = [T_list[0]]
        op_list_sub = [op_list[0]]
        for i in idx:
            mesh_list_sub.append(mesh_list[i + 1])
            T_list_sub.append(T_list[i + 1])
            op_list_sub.append(op_list[i + 1])

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
        # s.mesh.show(smooth=False)

        s.mesh.export(Path(export_dir, str(shape_count).zfill(4) + ".stl"))
        shape_count += 1
