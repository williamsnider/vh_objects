import trimesh
import numpy as np


# Script to generate stimulus set C (contains multi-joint stimuli, sheets)


# Linear segment
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

# from scripts.sheets import construct_sheet, bend_sheet, make_base_cp, plot_arr
import trimesh
from scipy.spatial.transform.rotation import Rotation
from objects.shaft import Shaft


uu = 50
vv = 50


def plot_cp(cp):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(cp.shape[0]):
        ax.plot(cp[i, :, 0], cp[i, :, 1], cp[i, :, 2], "b-*")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
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


NUM_CS = 5
NUM_CP_PER_CROSS_SECTION = 10

##################
### Components ###
##################
c_dict = {}


# Base cylinder
base_cylinder_radius = 20
base_cylinder_height = 0.5
base_cylinder = trimesh.creation.cylinder(radius=base_cylinder_radius, height=base_cylinder_height, sections=20)

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

# Tranform to be pointing upwards
T_point_z = np.eye(4)
T_point_z[:3, :3] = Rotation.from_euler("xyz", np.array([0, -np.pi / 2, 0])).as_matrix()
capsule_K0.mesh.apply_transform(T_point_z)
capsule_K1.mesh.apply_transform(T_point_z)

c_dict["capsule_K0"] = capsule_K0.mesh
c_dict["capsule_K1"] = capsule_K1.mesh

# Capsule_K1 flattened
thickness = 4
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

mesh_fairing_distance = 2
mesh_name = "capsule_K1_F2"

# Claw4
from trimesh.transformations import rotation_matrix as rotvec2T

num_meshes = 4
mesh_names = [mesh_name] * num_meshes
mesh_list = []
for i in range(num_meshes):
    mesh = copy.deepcopy(c_dict[mesh_names[i]])
    mesh.apply_scale(1 + 0.001 * i)
    mesh_list.append(mesh)
T_list = [rotvec2T(np.linspace(0, 2 * np.pi, num_meshes, endpoint=False)[i], [0, 0, 1]) for i in range(num_meshes)]
op_list = ["union"] * num_meshes
description = "claw"
label = "D000"

claw4_0 = [mesh_list[:1], T_list[:1], op_list[:1], label, description, "test", np.eye(4), mesh_fairing_distance]
claw4_01 = [mesh_list[:2], T_list[:2], op_list[:2], label, description, "test", np.eye(4), mesh_fairing_distance]
claw4_02 = [mesh_list[::2], T_list[::2], op_list[::2], label, description, "test", np.eye(4), mesh_fairing_distance]
claw4_012 = [mesh_list[:3], T_list[:3], op_list[:3], label, description, "test", np.eye(4), mesh_fairing_distance]
claw4_0123 = [mesh_list, T_list, op_list, label, description, "test", np.eye(4), mesh_fairing_distance]

# Claw3
num_meshes = 3
mesh_names = [mesh_name] * num_meshes
mesh_list = []
for i in range(num_meshes):
    mesh = copy.deepcopy(c_dict[mesh_names[i]])
    mesh.apply_scale(1 + 0.001 * i)
    mesh_list.append(mesh)
T_list = [rotvec2T(np.linspace(0, 2 * np.pi, num_meshes, endpoint=False)[i], [0, 0, 1]) for i in range(num_meshes)]
op_list = ["union"] * num_meshes
description = "claw"
label = "D001"

claw3_01 = [mesh_list[:2], T_list[:2], op_list[:2], label, description, "test", np.eye(4), mesh_fairing_distance]
claw3_012 = [mesh_list, T_list, op_list, label, description, "test", np.eye(4), mesh_fairing_distance]

# Spikes4
num_meshes = 4
mesh_names = [mesh_name] * num_meshes
mesh_list = []
for i in range(num_meshes):
    mesh = copy.deepcopy(c_dict[mesh_names[i]])
    mesh.apply_scale(1 + 0.001 * i)
    mesh_list.append(mesh)
T_list = [
    rotvec2T(np.linspace(0, 2 * np.pi, num_meshes, endpoint=False)[i], [0, 0, 1]) @ rotvec2T(np.pi / 2, [1, 0, 0])
    for i in range(num_meshes)
]
op_list = ["union"] * num_meshes
description = "spikes"
label = "D002"

spikes_4_0 = [mesh_list[:1], T_list[:1], op_list[:1], label, description, "test", np.eye(4), mesh_fairing_distance]
spikes4_01 = [mesh_list[:2], T_list[:2], op_list[:2], label, description, "test", np.eye(4), mesh_fairing_distance]
spikes4_02 = [mesh_list[::2], T_list[::2], op_list[::2], label, description, "test", np.eye(4), mesh_fairing_distance]
spikes4_012 = [mesh_list[:3], T_list[:3], op_list[:3], label, description, "test", np.eye(4), mesh_fairing_distance]
spikes4_0123 = [mesh_list, T_list, op_list, label, description, "test", np.eye(4), mesh_fairing_distance]

# Spikes 3
num_meshes = 3
mesh_names = [mesh_name] * num_meshes
mesh_list = []
for i in range(num_meshes):
    mesh = copy.deepcopy(c_dict[mesh_names[i]])
    mesh.apply_scale(1 + 0.001 * i)
    mesh_list.append(mesh)
T_list = [
    rotvec2T(np.linspace(0, 2 * np.pi, num_meshes, endpoint=False)[i], [0, 0, 1]) @ rotvec2T(np.pi / 2, [1, 0, 0])
    for i in range(num_meshes)
]
op_list = ["union"] * num_meshes
description = "spikes"
label = "D003"

spikes3_01 = [mesh_list[:2], T_list[:2], op_list[:2], label, description, "test", np.eye(4), mesh_fairing_distance]
spikes3_012 = [mesh_list, T_list, op_list, label, description, "test", np.eye(4), mesh_fairing_distance]

for claw in [spikes3_01, spikes3_012, spikes_4_0]:
    s = Shape(*claw)
    s.mesh.show(smooth=False)


# # scene = trimesh.Scene(base_cylinder)

# # # Claw stimulus
# # mesh_list = []
# # num_meshes = 3
# # for i in range(num_meshes):

# #     mesh = copy.deepcopy(capsule_K1.mesh)

# #     # Rotate about z-axis
# #     T = trimesh.transformations.rotation_matrix(np.linspace(0, 2 * np.pi, 3, endpoint=False)[i], [0, 0, 1])
# #     mesh.apply_transform(T)
# #     mesh.apply_scale(1 + 0.001 * i)

# #     mesh_list.append(mesh)

# # Leaf
# b_cp = approximate_arc(np.pi / 2, APPENDAGE_LENGTH, 5)
# b_cp = b_cp[:, [1, 2, 0]]  # Reorder
# b_cp[:, 0] *= -1  # Flip direction across yz axis
# b_appendage_K1 = Backbone(b_cp, reparameterize=True)

# # Flatten along z-axis

# # All z-values above thickness/2 are set to thickness/2
# leaf_K0_cp = capsule_K0.cp.copy()
# leaf_K0_cp[:, :, 2] = np.clip(leaf_K0_cp[:, :, 2], -thickness / 2, thickness / 2)
# leaf_K0_surf = make_surface(leaf_K0_cp)
# leaf_K0_mesh = make_mesh(leaf_K0_surf, uu, vv)
# leaf_K0_mesh.show(smooth=False)

# # Flatten along bend
# leaf_K1_cp = capsule_K1.cp.copy()

# cs_num = 7


# plot_cp(cp_new)


# # Transform to xy plane
# cs_cp = leaf_K1_cp[cs_num]
# centerpoint = np.mean(cs_cp, axis=0)
# vecT = centerpoint - np.array([0, 0, 0])
# vecT /= np.linalg.norm(vecT)
# vecN = np.cross(vecT, np.array([0, 0, 1]))
# vecN /= np.linalg.norm(vecN)
# vecB = np.cross(vecT, vecN)
# vecB /= np.linalg.norm(vecB)
# T = np.eye(4)
# T[:3, 0] = vecN
# T[:3, 1] = vecT
# T[:3, 2] = vecB
# T[:3, 3] = centerpoint
# T_inv = np.linalg.inv(T)
# cs_cp_h = np.hstack([cs_cp, np.ones((cs_cp.shape[0], 1))])
# cs_cp_h_xy = np.dot(T_inv, cs_cp_h.T).T[:, :3]


# scene.show()
