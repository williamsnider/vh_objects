from scripts.torso import comp_dict, plot_components, sf_radius, torso_length, torso_radius
from objects.shape import Shape
import numpy as np
import trimesh
from pathlib import Path

# plot_components(comp_dict)

pos_K0_A = np.array([-torso_radius * 1.5, 0, 2 * torso_length / 3])
pos_K0_B = np.array([torso_radius * 1.5, 0, 0 * torso_length / 3])
pos_K0_C = pos_K0_A * np.array([-1, 1, 1])
pos_K0_D = pos_K0_B * np.array([-1, 1, 1])

pos_K1_A = pos_K0_A + np.array([0, torso_radius * 1.5, 0])
pos_K1_B = pos_K0_B + np.array([0, torso_radius, 0])
pos_K1_C = pos_K0_C + np.array([0, torso_radius * 1.5, 0])
pos_K1_D = pos_K0_D + np.array([0, torso_radius, 0])
T_eye = np.eye(4)

T_K0_A_U = trimesh.transformations.euler_matrix(np.pi / 2, 0, -np.pi / 2)
T_K0_A_U[:3, 3] = pos_K0_A

T_K0_A_S = trimesh.transformations.euler_matrix(-np.pi / 2, np.pi / 2, np.pi / 2)
T_K0_A_S[:3, 3] = pos_K0_A

T_K0_B_U = trimesh.transformations.euler_matrix(np.pi / 2, 0, np.pi / 2)
T_K0_B_U[:3, 3] = pos_K0_B

T_K0_B_S = trimesh.transformations.euler_matrix(np.pi / 2, -np.pi / 2, np.pi / 2)
T_K0_B_S[:3, 3] = pos_K0_B

T_K0_C_U = T_K0_B_U.copy()
T_K0_C_U[:3, 3] = pos_K0_C

T_K0_C_S = T_K0_B_S.copy()
T_K0_C_S[:3, 3] = pos_K0_C

T_K0_D_U = T_K0_A_U.copy()
T_K0_D_U[:3, 3] = pos_K0_D

T_K0_D_S = T_K0_A_S.copy()
T_K0_D_S[:3, 3] = pos_K0_D

T_K1_A_U = trimesh.transformations.euler_matrix(-np.pi / 4, 0, 0) @ T_K0_A_U
T_K1_A_U[:3, 3] = pos_K1_A

T_K1_A_S = trimesh.transformations.euler_matrix(-np.pi / 4, 0, 0) @ T_K0_A_S
T_K1_A_S[:3, 3] = pos_K1_A

T_K1_B_U = T_K0_B_U.copy()
T_K1_B_S = T_K0_B_S.copy()

T_K1_C_U = trimesh.transformations.euler_matrix(-np.pi / 4, 0, 0) @ T_K0_C_U
T_K1_C_U[:3, 3] = pos_K1_C

T_K1_C_S = trimesh.transformations.euler_matrix(-np.pi / 4, 0, 0) @ T_K0_C_S
T_K1_C_S[:3, 3] = pos_K1_C

T_K1_D_U = T_K0_D_U.copy()

T_K1_D_S = T_K0_D_S.copy()


########################
### Torso components ###
########################

# Create shape
mesh_names = [
    "torso_dumbbell_K0",
    "sf_point",
    "sf_capsule_K1_F1",
    "sf_capsule_K1_F1",
    "sf_capsule_K1_F1",
]
mesh_list = [comp_dict[mesh_name].copy() for mesh_name in mesh_names]
T_list = [
    T_eye,
    T_K0_A_U,
    T_K0_B_U,
    T_K0_C_U,
    T_K0_D_U,
]
boolean_list = ["union"] * len(mesh_list)
boolean_list[-1] = "difference"
shape = Shape(
    mesh_list,
    T_list,
    boolean_list,
    label="test",
    description="test",
    save_dir=Path("./sample_shapes/"),
    mesh_fairing_distance=1,
)
shape.mesh.show(smooth=False)
