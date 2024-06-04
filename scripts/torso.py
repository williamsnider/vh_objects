# Torsos
from objects.shaft import Shaft
from scripts.stimulus_set_E import (
    CAPSULE_RADIUS,
    CAPSULE_LENGTH,
    FLATTENED_THICKNESS,
    NUM_CS,
    NUM_CP_PER_CROSS_SECTION,
    K1_THETA,
    CAPSULE_K1_LENGTH,
    comp_dict,
    T_point_z,
    plot_components,
)
import trimesh
import numpy as np

# Inputs
torso_length = 2 * CAPSULE_LENGTH
torso_radius = 1.25 * CAPSULE_RADIUS
K1_theta = np.pi / 4
football_r1 = torso_radius
football_r2 = 1.5 * torso_radius
football_r3 = 1 * torso_radius
cylinder_r1 = torso_radius
cylinder_r2 = torso_radius
cylinder_r3 = torso_radius
dumbbell_r1 = 1.5 * torso_radius
dumbbell_r2 = torso_radius
dumbbell_r3 = 1.5 * torso_radius

# football K0
torso_football_K0 = Shaft(
    torso_length,
    football_r1,
    football_r2,
    football_r3,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_football_K0.mesh.apply_transform(T_point_z)
comp_dict["torso_football_K0"] = torso_football_K0.mesh

# football K1
torso_football_K1 = Shaft(
    torso_length,
    football_r1,
    football_r2,
    football_r3,
    theta=K1_theta,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_football_K1.mesh.apply_transform(T_point_z)
comp_dict["torso_football_K1"] = torso_football_K1.mesh

# cylinder K0
torso_cylinder_K0 = Shaft(
    torso_length,
    cylinder_r1,
    cylinder_r2,
    cylinder_r3,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_cylinder_K0.mesh.apply_transform(T_point_z)
comp_dict["torso_cylinder_K0"] = torso_cylinder_K0.mesh

# cylinder K1
torso_cylinder_K1 = Shaft(
    torso_length,
    cylinder_r1,
    cylinder_r2,
    cylinder_r3,
    theta=K1_theta,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_cylinder_K1.mesh.apply_transform(T_point_z)
comp_dict["torso_cylinder_K1"] = torso_cylinder_K1.mesh

# dumbbell K0
torso_dumbbell_K0 = Shaft(
    torso_length,
    dumbbell_r1,
    dumbbell_r2,
    dumbbell_r3,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_dumbbell_K0.mesh.apply_transform(T_point_z)
comp_dict["torso_dumbbell_K0"] = torso_dumbbell_K0.mesh

# dumbbell K1
torso_dumbbell_K1 = Shaft(
    torso_length,
    dumbbell_r1,
    dumbbell_r2,
    dumbbell_r3,
    theta=K1_theta,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_dumbbell_K1.mesh.apply_transform(T_point_z)
comp_dict["torso_dumbbell_K1"] = torso_dumbbell_K1.mesh


########################
### Surface Features ###
########################

# inputs
sf_radius = CAPSULE_RADIUS
sf_radius_termination = 0.4  # Prevent sharp point
color_union = [0, 0, 255, 255]
color_difference = [255, 0, 0, 255]
NUM_CP_PER_BASE_SHEET = 50
NUM_CS_PER_SHEET = 11
uu = 50
vv = 50

# Point
sf_point = Shaft(
    sf_radius,
    sf_radius,
    0.5 * sf_radius,
    sf_radius_termination,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
sf_point.mesh.apply_transform(T_point_z)
# sf_point.mesh.visual.face_colors = [0, 0, 255, 255]
comp_dict["sf_point"] = sf_point.mesh

# Sphere
sf_sphere_union = trimesh.creation.icosphere(subdivisions=5, radius=sf_radius)
# sf_sphere_union.visual.face_colors = color_union
comp_dict["sf_sphere_union"] = sf_sphere_union

sf_sphere_difference = sf_sphere_union.copy()
sf_sphere_difference.visual.face_colors = color_difference
comp_dict["sf_sphere_difference"] = sf_sphere_difference


# # plot_components(comp_dict)
from scripts.stimulus_set_common import construct_sheet
from objects.utilities import make_surface, make_mesh

t = np.linspace(0, 2 * np.pi, NUM_CP_PER_BASE_SHEET, endpoint=False).reshape(-1, 1)
round_cs_cp = np.hstack([np.zeros(t.shape), np.cos(t), np.sin(t)])
base_sheet = round_cs_cp * (sf_radius - FLATTENED_THICKNESS / 4)
cp = construct_sheet(base_sheet, sheet_thickness=FLATTENED_THICKNESS, num_cs=NUM_CS_PER_SHEET)
surf = make_surface(cp)
sf_round_flat = make_mesh(surf, uu, vv)
comp_dict["sf_round_flat_union"] = sf_round_flat
comp_dict["sf_round_flat_union"].visual.face_colors = color_union
comp_dict["sf_round_flat_difference"] = sf_round_flat.copy()
comp_dict["sf_round_flat_difference"].visual.face_colors = color_difference

plot_components(comp_dict)
# # Sphere flattened and faired
# # sf_sphere_union.visual.face_colors = color_union
# sf_sphere_flat = sf_sphere_union.copy()
# sf_sphere_flat.vertices[:, 2] = np.clip(
#     sf_sphere_flat.vertices[:, 2], -FLATTENED_THICKNESS / 2, FLATTENED_THICKNESS / 2
# )
# idx_pos = sf_sphere_flat.vertices[:, 2] >= sf_sphere_flat.vertices[:, 2].max()
# idx_neg = sf_sphere_flat.vertices[:, 2] <= sf_sphere_flat.vertices[:, 2].min()
# idx_changed = idx_pos | idx_neg
# idx_unchanged = ~idx_changed
# idx_to_fair = np.arange(len(sf_sphere_flat.vertices), dtype="int")[idx_unchanged]

# # sf_sphere_flat.visual.face_colors = [125, 125, 125, 125]
# # sf_sphere_flat.visual.vertex_colors[idx_unchanged] = color_union

# # # Fair mesh
# from objects.utilities import fair_mesh, calc_mesh_principal_curvatures

# # faired = fair_mesh(sf_sphere_flat, idx_to_fair, 3)
# # # faired.show(smooth=False)

# # K1 and K2
# mesh_copy = sf_round_F1.copy()
# k1, k2 = calc_mesh_principal_curvatures(mesh_copy)

# # Show K1 as heatmap
# k1_norm = (k1 - k1.min()) / (k1.max() - k1.min())
# k2_norm = (k2 - k2.min()) / (k2.max() - k2.min())
# gauss = k1 * k2
# gauss_norm = (gauss - gauss.min()) / (gauss.max() - gauss.min())

# import matplotlib.pyplot as plt

# colormap = plt.cm.viridis
# rgba_colors = colormap(gauss_norm)

# mesh_copy.visual.vertex_colors = [255, 0, 0, 255]
# mesh_copy.visual.vertex_colors = rgba_colors
# mesh_copy.show(smooth=False)


# # # Color these vertices
# # unchanged_vertices_idx = [list(setA).index(v) for v in unchanged_vertices]
# # sf_sphere_flat.visual.vertex_colors[unchanged_vertices_idx] = color_difference

# # Get list of unchanged vertices
# sf_sphere_flat.show(smooth=False)
