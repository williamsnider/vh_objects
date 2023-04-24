# Linear segment
import copy
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.utilities import (
    approximate_arc,
    make_mesh,
    make_surface,
    calc_hemisphere_controlpoints,
    angle_between,
    fuse_meshes,
    calc_mesh_boolean_and_edges,
)
from objects.parameters import INTERFACE_PATH, INTERFACE_SHIFT
from objects.interface import load_interface
from scripts.sheets import construct_sheet, bend_sheet, make_base_cp
from scripts.hemi import calc_sphere_controlpoints
import trimesh
from scipy.spatial.transform.rotation import Rotation
from pathlib import Path
from objects.shaft import Shaft
from scripts.stimulus_set_params import (
    NUM_CP_PER_BACKBONE,
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
    POST_RADIUS,
    POST_OFFSET,
    SAVE_DIR,
    XYZ_OFFSET,
)

FAIRING_DISTANCE = 3
POST_Z_SHIFT = 0

uu = 50
vv = 50

######################################
### Base Components and Appendages ###
######################################


def slice_mesh(mesh, extent, T):
    mesh = mesh.copy()
    slicer = trimesh.primitives.Box(
        extents=np.array([extent, extent, extent]), transform=T
    )
    split_mesh, _ = calc_mesh_boolean_and_edges(mesh, slicer, "difference")

    return split_mesh


thin = Shaft(
    SEGMENT_LENGTH,
    VOLUMETRIC_RADII[0],
    VOLUMETRIC_RADII[0],
    VOLUMETRIC_RADII[0],
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)

volumetric = Shaft(
    SEGMENT_LENGTH,
    VOLUMETRIC_RADII[0],
    VOLUMETRIC_RADII[1],
    VOLUMETRIC_RADII[2],
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)

# Transformation matrix so that shafts are pointing towards +Z axis
T_point_z = np.eye(4)
T_point_z[:3, :3] = Rotation.from_euler("xyz", np.array([0, -np.pi / 2, 0])).as_matrix()

app1 = Shaft(
    APPENDAGE_LENGTH,
    1.0 * X_WIDTH,
    1.0 * X_WIDTH,
    1.4 * X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app1.apply_transform(T_point_z)

app2 = Shaft(
    APPENDAGE_LENGTH,
    1.0 * X_WIDTH,
    1.5 * X_WIDTH,
    0.1 * X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app2.apply_transform(T_point_z)

app3 = Shaft(
    APPENDAGE_LENGTH,
    1.0 * X_WIDTH,
    1.5 * X_WIDTH,
    1.0 * X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app3.apply_transform(T_point_z)

app4 = Shaft(
    APPENDAGE_LENGTH,
    1.0 * X_WIDTH,
    1.0 * X_WIDTH,
    0.1 * X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app4.apply_transform(T_point_z)

app_point = Shaft(
    2 * X_WIDTH,
    1.25 * X_WIDTH,
    0.6 * X_WIDTH,
    0.1 * X_WIDTH,
    0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app_point.apply_transform(T_point_z)
app_point_convex = app_point.mesh

# Flip across yz plane for concave
T = np.eye(4)
T[:3, :3] = Rotation.from_euler("xyz", np.array([np.pi, 0, 0])).as_matrix()
app_point_concave = app_point_convex.copy()
app_point_concave = app_point_concave.apply_transform(T)


# Slice convex to keep it from going through shape
extent = 100
T = np.eye(4)
T[2, 3] = -extent / 2 - 1 * X_WIDTH
app_point_convex = slice_mesh(
    app_point_convex,
    100,
    T,
)

app_round_concave = trimesh.primitives.creation.icosphere(3, radius=1.25 * X_WIDTH)
app_round_convex = app_round_concave.copy()


# For sheets, create a controlpoint grid, surface, and mesh from scratch (i.e. without the Shaft class).

# Rotation for K2
T_K2 = np.eye(4)
T_K2[:3, :3] = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()

# Backbone for bending sheets
b_cp = approximate_arc(np.pi / 2, APPENDAGE_LENGTH, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

b_cp = approximate_arc(np.pi / 2, X_WIDTH, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
backbone_x_width = Backbone(b_cp, reparameterize=True)

# Round sheet
t = np.linspace(0, 2 * np.pi, NUM_CP_PER_BASE_SHEET, endpoint=False).reshape(-1, 1)
round_cs_cp = np.hstack([np.zeros(t.shape), np.cos(t), np.sin(t)])
base_sheet = round_cs_cp * X_WIDTH
cp = construct_sheet(
    base_sheet, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET
)
surf = make_surface(cp)
sheet_round_K0 = make_mesh(surf, uu, vv)

# Bend round sheet
bent_cp = bend_sheet(cp, backbone_x_width, X_WIDTH)
surf = make_surface(bent_cp)
sheet_round_K1 = make_mesh(surf, uu, vv)
sheet_round_K2 = sheet_round_K1.copy()
sheet_round_K2.apply_transform(T_K2)


# Leaf sheet
num_edge_cp = 7
base_round_cp = 3
top_round_cp = 1
leaf_x = np.linspace(0, 1, 3) * APPENDAGE_LENGTH
leaf_y = LEAF_RADII
leaf_poly = np.polyfit(leaf_x, leaf_y, 2)
leaf_cp = make_base_cp(leaf_poly, leaf_x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = leaf_cp.mean(axis=0)
leaf_cp = leaf_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(leaf_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
surf = make_surface(cp)
sheet_leaf_K0 = make_mesh(surf, uu, vv)

# Bent leaf sheet
bent_cp = bend_sheet(cp, b_appendage_K1, leaf_x[2] - leaf_x[0])
surf = make_surface(bent_cp)
sheet_leaf_K1 = make_mesh(surf, uu, vv)
sheet_leaf_K2 = sheet_leaf_K1.copy()
sheet_leaf_K2.apply_transform(T_K2)


# Point sheet

# Calculate widths
point_x = np.linspace(0, APPENDAGE_LENGTH, 3)
point_y = POINT_RADII  # 3 radii determine polynomial form
point_poly = np.polyfit(point_x, point_y, 2)

# But sample along 4th position to ensure smooth transition into shape
xvals = np.linspace(-APPENDAGE_LENGTH * 2 / 3, APPENDAGE_LENGTH, NUM_CS)
widths = np.polyval(point_poly, xvals)

# Calculate z_levels
z_levels = np.linspace(-APPENDAGE_LENGTH * 2 / 3, APPENDAGE_LENGTH, NUM_CS)

# Assign controlpoints
cp = np.zeros((NUM_CS, 8, 3))
for i, width in enumerate(widths):

    if width == 0:
        inner = np.zeros((8, 2))
    else:
        inner = np.array(
            [
                [SHEET_THICKNESS / 2, 0],
                [SHEET_THICKNESS / 2, width - POINT_ROUNDOVER_OFFSET],
                [0, width],
                [-SHEET_THICKNESS / 2, width - POINT_ROUNDOVER_OFFSET],
                [-SHEET_THICKNESS / 2, 0],
                [-SHEET_THICKNESS / 2, -width + POINT_ROUNDOVER_OFFSET],
                [0, -width],
                [SHEET_THICKNESS / 2, -width + POINT_ROUNDOVER_OFFSET],
            ]
        )

    xyz = np.hstack([inner, z_levels[i] * np.ones((inner.shape[0], 1))])

    cp[i, :, :] = xyz

# Roundover edges

# side_y = POINT_RADII + POINT_ROUNDOVER_OFFSET
side_poly = point_poly  # np.polyfit(x, side_y, 2)
bot, _ = calc_hemisphere_controlpoints(
    cp[0],
    np.array([0, 0, 1]),
    cp[0].mean(axis=0),
    side_poly,
    -APPENDAGE_LENGTH * 2 / 3,
    morph_to_ellipse=True,
)
top, _ = calc_hemisphere_controlpoints(
    cp[-1],
    np.array([0, 0, 1]),
    cp[-1].mean(axis=0),
    side_poly,
    point_x[-1],
    morph_to_ellipse=True,
)
sheet_point_cp = np.vstack([bot, cp, top[-2::-1]])
surf = make_surface(sheet_point_cp)
sheet_point_K0 = make_mesh(surf, uu, vv)

bent_cp = bend_sheet(sheet_point_cp, b_appendage_K1, point_x[2] - point_x[0])
surf = make_surface(bent_cp)
sheet_point_K1 = make_mesh(
    surf, uu, vv
)  # TODO: Has artifact, fix after deciding on thickness/size
# sheet_point_bent.show(smooth=False)
sheet_point_K2 = sheet_point_K1.copy()
sheet_point_K2.apply_transform(T_K2)


mesh_dict = {
    "thin": thin.mesh,
    "volumetric": volumetric.mesh,
    "app1": app1.mesh,
    "app2": app2.mesh,
    "app3": app3.mesh,
    "app4": app4.mesh,
    "app_point_concave": app_point_concave,
    "app_point_convex": app_point_convex,
    "sheet_round_K0": sheet_round_K0,
    "sheet_round_K1": sheet_round_K1,
    "sheet_round_K2": sheet_round_K2,
    "app_round_concave": app_round_concave,
    "app_round_convex": app_round_convex,
    "sheet_leaf_K0": sheet_leaf_K0,
    "sheet_leaf_K1": sheet_leaf_K1,
    "sheet_leaf_K2": sheet_leaf_K2,
    "sheet_point_K0": sheet_point_K0,
    "sheet_point_K1": sheet_point_K1,
    "sheet_point_K2": sheet_point_K2,
}


# scene = trimesh.Scene()
# shift = np.array([0.0, 0.0, 0.0])
# for k, m in mesh_dict.items():

#     mesh = copy.deepcopy(m)

#     # Shift so bounds in lower left corner
#     mesh = mesh.apply_translation(np.array([0, mesh.extents[1] / 2, 0]))
#     mesh = mesh.apply_translation(shift)

#     scene.add_geometry(mesh)

#     extents = np.copy(mesh.extents)
#     extents[0] = 0
#     extents[2] = 0
#     shift += extents * 1.1
# scene.show()

#######################
### Transformations ###
#######################


def find_line_mesh_intersection(mesh, vec, origin):

    # Find points that the vector is pointing to (correct side of mesh)
    angles = angle_between(mesh.vertices - origin, vec)
    pts = mesh.vertices[angles < np.pi / 2]

    # Calc dist from these points to the vector
    dists = np.linalg.norm(np.cross(pts - origin, vec), axis=1) / np.linalg.norm(vec)

    # Choose smallest dist as point
    idx = dists.argmin()
    xyz = pts[idx]

    return xyz


# For these transformations, consider that the base shape is centered at the origin, with its "sheet" dimension in the yz plane, and it's thickness dimension in the x plane. It also bends in the +X direction. Based on this, which transformations result in it being on the shape in the Up/Down position and pointing forward/left/back/right.

vec_axial_component = np.array([1, 0, 0])
x_th = 0  # -np.pi / 4
y_th = 0  # np.pi / 4  # np.pi / 9
vec_to_J1 = np.array([0, np.sin(x_th), np.cos(x_th)])
vec_to_J1_orth = np.cross(vec_to_J1, vec_axial_component)
vec_to_J2 = np.array([0, np.sin(x_th), np.cos(x_th)])
vec_to_J2_orth = np.cross(vec_to_J2, vec_axial_component)

J1_pos = volumetric.backbone.r(0.5)
J2_pos = np.array([SEGMENT_LENGTH - X_WIDTH, 0, 0])
CO_xyz = np.array([SEGMENT_LENGTH, 0, 0])

# Volumetric
J1_volu_xyz_U = (
    find_line_mesh_intersection(volumetric.mesh, vec_to_J1, J1_pos)
    + XYZ_OFFSET * vec_to_J1
)
J1_volu_xyz_D = (
    find_line_mesh_intersection(volumetric.mesh, -vec_to_J1, J1_pos)
    + -XYZ_OFFSET * vec_to_J1
)
J2_volu_xyz_U = (
    find_line_mesh_intersection(volumetric.mesh, vec_to_J2, J2_pos)
    + XYZ_OFFSET * vec_to_J2
)
J2_volu_xyz_D = (
    find_line_mesh_intersection(volumetric.mesh, -vec_to_J2, J2_pos)
    + -XYZ_OFFSET * vec_to_J2
)

# Thin
J1_thin_xyz_U = (
    find_line_mesh_intersection(thin.mesh, vec_to_J1, J1_pos) + XYZ_OFFSET * vec_to_J1
)
J1_thin_xyz_D = (
    find_line_mesh_intersection(thin.mesh, -vec_to_J1, J1_pos) + -XYZ_OFFSET * vec_to_J1
)
J2_thin_xyz_U = (
    find_line_mesh_intersection(thin.mesh, vec_to_J2, J2_pos) + XYZ_OFFSET * vec_to_J2
)
J2_thin_xyz_D = (
    find_line_mesh_intersection(thin.mesh, -vec_to_J2, J2_pos) + -XYZ_OFFSET * vec_to_J2
)

R_F_U = Rotation.from_euler("zyx", np.array([0 * np.pi / 2, y_th, -x_th])).as_matrix()
R_L_U = Rotation.from_euler("zyx", np.array([1 * np.pi / 2, y_th, -x_th])).as_matrix()
R_B_U = Rotation.from_euler("zyx", np.array([2 * np.pi / 2, y_th, -x_th])).as_matrix()
R_R_U = Rotation.from_euler("zyx", np.array([3 * np.pi / 2, y_th, -x_th])).as_matrix()

R_F_D = Rotation.from_euler(
    "zyx", np.array([0 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_L_D = Rotation.from_euler(
    "zyx", np.array([3 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_B_D = Rotation.from_euler(
    "zyx", np.array([2 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_R_D = Rotation.from_euler(
    "zyx", np.array([1 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()

# Collinear rotations
R_U = Rotation.from_euler(
    "zyx", np.array([np.pi, np.pi / 2, -x_th + 0 * np.pi / 2])
).as_matrix()
R_L = Rotation.from_euler(
    "zyx", np.array([np.pi, np.pi / 2, -x_th + 3 * np.pi / 2])
).as_matrix()
R_D = Rotation.from_euler(
    "zyx", np.array([np.pi, np.pi / 2, -x_th + 2 * np.pi / 2])
).as_matrix()
R_R = Rotation.from_euler(
    "zyx", np.array([np.pi, np.pi / 2, -x_th + 1 * np.pi / 2])
).as_matrix()


def calc_T(R, xyz):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


# J1, direction of curvature (forward, left, right, back), up/down
T_volu_J1_B_U = calc_T(R_B_U, J1_volu_xyz_U)
T_volu_J1_B_D = calc_T(R_B_D, J1_volu_xyz_D)
T_volu_J1_L_U = calc_T(R_L_U, J1_volu_xyz_U)
T_volu_J1_L_D = calc_T(R_L_D, J1_volu_xyz_D)
T_volu_J1_F_U = calc_T(R_F_U, J1_volu_xyz_U)
T_volu_J1_F_D = calc_T(R_F_D, J1_volu_xyz_D)
T_volu_J1_R_U = calc_T(R_R_U, J1_volu_xyz_U)
T_volu_J1_R_D = calc_T(R_R_D, J1_volu_xyz_D)

# J2
T_volu_J2_B_U = calc_T(R_B_U, J2_volu_xyz_U)
T_volu_J2_B_D = calc_T(R_B_D, J2_volu_xyz_D)
T_volu_J2_L_U = calc_T(R_L_U, J2_volu_xyz_U)
T_volu_J2_L_D = calc_T(R_L_D, J2_volu_xyz_D)
T_volu_J2_F_U = calc_T(R_F_U, J2_volu_xyz_U)
T_volu_J2_F_D = calc_T(R_F_D, J2_volu_xyz_D)
T_volu_J2_R_U = calc_T(R_R_U, J2_volu_xyz_U)
T_volu_J2_R_D = calc_T(R_R_D, J2_volu_xyz_D)

# J1, direction of curvature (forward, left, right, back), up/down
T_thin_J1_B_U = calc_T(R_B_U, J1_thin_xyz_U)
T_thin_J1_B_D = calc_T(R_B_D, J1_thin_xyz_D)
T_thin_J1_L_U = calc_T(R_L_U, J1_thin_xyz_U)
T_thin_J1_L_D = calc_T(R_L_D, J1_thin_xyz_D)
T_thin_J1_F_U = calc_T(R_F_U, J1_thin_xyz_U)
T_thin_J1_F_D = calc_T(R_F_D, J1_thin_xyz_D)
T_thin_J1_R_U = calc_T(R_R_U, J1_thin_xyz_U)
T_thin_J1_R_D = calc_T(R_R_D, J1_thin_xyz_D)

# J2
T_thin_J2_B_U = calc_T(R_B_U, J2_thin_xyz_U)
T_thin_J2_B_D = calc_T(R_B_D, J2_thin_xyz_D)
T_thin_J2_L_U = calc_T(R_L_U, J2_thin_xyz_U)
T_thin_J2_L_D = calc_T(R_L_D, J2_thin_xyz_D)
T_thin_J2_F_U = calc_T(R_F_U, J2_thin_xyz_U)
T_thin_J2_F_D = calc_T(R_F_D, J2_thin_xyz_D)
T_thin_J2_R_U = calc_T(R_R_U, J2_thin_xyz_U)
T_thin_J2_R_D = calc_T(R_R_D, J2_thin_xyz_D)


# Collinear
T_CO_U = calc_T(R_U, CO_xyz)
T_CO_L = calc_T(R_L, CO_xyz)
T_CO_D = calc_T(R_D, CO_xyz)
T_CO_R = calc_T(R_R, CO_xyz)

combs = []


T_dict = {
    "T_eye": np.eye(4),
    "T_volu_J1_B_U": T_volu_J1_B_U,
    "T_volu_J1_B_D": T_volu_J1_B_D,
    "T_volu_J1_L_U": T_volu_J1_L_U,
    "T_volu_J1_L_D": T_volu_J1_L_D,
    "T_volu_J1_F_U": T_volu_J1_F_U,
    "T_volu_J1_F_D": T_volu_J1_F_D,
    "T_volu_J1_R_U": T_volu_J1_R_U,
    "T_volu_J1_R_D": T_volu_J1_R_D,
    "T_volu_J2_B_U": T_volu_J2_B_U,
    "T_volu_J2_B_D": T_volu_J2_B_D,
    "T_volu_J2_L_U": T_volu_J2_L_U,
    "T_volu_J2_L_D": T_volu_J2_L_D,
    "T_volu_J2_F_U": T_volu_J2_F_U,
    "T_volu_J2_F_D": T_volu_J2_F_D,
    "T_volu_J2_R_U": T_volu_J2_R_U,
    "T_volu_J2_R_D": T_volu_J2_R_D,
    "T_thin_J1_B_U": T_thin_J1_B_U,  #
    "T_thin_J1_B_D": T_thin_J1_B_D,
    "T_thin_J1_L_U": T_thin_J1_L_U,
    "T_thin_J1_L_D": T_thin_J1_L_D,
    "T_thin_J1_F_U": T_thin_J1_F_U,
    "T_thin_J1_F_D": T_thin_J1_F_D,
    "T_thin_J1_R_U": T_thin_J1_R_U,
    "T_thin_J1_R_D": T_thin_J1_R_D,
    "T_thin_J2_B_U": T_thin_J2_B_U,
    "T_thin_J2_B_D": T_thin_J2_B_D,
    "T_thin_J2_L_U": T_thin_J2_L_U,
    "T_thin_J2_L_D": T_thin_J2_L_D,
    "T_thin_J2_F_U": T_thin_J2_F_U,
    "T_thin_J2_F_D": T_thin_J2_F_D,
    "T_thin_J2_R_U": T_thin_J2_R_U,
    "T_thin_J2_R_D": T_thin_J2_R_D,
    "T_CO_U": T_CO_U,
    "T_CO_L": T_CO_L,
    "T_CO_D": T_CO_D,
    "T_CO_R": T_CO_R,
}
#############################################################
### Combinations of Shapes (Volumetric/Thin + Appendages) ###
#############################################################
combs = []
count = 200


def build_shape(inputs):
    mesh_list = [mesh_dict[n] for n in inputs[0]]
    T_list = [T_dict[n] for n in inputs[1]]
    boolean_list = inputs[2]
    label = str(inputs[3])

    description = inputs[4]
    save_dir = inputs[5]
    T_final = T_dict[inputs[6]]
    fairing_distance = inputs[7]
    post_z_shift = inputs[8]

    s = Shape(
        mesh_list,
        T_list,
        boolean_list,
        label,
        description,
        save_dir,
        T_final,
        fairing_distance,
        post_z_shift,
    )
    # s.mesh_with_interface.show()


###########################
### Case: Thin Backbone ###
###########################

# # J1 and J2 and Collinear
# for J_app in ["app1", "app2", "app3", "app4"]:
#     for CO_app in ["app1", "app2", "app3", "app4"]:

#         # Iterate through J1 only, J2 only, J1+CO, J2+CO, J1+J2+CO
#         for J1_J2_CO in [
#             [True, False, False],
#             [False, True, False],
#             [True, True, False],
#             [True, False, True],
#             [False, True, True],
#             [True, True, True],
#         ]:

#             withJ1, withJ2, withCO = J1_J2_CO

#             for hox in [False, True]:

#                 # Transformation matrices
#                 T_J1 = "T_thin_J1_F_U"
#                 T_J2 = "T_thin_J2_F_U"
#                 T_CO = "T_CO_U"
#                 T_J1_hox = T_J1[:-1] + "D"
#                 T_J2_hox = T_J2[:-1] + "D"

#                 # Construct mesh and T_list
#                 mesh_list = ["thin"]
#                 T_list = ["T_eye"]

#                 # Add CO
#                 if withCO == True:
#                     mesh_list.append(CO_app)
#                     T_list.append(T_CO)

#                 # Add J1
#                 if withJ1 == True:
#                     mesh_list.append(J_app)
#                     T_list.append(T_J1)

#                     if hox == True:
#                         mesh_list.append(J_app)
#                         T_list.append(T_J1_hox)

#                 # Add J2
#                 if withJ2 == True:
#                     mesh_list.append(J_app)
#                     T_list.append(T_J2)

#                     if hox == True:
#                         mesh_list.append(J_app)
#                         T_list.append(T_J2_hox)

#                 # Prevent duplicates for shapes by not looping for CO if not present
#                 if withCO == False and CO_app != "app1":
#                     continue

#                 # Assign combination of inputs
#                 comb = [
#                     mesh_list,
#                     T_list,
#                     str(count),
#                     "",
#                     SAVE_DIR,
#                     "T_eye",
#                     FAIRING_DISTANCE,
#                     POST_Z_SHIFT,
#                 ]

#                 # Do not add duplicates
#                 if comb not in combs:
#                     combs.append(comb)
#                     count += 1

#################################
### Case: Volumetric Backbone ###
#################################

# J1 and J2 and Collinear
for J_app in [
    "app1",
    "app2",
    "app3",
    "app4",
    "app_point_convex",
    "app_point_concave",
    "app_round_convex",
    "app_round_concave",
]:
    for CO_app in ["app1", "app2", "app3", "app4"]:

        # Iterate through J1 only, J2 only, J1+CO, J2+CO, J1+J2+CO
        for J1_J2_CO in [
            [True, False, False],
            [False, True, False],
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, True, True],
        ]:

            withJ1, withJ2, withCO = J1_J2_CO

            # Appendages with base=2X cannot be at J2, so skip
            if withJ2 == True and J_app in [
                "app_point_convex",
                "app_point_concave",
                "app_round_convex",
                "app_round_concave",
            ]:
                continue

            for hox in [False, True]:

                # Transformation matrices
                T_J1 = "T_volu_J1_F_U"
                T_J2 = "T_volu_J2_F_U"
                T_CO = "T_CO_U"
                T_J1_hox = T_J1[:-1] + "D"
                T_J2_hox = T_J2[:-1] + "D"

                # Construct mesh and T_list
                mesh_list = ["volumetric"]
                T_list = ["T_eye"]
                boolean_list = [None]

                # Add CO
                if withCO == True:
                    mesh_list.append(CO_app)
                    T_list.append(T_CO)
                    boolean_list.append("union")

                # Add J1
                if withJ1 == True:
                    mesh_list.append(J_app)
                    T_list.append(T_J1)

                    if "concave" in J_app:
                        boolean_list.append("difference")
                    else:
                        boolean_list.append("union")

                    if hox == True:
                        mesh_list.append(J_app)
                        T_list.append(T_J1_hox)

                        if "concave" in J_app:
                            boolean_list.append("difference")
                        else:
                            boolean_list.append("union")

                # Add J2
                if withJ2 == True:
                    mesh_list.append(J_app)
                    T_list.append(T_J2)
                    boolean_list.append("union")

                    if hox == True:
                        mesh_list.append(J_app)
                        T_list.append(T_J2_hox)
                        boolean_list.append("union")

                # Prevent duplicates for shapes by not looping for CO if not present
                if withCO == False and CO_app != "app1":
                    continue

                # Assign combination of inputs
                comb = [
                    mesh_list,
                    T_list,
                    boolean_list,
                    str(count),
                    "",
                    SAVE_DIR,
                    "T_eye",
                    FAIRING_DISTANCE,
                    POST_Z_SHIFT,
                ]

                # Do not add duplicates
                if comb not in combs:
                    combs.append(comb)
                    count += 1

# # J1
# for appendage_type in ["sheet_round", "sheet_leaf", "sheet_point"]:
#     for curvature_profile in ["K0", "K1", "K2"]:

#         appendage = eval(appendage_type + "_" + curvature_profile).copy()

#         for T1 in ["T_J1_F_U", "T_J1_L_U"]:
#             for hox in [False, True]:

#                 mesh_list = ["volumetric.mesh"]

#                 mesh = appendage.copy()
#                 mesh.apply_transform(eval(T1))
#                 mesh_list.append(mesh)

#                 if hox == True:
#                     T2 = T1[:-1] + "D"
#                     hox_mesh = appendage.copy()
#                     hox_mesh.apply_transform(eval(T2))
#                     mesh_list.append(hox_mesh)

#                 # Fuse meshes
#                 meshA = mesh_list[0]
#                 for meshB in mesh_list[1:]:
#                     meshA = fuse_meshes(meshA, meshB, 2, "union")
#                 meshA.show(smooth=False)
#                 # scene = trimesh.Scene()
#                 # scene.add_geometry(mesh_list)
#                 # scene.show()

########################
### Construct Shapes ###
########################


# for comb in combs:
#     build_shape(comb)

if __name__ == "__main__":

    with Pool() as pool:
        mapped_values = list(
            tqdm(pool.imap_unordered(build_shape, combs), total=len(combs))
        )


# def build_shape(inputs):

#     shape_number, mesh_names, T_names = inputs

#     # Get mesh and T values
#     mesh_list = []
#     T_list = []
#     for i in range(len(mesh_names)):
#         mesh_list.append(mesh_dict[mesh_names[i]])
#         T_list.append(T_dict[T_names[i]])

#     # # Additional verts to fair
#     # add_vert_indices = []
#     # for i, mesh in enumerate(mesh_list):

#     #     if i == 0:
#     #         add_vert_indices.append(np.array([]))
#     #     else:
#     #         add_vert_indices.append(mesh.vertices[:, 2] < 0)

#     # Transform meshes
#     new_mesh_list = []
#     for i, mesh in enumerate(mesh_list):
#         mesh = mesh.copy()
#         new_mesh_list.append(mesh.apply_transform(T_list[i]))

#     # Split mesh by yz-plane to prevent going through shape
#     split_mesh_list = []
#     for i, mesh in enumerate(new_mesh_list):

#         mesh = mesh.copy()
#         T_name = T_names[i]

#         # Make slicer mesh
#         T = np.eye(4)
#         extent = 100
#         if T_name in [
#             "T_J1_B_U",
#             "T_J1_L_U",
#             "T_J1_F_U",
#             "T_J1_R_U",
#             "T_J2_B_U",
#             "T_J2_L_U",
#             "T_J2_F_U",
#             "T_J2_R_U",
#         ]:
#             T[2, 3] = -extent / 2
#             split_mesh = slice_mesh(mesh, extent, T)

#         elif T_name in [
#             "T_J1_B_D",
#             "T_J1_L_D",
#             "T_J1_F_D",
#             "T_J1_R_D",
#             "T_J2_B_D",
#             "T_J2_L_D",
#             "T_J2_F_D",
#             "T_J2_R_D",
#         ]:
#             T[2, 3] = extent / 2
#             split_mesh = slice_mesh(mesh, extent, T)

#         elif T_name in [
#             "T_eye",
#             "T_CO_U",
#             "T_CO_L",
#             "T_CO_D",
#             "T_CO_R",
#         ]:
#             split_mesh = mesh
#         else:
#             raise NotImplementedError

#         split_mesh_list.append(split_mesh)

#     # slicer = trimesh.primitives.Box(
#     #     extents=np.array([extent, extent, extent]), transform=T
#     # )
#     # scene = trimesh.Scene()
#     # scene.add_geometry([mesh, slicer])
#     # scene.show()

#     # Fuse meshes
#     meshA = split_mesh_list[0]
#     i = 1
#     for meshB in split_mesh_list[1:]:
#         meshA = fuse_meshes(meshA, meshB, fairing_distance, "union")

#     # Attach post
#     meshA = fuse_meshes(meshA, post_ac.mesh, 2, "union")

#     # Attach interface
#     label = str(shape_number).zfill(4)
#     # label_split = [label[i] for i in range(len(label))]
#     # label_formatted = "_".join(label_split)  # Underscores denote new lines
#     interface = load_interface(INTERFACE_PATH, label)
#     mesh_with_interface = fuse_meshes(meshA, interface, 0, "union")

#     # Construct save_dir
#     if SAVE_DIR.is_dir() is False:
#         SAVE_DIR.mkdir(parents=True)

#     # Export
#     filename = Path(SAVE_DIR, label).with_suffix(".stl")
#     mesh_with_interface.export(filename)

#     mesh_with_interface.show(smooth=False)
#     import trimesh.scene

#     scene = trimesh.Scene()
#     scene.add_geometry(mesh_with_interface)
#     scene.lights = trimesh.scene.lighting.autolight(scene)
#     return mesh_with_interface


count = 15

# Sheet point
inputs = (
    count,
    [
        "volumetric",
        "app1",
        "app1",
        "app1",
        "app1",
        "app1",
    ],
    [
        "T_eye",
        "T_CO_U",
        "T_J1_F_U",
        "T_J2_F_U",
        "T_J1_F_D",
        "T_J2_F_D",
    ],
)


build_shape(inputs)
count += 1


# SPHERE_RADIUS = 15

# Derived parameters
# cs_radii = np.arange(3) * X_WIDTH
# appendage_length = SEGMENT_LENGTH / 2

##############################
### Cross Section Profiles ###
##############################

# Circular cross sections
t = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
round_cp = np.hstack([np.cos(t), np.sin(t)])

#################
### Backbones ###
#################

# Linear 0.5 segment
b_lin05_cp = np.hstack(
    [
        np.linspace(0, SEGMENT_LENGTH / 2, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_lin05 = Backbone(b_lin05_cp, reparameterize=True)

# Linear 1 segment
b_lin1_cp = np.hstack(
    [
        np.linspace(0, SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_lin1 = Backbone(b_lin1_cp, reparameterize=True)

# Linear 2 segment (L1+L2)
b_lin2_cp = np.hstack(
    [
        np.linspace(0, 2 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_lin2 = Backbone(b_lin2_cp, reparameterize=True)

# Curved 0.5 segment
b_cur05_cp = approximate_arc(np.pi / 2, SEGMENT_LENGTH / 2, NUM_CP_PER_BACKBONE)
b_cur05 = Backbone(b_cur05_cp, reparameterize=True)

# Curved 1 segment
b_cur1_cp = approximate_arc(np.pi / 2, SEGMENT_LENGTH, NUM_CP_PER_BACKBONE)
b_cur1 = Backbone(b_cur1_cp, reparameterize=True)


########################
### Axial Components ###
########################

### Common variables ###
pos = np.linspace(0, 1, NUM_CS)  # Position from (0,1)
pos_seg1 = pos * SEGMENT_LENGTH  # Position from (0, SEGMENT_LENGTH)
pos_seg2 = pos_seg1 * 2  # Position from (0, 2*SEGMENT_LENGTH)
pos_seg05 = pos * SEGMENT_LENGTH / 2  # Position from (0, SEGMENT_LENGTH/2)

### Volumetric ("football") ###

# Fit quadratic polynomial to determine scaling of cross sections
volumetric_x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH * 2
volumetric_y = VOLUMETRIC_RADII
volumetric_poly = np.polyfit(volumetric_x, volumetric_y, 2)
volumetric_scale = np.polyval(volumetric_poly, pos_seg2)
volumetric_cs = [
    CrossSection(volumetric_scale[i] * round_cp, pos[i]) for i in range(NUM_CS)
]

# Construct axial component
volumetric = AxialComponent(
    b_lin2,
    volumetric_cs,
    hemispherical_ends=True,
    hemispherical_polynomial=volumetric_poly,
    hemisphere_x=[0, 2 * SEGMENT_LENGTH],
)
##############
### Sheets ###
##############
T_K2 = np.eye(4)
T_K2[:3, :3] = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()

# Sphere

# Straight backbone for appendages
b_cp = np.hstack(
    [
        np.linspace(0, APPENDAGE_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_appendage_K0 = Backbone(b_cp, reparameterize=True)

# Backbone for bending sheets
b_cp = approximate_arc(np.pi / 2, APPENDAGE_LENGTH, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

b_cp = approximate_arc(np.pi / 2, X_WIDTH, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
backbone_x_width = Backbone(b_cp, reparameterize=True)

# Round sheet
t = np.linspace(0, 2 * np.pi, NUM_CP_PER_BASE_SHEET, endpoint=False).reshape(-1, 1)
round_cs_cp = np.hstack([np.zeros(t.shape), np.cos(t), np.sin(t)])
base_sheet = round_cs_cp * X_WIDTH
cp = construct_sheet(
    base_sheet, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET
)
surf = make_surface(cp)
sheet_round_K0 = make_mesh(surf, uu, vv)

# Bend round sheet
bent_cp = bend_sheet(cp, backbone_x_width, X_WIDTH)
surf = make_surface(bent_cp)
sheet_round_K1 = make_mesh(surf, uu, vv)
sheet_round_K2 = sheet_round_K1.copy()
sheet_round_K2.apply_transform(T_K2)


# Leaf sheet
num_edge_cp = 7
base_round_cp = 3
top_round_cp = 1
leaf_x = np.linspace(0, 1, 3) * APPENDAGE_LENGTH
leaf_y = LEAF_RADII
leaf_poly = np.polyfit(leaf_x, leaf_y, 2)
leaf_cp = make_base_cp(leaf_poly, leaf_x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = leaf_cp.mean(axis=0)
leaf_cp = leaf_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(leaf_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
surf = make_surface(cp)
sheet_leaf_K0 = make_mesh(surf, uu, vv)

# Bent leaf sheet
bent_cp = bend_sheet(cp, b_appendage_K1, leaf_x[2] - leaf_x[0])
surf = make_surface(bent_cp)
sheet_leaf_K1 = make_mesh(surf, uu, vv)
sheet_leaf_K2 = sheet_leaf_K1.copy()
sheet_leaf_K2.apply_transform(T_K2)


# Point sheet

# Calculate widths
point_x = np.linspace(0, APPENDAGE_LENGTH, 3)
point_y = POINT_RADII  # 3 radii determine polynomial form
point_poly = np.polyfit(point_x, point_y, 2)

# But sample along 4th position to ensure smooth transition into shape
xvals = np.linspace(-APPENDAGE_LENGTH * 2 / 3, APPENDAGE_LENGTH, NUM_CS)
widths = np.polyval(point_poly, xvals)

# Calculate z_levels
z_levels = np.linspace(-APPENDAGE_LENGTH * 2 / 3, APPENDAGE_LENGTH, NUM_CS)

# Assign controlpoints
cp = np.zeros((NUM_CS, 8, 3))
for i, width in enumerate(widths):

    if width == 0:
        inner = np.zeros((8, 2))
    else:
        inner = np.array(
            [
                [SHEET_THICKNESS / 2, 0],
                [SHEET_THICKNESS / 2, width - POINT_ROUNDOVER_OFFSET],
                [0, width],
                [-SHEET_THICKNESS / 2, width - POINT_ROUNDOVER_OFFSET],
                [-SHEET_THICKNESS / 2, 0],
                [-SHEET_THICKNESS / 2, -width + POINT_ROUNDOVER_OFFSET],
                [0, -width],
                [SHEET_THICKNESS / 2, -width + POINT_ROUNDOVER_OFFSET],
            ]
        )

    xyz = np.hstack([inner, z_levels[i] * np.ones((inner.shape[0], 1))])

    cp[i, :, :] = xyz

# Roundover edges

# side_y = POINT_RADII + POINT_ROUNDOVER_OFFSET
side_poly = point_poly  # np.polyfit(x, side_y, 2)
bot, _ = calc_hemisphere_controlpoints(
    cp[0],
    np.array([0, 0, 1]),
    cp[0].mean(axis=0),
    side_poly,
    -APPENDAGE_LENGTH * 2 / 3,
    morph_to_ellipse=True,
)
top, _ = calc_hemisphere_controlpoints(
    cp[-1],
    np.array([0, 0, 1]),
    cp[-1].mean(axis=0),
    side_poly,
    point_x[-1],
    morph_to_ellipse=True,
)
sheet_point_cp = np.vstack([bot, cp, top[-2::-1]])
surf = make_surface(sheet_point_cp)
sheet_point_K0 = make_mesh(surf, uu, vv)

bent_cp = bend_sheet(sheet_point_cp, b_appendage_K1, point_x[2] - point_x[0])
surf = make_surface(bent_cp)
sheet_point_K1 = make_mesh(
    surf, uu, vv
)  # TODO: Has artifact, fix after deciding on thickness/size
# sheet_point_bent.show(smooth=False)
sheet_point_K2 = sheet_point_K1.copy()
sheet_point_K2.apply_transform(T_K2)
# Sphere

num_cp = 11
t = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
round_cp = np.hstack([np.cos(t), np.sin(t)])
base_cp = np.hstack([np.zeros((round_cp.shape[0], 1)), round_cp])
cp = calc_sphere_controlpoints(
    base_cp,
    num_cp,
    np.array([1, 0, 0]),
    np.array([0, 0, 0]),
    x=0,
)
cp *= X_WIDTH

# surf = make_surface(cp)
# sphere = make_mesh(surf, uu, vv)
# max_bbox = trimesh.creation.box([100, 45, 45])
# inch_sphere = trimesh.creation.icosphere(3, 25.4 / 2)

##################
### Non-Sheets ###
##################
def transform_ac(ac, T, offset):

    # Align backbone to origin at position t
    if offset != 0.0:
        T_shift_to_origin = np.eye(4)
        T_shift_to_origin[0, 3] = -ac.mesh.bounds[0, 0] - offset
        ac.mesh = ac.mesh.apply_transform(T_shift_to_origin)
        ac.controlpoints = (
            np.dstack([ac.controlpoints, np.ones([*ac.controlpoints.shape[:-1], 1])])
            @ T_shift_to_origin.T
        )[:, :, :3]

    # Rotate about y-axis
    ac.mesh = ac.mesh.apply_transform(T)
    ac.controlpoints = (
        np.dstack([ac.controlpoints, np.ones([*ac.controlpoints.shape[:-1], 1])]) @ T.T
    )[:, :, :3]
    return ac


T_ac = np.eye(4)
T_ac[:3, :3] = Rotation.from_euler("zyx", np.array([0, -np.pi / 2, 0])).as_matrix()

pos_app = np.linspace(0, APPENDAGE_LENGTH, NUM_CS)

# Point
scale = np.polyval(point_poly, pos_app)
cs_list = [
    CrossSection(controlpoints=round_cp * scale[i], position=pos[i])
    for i in range(NUM_CS)
]
ac_point = AxialComponent(
    b_appendage_K0,
    cs_list,
    smooth_with_post=False,
    hemispherical_ends=True,
    hemispherical_polynomial=point_poly,
    hemisphere_x=[pos_app[0], pos_app[-1]],
)

ac_point = transform_ac(ac_point, T_ac, 0.0)


# Leaf
scale = np.polyval(leaf_poly, pos_app)
cs_list = [
    CrossSection(controlpoints=round_cp * scale[i], position=pos[i])
    for i in range(NUM_CS)
]
ac_leaf = AxialComponent(
    b_appendage_K0,
    cs_list,
    smooth_with_post=False,
    hemispherical_ends=True,
    hemispherical_polynomial=leaf_poly,
    hemisphere_x=[pos_app[0], pos_app[-1]],
)
ac_leaf = transform_ac(ac_leaf, T_ac, 0.0)


# ac_leaf.mesh = ac_leaf.mesh.apply_transform(T_ac)
# ac_leaf.controlpoints = (
#     np.dstack([ac_leaf.controlpoints, np.ones([*ac_leaf.controlpoints.shape[:-1], 1])])
#     @ T_ac.T
# )[:, :, :3]

from scripts.optimize_shaft import optimize_backbone_length, make_ac


# appendage_extra_factor = 1
# appendage_extra = APPENDAGE_LENGTH * appendage_extra_factor
# offset = appendage_extra * (1 - 1 / appendage_extra_factor)
offset = 0.0

from scripts.shaft import make_shaft

extra_fac = 0.0
shaft1 = make_shaft(
    APPENDAGE_LENGTH * (1 + extra_fac),
    X_WIDTH,
    1 * X_WIDTH,
    1.3 * X_WIDTH,
    "one_hemi",
    22,
    NUM_CP_PER_CROSS_SECTION,
)

cp = shaft1.cp
# Convert to cylinder at thinnest point
yz = cp[:, :, 1:]
r = np.linalg.norm(yz, axis=2).mean(axis=1)

# Min value in middle third
start_idx = cp.shape[0] // 3
idx = np.argmin(r[start_idx:-start_idx]) + start_idx
cp[2:idx, :, 1:] = cp[idx, :, 1:]

# Elongate
zvals = cp[:idx, 0, 0]
zshift = zvals - zvals[-1]
zscale = zshift * 2
znew = zscale + zvals[-1]
cp[:idx, :, 0] = znew.reshape(-1, 1)
# plot_arr(cp)
surf = make_surface(cp)
mesh = make_mesh(surf, uu, vv)

ac1 = mesh
T = T_ac
T[2, 3] = -APPENDAGE_LENGTH * extra_fac
ac1.apply_transform(T_ac)

# ac1


# # ac1
# # y = np.array([1.0 * X_WIDTH, 1.0 * X_WIDTH, 1.25 * X_WIDTH])
# # optimal_backbone_length = optimize_backbone_length(APPENDAGE_LENGTH, *y, "far")
# # ac1 = make_ac(optimal_backbone_length, *y)
# ac1 = transform_ac(ac1, T_ac, offset)

# # Expand bottom
# ac1.controlpoints[:10, :, 2] *= 3
# ac1.controlpoints[1:10, :, :2] = ac1.controlpoints[10, :, :2]
# ac1.surface = make_surface(ac1.controlpoints)
# ac1.mesh = make_mesh(ac1.surface, 250, 250)
# # ac1.mesh.show()

# # ac2
# y = np.array([1.0 * X_WIDTH, 1.5 * X_WIDTH, 0.1 * X_WIDTH])
# optimal_backbone_length = optimize_backbone_length(APPENDAGE_LENGTH, *y, "far")
# ac2 = make_ac(optimal_backbone_length, *y)
# ac2 = transform_ac(ac2, T_ac, offset)

# # ac3
# y = np.array([1.0 * X_WIDTH, 1.5 * X_WIDTH, 1.0 * X_WIDTH])
# optimal_backbone_length = optimize_backbone_length(APPENDAGE_LENGTH, *y, "far")
# ac3 = make_ac(optimal_backbone_length, *y)
# ac3 = transform_ac(ac3, T_ac, offset)

# # ac4
# y = np.array([1.0 * X_WIDTH, 1.0 * X_WIDTH, 0.1 * X_WIDTH])
# optimal_backbone_length = optimize_backbone_length(APPENDAGE_LENGTH, *y, "far")
# ac4 = make_ac(optimal_backbone_length, *y)
# ac4 = transform_ac(ac4, T_ac, offset)


# # Compare ac_point and sheet_point controlpoints
# import matplotlib.pyplot as plt

# ax = plt.figure().add_subplot(projection="3d")

# arr = sheet_point_cp
# xs = np.array([])
# ys = np.array([])
# zs = np.array([])
# for i in range(arr.shape[0]):
#     ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "g-*")
#     xs = np.concatenate([xs, arr[:, :, 0].ravel()])
#     ys = np.concatenate([ys, arr[:, :, 1].ravel()])
#     zs = np.concatenate([zs, arr[:, :, 2].ravel()])
# arr = ac_point.controlpoints[5:]
# for i in range(arr.shape[0]):
#     ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")
#     xs = np.concatenate([xs, arr[:, :, 0].ravel()])
#     ys = np.concatenate([ys, arr[:, :, 1].ravel()])
#     zs = np.concatenate([zs, arr[:, :, 2].ravel()])
# # for i in range(arr.shape[1]):
# #     ax.plot(arr[:, i, 0], arr[:, i, 1], arr[:, i, 2], "g-")
# # Set scale

# ax.set_box_aspect(
#     (np.ptp(xs), np.ptp(ys), np.ptp(zs))
# )  # aspect ratio is 1:1:1 in data space
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.show()


T = np.eye(4)
extent = 100
T[2, 3] = -extent / 2 + SLICER_DEPTH
slicer = trimesh.primitives.Box(extents=np.array([extent, extent, extent]), transform=T)
sheet_round_K0, _ = calc_mesh_boolean_and_edges(sheet_round_K0, slicer, "difference")
sheet_round_K1, _ = calc_mesh_boolean_and_edges(sheet_round_K1, slicer, "difference")
sheet_round_K2, _ = calc_mesh_boolean_and_edges(sheet_round_K2, slicer, "difference")
sheet_leaf_K0, _ = calc_mesh_boolean_and_edges(sheet_leaf_K0, slicer, "difference")
sheet_leaf_K1, _ = calc_mesh_boolean_and_edges(sheet_leaf_K1, slicer, "difference")
sheet_leaf_K2, _ = calc_mesh_boolean_and_edges(sheet_leaf_K2, slicer, "difference")
sheet_point_K0, _ = calc_mesh_boolean_and_edges(sheet_point_K0, slicer, "difference")
sheet_point_K1, _ = calc_mesh_boolean_and_edges(sheet_point_K1, slicer, "difference")
sheet_point_K2, _ = calc_mesh_boolean_and_edges(sheet_point_K2, slicer, "difference")
point, _ = calc_mesh_boolean_and_edges(ac_point.mesh, slicer, "difference")
leaf, _ = calc_mesh_boolean_and_edges(ac_leaf.mesh, slicer, "difference")
# ac1, _ = calc_mesh_boolean_and_edges(ac1.mesh, slicer, "difference")
# ac2, _ = calc_mesh_boolean_and_edges(ac2.mesh, slicer, "difference")
# ac3, _ = calc_mesh_boolean_and_edges(ac3.mesh, slicer, "difference")
# ac4, _ = calc_mesh_boolean_and_edges(ac4.mesh, slicer, "difference")
# ac1 = ac1.mesh
# ac2 = ac2.mesh
# ac3 = ac3.mesh
# ac4 = ac4.mesh

mesh_list = [
    sheet_round_K0,
    sheet_round_K1,
    sheet_round_K2,
    leaf,
    sheet_leaf_K0,
    sheet_leaf_K1,
    sheet_leaf_K2,
    point,
    sheet_point_K0,
    sheet_point_K1,
    sheet_point_K2,
    ac1,
    # ac2,
    # ac3,
    # ac4,
    volumetric.mesh,
]

mesh_dict = {
    "sheet_round_K0": sheet_round_K0,
    "sheet_round_K1": sheet_round_K1,
    "sheet_round_K2": sheet_round_K2,
    "sheet_leaf_K0": sheet_leaf_K0,
    "sheet_leaf_K1": sheet_leaf_K1,
    "sheet_leaf_K2": sheet_leaf_K2,
    "sheet_point_K0": sheet_point_K0,
    "sheet_point_K1": sheet_point_K1,
    "sheet_point_K2": sheet_point_K2,
    "point": point,
    "ac1": ac1,
    # "ac2": ac2,
    # "ac3": ac3,
    # "ac4": ac4,
    "volumetric": volumetric.mesh,
}

scene = trimesh.Scene()
shift = np.array([0.0, 0.0, 0.0])
for m in mesh_list:

    mesh = copy.deepcopy(m)

    # Shift so bounds in lower left corner
    mesh = mesh.apply_translation(np.array([0, mesh.extents[1] / 2, 0]))
    mesh = mesh.apply_translation(shift)

    scene.add_geometry(mesh)

    extents = np.copy(mesh.extents)
    extents[0] = 0
    extents[2] = 0
    shift += extents * 1.1
scene.show()

#######################
### Transformations ###
#######################


def find_line_mesh_intersection(mesh, vec, origin):

    # Find points that the vector is pointing to (correct side of mesh)
    angles = angle_between(mesh.vertices - origin, vec)
    pts = mesh.vertices[angles < np.pi / 2]

    # Calc dist from these points to the vector
    dists = np.linalg.norm(np.cross(pts - origin, vec), axis=1) / np.linalg.norm(vec)

    # Choose smallest dist as point
    idx = dists.argmin()
    xyz = pts[idx]

    return xyz


#######################
### Transformations ###
#######################

# For these transformations, consider that the base shape is centered at the origin, with its "sheet" dimension in the yz plane, and it's thickness dimension in the x plane. It also bends in the +X direction. Based on this, which transformations result in it being on the shape in the Up/Down position and pointing forward/left/back/right.

# Find r at center of sphere


vec_axial_component = np.array([1, 0, 0])
x_th = 0  # -np.pi / 4
y_th = 0  # np.pi / 4  # np.pi / 9
vec_to_J1 = np.array([0, np.sin(x_th), np.cos(x_th)])
vec_to_J1_orth = np.cross(vec_to_J1, vec_axial_component)
vec_to_J2 = np.array([0, np.sin(x_th), np.cos(x_th)])
vec_to_J2_orth = np.cross(vec_to_J2, vec_axial_component)

J1_pos = 0.5
J2_pos = 0.95
J1_xyz_U = (
    find_line_mesh_intersection(volumetric.mesh, vec_to_J1, volumetric.r(J1_pos))
    + APP_OFFSET * vec_to_J1
)
J1_xyz_D = (
    find_line_mesh_intersection(volumetric.mesh, -vec_to_J1, volumetric.r(J1_pos))
    + -APP_OFFSET * vec_to_J1
)
J2_xyz_U = (
    find_line_mesh_intersection(volumetric.mesh, vec_to_J2, volumetric.r(J2_pos))
    + APP_OFFSET * vec_to_J2
)
J2_xyz_D = (
    find_line_mesh_intersection(volumetric.mesh, -vec_to_J2, volumetric.r(J2_pos))
    + -APP_OFFSET * vec_to_J2
)
CO_xyz = find_line_mesh_intersection(
    volumetric.mesh, vec_axial_component, volumetric.r(J2_pos)
)
# CO_xyz = (
#     find_line_mesh_intersection(volumetric.mesh, vec_axial_component, volumetric.r(1.0))
#     + APP_OFFSET * vec_axial_component
# )

R_F_U = Rotation.from_euler("zyx", np.array([0 * np.pi / 2, y_th, -x_th])).as_matrix()
R_L_U = Rotation.from_euler("zyx", np.array([1 * np.pi / 2, y_th, -x_th])).as_matrix()
R_B_U = Rotation.from_euler("zyx", np.array([2 * np.pi / 2, y_th, -x_th])).as_matrix()
R_R_U = Rotation.from_euler("zyx", np.array([3 * np.pi / 2, y_th, -x_th])).as_matrix()

R_F_D = Rotation.from_euler(
    "zyx", np.array([0 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_L_D = Rotation.from_euler(
    "zyx", np.array([3 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_B_D = Rotation.from_euler(
    "zyx", np.array([2 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_R_D = Rotation.from_euler(
    "zyx", np.array([1 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()

# Collinear rotations
R_U = Rotation.from_euler(
    "zyx", np.array([np.pi, np.pi / 2, -x_th + 0 * np.pi / 2])
).as_matrix()
R_L = Rotation.from_euler(
    "zyx", np.array([np.pi, np.pi / 2, -x_th + 3 * np.pi / 2])
).as_matrix()
R_D = Rotation.from_euler(
    "zyx", np.array([np.pi, np.pi / 2, -x_th + 2 * np.pi / 2])
).as_matrix()
R_R = Rotation.from_euler(
    "zyx", np.array([np.pi, np.pi / 2, -x_th + 1 * np.pi / 2])
).as_matrix()


def calc_T(R, xyz):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


# J1, direction of curvature (forward, left, right, back), up/down
T_J1_B_U = calc_T(R_B_U, J1_xyz_U)
T_J1_B_D = calc_T(R_B_D, J1_xyz_D)
T_J1_L_U = calc_T(R_L_U, J1_xyz_U)
T_J1_L_D = calc_T(R_L_D, J1_xyz_D)
T_J1_F_U = calc_T(R_F_U, J1_xyz_U)
T_J1_F_D = calc_T(R_F_D, J1_xyz_D)
T_J1_R_U = calc_T(R_R_U, J1_xyz_U)
T_J1_R_D = calc_T(R_R_D, J1_xyz_D)

# J2
T_J2_B_U = calc_T(R_B_U, J2_xyz_U)
T_J2_B_D = calc_T(R_B_D, J2_xyz_D)
T_J2_L_U = calc_T(R_L_U, J2_xyz_U)
T_J2_L_D = calc_T(R_L_D, J2_xyz_D)
T_J2_F_U = calc_T(R_F_U, J2_xyz_U)
T_J2_F_D = calc_T(R_F_D, J2_xyz_D)
T_J2_R_U = calc_T(R_R_U, J2_xyz_U)
T_J2_R_D = calc_T(R_R_D, J2_xyz_D)

# Collinear
T_CO_U = calc_T(R_U, CO_xyz)
T_CO_L = calc_T(R_L, CO_xyz)
T_CO_D = calc_T(R_D, CO_xyz)
T_CO_R = calc_T(R_R, CO_xyz)

combs = []


T_dict = {
    "T_eye": np.eye(4),
    "T_J1_B_U": T_J1_B_U,
    "T_J1_B_D": T_J1_B_D,
    "T_J1_L_U": T_J1_L_U,
    "T_J1_L_D": T_J1_L_D,
    "T_J1_F_U": T_J1_F_U,
    "T_J1_F_D": T_J1_F_D,
    "T_J1_R_U": T_J1_R_U,
    "T_J1_R_D": T_J1_R_D,
    "T_J2_B_U": T_J2_B_U,
    "T_J2_B_D": T_J2_B_D,
    "T_J2_L_U": T_J2_L_U,
    "T_J2_L_D": T_J2_L_D,
    "T_J2_F_U": T_J2_F_U,
    "T_J2_F_D": T_J2_F_D,
    "T_J2_R_U": T_J2_R_U,
    "T_J2_R_D": T_J2_R_D,
    "T_CO_U": T_CO_U,
    "T_CO_L": T_CO_L,
    "T_CO_D": T_CO_D,
    "T_CO_R": T_CO_R,
}
# # Test directions of rotations
# appendage = sheet_point_K1
# mesh_list = [volumetric.mesh]
# mesh = appendage.copy()
# mesh.apply_transform(eval("T_J1_R_U"))
# mesh_list.append(mesh)

# mesh = appendage.copy()
# mesh.apply_transform(eval("T_CO_R"))
# mesh_list.append(mesh)

# scene = trimesh.Scene()
# scene.add_geometry(mesh_list)
# interface = trimesh.load_mesh("/home/williamsnider/Code/objects/assets/Interface_0024 v2.stl")

# # Align with shape
# T = np.eye(4)
# R = Rotation.from_euler("xyz", np.array([0, 0, -np.pi / 2])).as_matrix()
# T[:3, :3] = R
# T[:3, 3] = np.array([-25, 0, 0])
# interface.apply_transform(T)
# scene.add_geometry(interface)
# scene.show()


# J1
# for appendage_type in ["sheet_round", "sheet_leaf", "sheet_point"]:
#     for curvature_profile in ["K0", "K1", "K2"]:

#         appendage = eval(appendage_type + "_" + curvature_profile).copy()

#         for T1 in ["T_J1_F_U", "T_J1_L_U"]:
#             for hox in [False, True]:

#                 mesh_list = [volumetric.mesh]

#                 mesh = appendage.copy()
#                 mesh.apply_transform(eval(T1))
#                 mesh_list.append(mesh)

#                 if hox == True:
#                     T2 = T1[:-1] + "D"
#                     hox_mesh = appendage.copy()
#                     hox_mesh.apply_transform(eval(T2))
#                     mesh_list.append(hox_mesh)

#                 # Fuse meshes
#                 meshA = mesh_list[0]
#                 for meshB in mesh_list[1:]:
#                     meshA = fuse_meshes(meshA, meshB, 2, "union")
#                 meshA.show(smooth=False)
#                 # scene = trimesh.Scene()
#                 # scene.add_geometry(mesh_list)
#                 # scene.show()

# J2
# for appendage_type in ["sheet_round", "sheet_leaf", "sheet_point"]:
#     for curvature_profile in ["K0", "K1", "K2"]:

#         appendage = eval(appendage_type + "_" + curvature_profile).copy()

#         for T1 in ["T_J2_L_U"]:
#             for hox in [False, True]:

#                 mesh_list = [volumetric.mesh]

#                 mesh = appendage.copy()
#                 mesh.apply_transform(eval(T1))
#                 mesh_list.append(mesh)

#                 if hox == True:
#                     T2 = T1[:-1] + "D"
#                     hox_mesh = appendage.copy()
#                     hox_mesh.apply_transform(eval(T2))
#                     mesh_list.append(hox_mesh)

#                 # Fuse meshes
#                 meshA = mesh_list[0]
#                 for meshB in mesh_list[1:]:
#                     meshA = fuse_meshes(meshA, meshB, 2, "union")
#                 meshA.show(smooth=False)

# # J1 and J2
# for appendage_type in ["sheet_point"]:
#     for curvature_profile in ["K0", "K1", "K2"]:

#         appendage = eval(appendage_type + "_" + curvature_profile).copy()

#         for hox in [False, True]:

#             T1 = "T_J1_L_U"
#             T2 = "T_J2_L_U"

#             mesh_list = [volumetric.mesh]

#             mesh = appendage.copy()
#             mesh.apply_transform(eval(T1))
#             mesh_list.append(mesh)

#             mesh = appendage.copy()
#             mesh.apply_transform(eval(T2))
#             mesh_list.append(mesh)

#             if hox == True:
#                 T1_hox = T1[:-1] + "D"
#                 hox_mesh = appendage.copy()
#                 hox_mesh.apply_transform(eval(T1_hox))
#                 mesh_list.append(hox_mesh)

#                 T2_hox = T2[:-1] + "D"
#                 mesh = appendage.copy()
#                 mesh.apply_transform(eval(T2_hox))
#                 mesh_list.append(mesh)

#             # Fuse meshes
#             meshA = mesh_list[0]
#             for meshB in mesh_list[1:]:
#                 meshA = fuse_meshes(meshA, meshB, 2, "union")
#             # meshA.show(smooth=False)

#             # Add interface
#             interface = load_interface(INTERFACE_PATH, "0_0_0_0")

#             # Linear 0.5 segment
#             post_backbone_cp = np.hstack(
#                 [
#                     np.linspace(POST_OFFSET, INTERFACE_SHIFT - POST_OFFSET, NUM_CP_PER_BACKBONE).reshape(-1, 1),
#                     np.zeros((NUM_CP_PER_BACKBONE, 1)),
#                     np.zeros((NUM_CP_PER_BACKBONE, 1)),
#                 ]
#             )

#             # Add post
#             POST_RADIUS = VOLUMETRIC_RADII[0]
#             OVERLAP_OFFSET = 1
#             NUM_CP_PER_CROSS_SECTION = 16
#             post_backbone = Backbone(post_backbone_cp, reparameterize=True)
#             post_radius = 5
#             post_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
#             post_cp = np.hstack((POST_RADIUS * np.cos(post_th), POST_RADIUS * np.sin(post_th)))
#             post_cs_list = [
#                 CrossSection(controlpoints=post_cp, position=0.0),
#                 CrossSection(controlpoints=post_cp, position=0.01),
#                 CrossSection(controlpoints=post_cp, position=0.99),
#                 CrossSection(controlpoints=post_cp, position=1.0),
#             ]
#             post_ac = AxialComponent(post_backbone, post_cs_list, smooth_with_post=False)

#             # Fuse post with mesh
#             meshA = fuse_meshes(meshA, post_ac.mesh, 2, "union")

#             mesh_with_interface = fuse_meshes(meshA, interface, 0, "union")

#             bbox = trimesh.primitives.Box(extents=np.array([120, 45, 70]))
#             bbox.apply_translation(np.array([bbox.extents[0] / 2 + INTERFACE_SHIFT - 20, 0, 0]))
#             bbox.visual.face_colors = np.array([0, 255, 125, 25])
#             scene = trimesh.Scene()
#             scene.add_geometry(bbox)
#             scene.add_geometry(mesh_with_interface)
#             scene.show()

# # J1 and collinear
# for J1_appendage_type in ["sheet_point"]:

#     for CO_appendage_type in ["sheet_point"]:
#         for curvature_profile in ["K0"]:

#             J1_appendage = eval(J1_appendage_type + "_" + curvature_profile).copy()
#             CO_appendage = eval(CO_appendage_type + "_" + curvature_profile).copy()

#             for hox in [False, True]:

#                 for J1_T in ["T_J1_F_U", "T_J1_L_U"]:
#                     for CO_T in ["T_CO_U", "T_CO_L"]:

#                         mesh_list = [volumetric.mesh]

#                         J1 = J1_appendage.copy()
#                         J1.apply_transform(eval(J1_T))
#                         mesh_list.append(J1)

#                         CO = CO_appendage.copy()
#                         CO.apply_transform(eval(CO_T))
#                         mesh_list.append(CO)

#                         if hox == True:

#                             if J1_T[-1] == "U":
#                                 J1_T_hox = J1_T[:-1] + "D"
#                             elif J1_T[-1] == "L":
#                                 J1_T_hox = J1_T[:-1] + "R"
#                             else:
#                                 raise NotImplementedError

#                             hox_mesh = J1_appendage.copy()
#                             hox_mesh.apply_transform(eval(J1_T_hox))
#                             mesh_list.append(hox_mesh)

#                             # CO_T_hox = CO_T[:-1] + "D"
#                             # mesh = appendage.copy()
#                             # mesh.apply_transform(eval(T2_hox))
#                             # mesh_list.append(mesh)

#                         # Fuse meshes
#                         meshA = mesh_list[0]
#                         for meshB in mesh_list[1:]:
#                             meshA = fuse_meshes(meshA, meshB, fairing_distance, "union")
#                         # meshA.show(smooth=False)

#                         # Add interface
#                         interface = load_interface(INTERFACE_PATH, "0_0_0_0")

#                         # Linear 0.5 segment
#                         post_backbone_cp = np.hstack(
#                             [
#                                 np.linspace(POST_OFFSET, INTERFACE_SHIFT - POST_OFFSET, NUM_CP_PER_BACKBONE).reshape(
#                                     -1, 1
#                                 ),
#                                 np.zeros((NUM_CP_PER_BACKBONE, 1)),
#                                 np.zeros((NUM_CP_PER_BACKBONE, 1)),
#                             ]
#                         )

#                         # Add post
#                         POST_RADIUS = VOLUMETRIC_RADII[0]
#                         OVERLAP_OFFSET = 1
#                         NUM_CP_PER_CROSS_SECTION = 16
#                         post_backbone = Backbone(post_backbone_cp, reparameterize=True)
#                         post_radius = 5
#                         post_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
#                         post_cp = np.hstack((POST_RADIUS * np.cos(post_th), POST_RADIUS * np.sin(post_th)))
#                         post_cs_list = [
#                             CrossSection(controlpoints=post_cp, position=0.0),
#                             CrossSection(controlpoints=post_cp, position=0.01),
#                             CrossSection(controlpoints=post_cp, position=0.99),
#                             CrossSection(controlpoints=post_cp, position=1.0),
#                         ]
#                         post_ac = AxialComponent(post_backbone, post_cs_list, smooth_with_post=False)

#                         # Fuse post with mesh
#                         meshA = fuse_meshes(meshA, post_ac.mesh, 2, "union")

#                         mesh_with_interface = fuse_meshes(meshA, interface, 0, "union")

#                         bbox = trimesh.primitives.Box(extents=np.array([120, 45, 70]))
#                         bbox.apply_translation(np.array([bbox.extents[0] / 2 + INTERFACE_SHIFT - 20, 0, 0]))
#                         bbox.visual.face_colors = np.array([0, 255, 125, 25])
#                         scene = trimesh.Scene()
#                         scene.add_geometry(bbox)
#                         scene.add_geometry(mesh_with_interface)
#                         scene.show()


# Post

post_backbone_cp = np.hstack(
    [
        np.linspace(
            POST_OFFSET, INTERFACE_SHIFT - POST_OFFSET, NUM_CP_PER_BACKBONE
        ).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
post_backbone = Backbone(post_backbone_cp, reparameterize=True)
post_radius = 5
post_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(
    -1, 1
)
post_cp = np.hstack((POST_RADIUS * np.cos(post_th), POST_RADIUS * np.sin(post_th)))
post_cs_list = [
    CrossSection(controlpoints=post_cp, position=0.0),
    CrossSection(controlpoints=post_cp, position=0.01),
    CrossSection(controlpoints=post_cp, position=0.99),
    CrossSection(controlpoints=post_cp, position=1.0),
]
post_ac = AxialComponent(post_backbone, post_cs_list, smooth_with_post=False)


# Add interface


def slice_mesh(mesh, extent, T):
    mesh = mesh.copy()
    slicer = trimesh.primitives.Box(
        extents=np.array([extent, extent, extent]), transform=T
    )
    split_mesh, _ = calc_mesh_boolean_and_edges(mesh, slicer, "difference")

    return split_mesh


def build_shape(inputs):

    shape_number, mesh_names, T_names = inputs

    # Get mesh and T values
    mesh_list = []
    T_list = []
    for i in range(len(mesh_names)):
        mesh_list.append(mesh_dict[mesh_names[i]])
        T_list.append(T_dict[T_names[i]])

    # # Additional verts to fair
    # add_vert_indices = []
    # for i, mesh in enumerate(mesh_list):

    #     if i == 0:
    #         add_vert_indices.append(np.array([]))
    #     else:
    #         add_vert_indices.append(mesh.vertices[:, 2] < 0)

    # Transform meshes
    new_mesh_list = []
    for i, mesh in enumerate(mesh_list):
        mesh = mesh.copy()
        new_mesh_list.append(mesh.apply_transform(T_list[i]))

    # Split mesh by yz-plane to prevent going through shape
    split_mesh_list = []
    for i, mesh in enumerate(new_mesh_list):

        mesh = mesh.copy()
        T_name = T_names[i]

        # Make slicer mesh
        T = np.eye(4)
        extent = 100
        if T_name in [
            "T_J1_B_U",
            "T_J1_L_U",
            "T_J1_F_U",
            "T_J1_R_U",
            "T_J2_B_U",
            "T_J2_L_U",
            "T_J2_F_U",
            "T_J2_R_U",
        ]:
            T[2, 3] = -extent / 2
            split_mesh = slice_mesh(mesh, extent, T)

        elif T_name in [
            "T_J1_B_D",
            "T_J1_L_D",
            "T_J1_F_D",
            "T_J1_R_D",
            "T_J2_B_D",
            "T_J2_L_D",
            "T_J2_F_D",
            "T_J2_R_D",
        ]:
            T[2, 3] = extent / 2
            split_mesh = slice_mesh(mesh, extent, T)

        elif T_name in [
            "T_eye",
            "T_CO_U",
            "T_CO_L",
            "T_CO_D",
            "T_CO_R",
        ]:
            split_mesh = mesh
        else:
            raise NotImplementedError

        split_mesh_list.append(split_mesh)

    # slicer = trimesh.primitives.Box(
    #     extents=np.array([extent, extent, extent]), transform=T
    # )
    # scene = trimesh.Scene()
    # scene.add_geometry([mesh, slicer])
    # scene.show()

    # Fuse meshes
    meshA = split_mesh_list[0]
    i = 1
    for meshB in split_mesh_list[1:]:
        meshA = fuse_meshes(meshA, meshB, fairing_distance, "union")

    # Attach post
    meshA = fuse_meshes(meshA, post_ac.mesh, 2, "union")

    # Attach interface
    label = str(shape_number).zfill(4)
    # label_split = [label[i] for i in range(len(label))]
    # label_formatted = "_".join(label_split)  # Underscores denote new lines
    interface = load_interface(INTERFACE_PATH, label)
    mesh_with_interface = fuse_meshes(meshA, interface, 0, "union")

    # Construct save_dir
    if SAVE_DIR.is_dir() is False:
        SAVE_DIR.mkdir(parents=True)

    # Export
    filename = Path(SAVE_DIR, label).with_suffix(".stl")
    mesh_with_interface.export(filename)

    mesh_with_interface.show(smooth=False)
    import trimesh.scene

    scene = trimesh.Scene()
    scene.add_geometry(mesh_with_interface)
    scene.lights = trimesh.scene.lighting.autolight(scene)
    return mesh_with_interface


count = 15


# def show_scene(inputs):
#     scene = trimesh.Scene()

#     shape_number, mesh_names, T_names = inputs

#     # Get mesh and T values
#     mesh_list = []
#     T_list = []
#     for i in range(len(mesh_names)):
#         mesh_list.append(mesh_dict[mesh_names[i]])
#         T_list.append(T_dict[T_names[i]])

#     # Transform meshes
#     new_mesh_list = []
#     for i, mesh in enumerate(mesh_list):
#         mesh = mesh.copy()
#         new_mesh_list.append(mesh.apply_transform(T_list[i]))

#     scene.add_geometry(new_mesh_list)
#     scene.show()


# # Samples to print

# Sheet point
inputs = (
    count,
    [
        "volumetric",
        "ac1",
        "ac1",
        "ac1",
        "ac1",
        "ac1",
    ],
    [
        "T_eye",
        "T_CO_U",
        "T_J1_F_U",
        "T_J2_F_U",
        "T_J1_F_D",
        "T_J2_F_D",
    ],
)
# show_scene(inputs)
s = build_shape(inputs)
count += 1

inputs = (
    count,
    [
        "volumetric",
        "sheet_point_K0",
        "sheet_point_K1",
        "sheet_point_K1",
        "sheet_point_K1",
        "sheet_point_K1",
    ],
    [
        "T_eye",
        "T_CO_L",
        "T_J1_F_U",
        "T_J2_F_U",
        "T_J1_B_D",
        "T_J2_B_D",
    ],
)
s = build_shape(inputs)
count += 1

inputs = (
    count,
    [
        "volumetric",
        "sheet_leaf_K0",
        "sheet_leaf_K0",
        "sheet_point_K0",
    ],
    [
        "T_eye",
        "T_J1_F_U",
        "T_J1_F_D",
        "T_CO_L",
    ],
)
s = build_shape(inputs)
count += 1

inputs = (
    count,
    [
        "volumetric",
        "sheet_leaf_K0",
        "sheet_leaf_K0",
        "sheet_leaf_K0",
    ],
    [
        "T_eye",
        "T_J1_L_U",
        "T_J1_L_D",
        "T_CO_L",
    ],
)
s = build_shape(inputs)
count += 1

inputs = (
    count,
    [
        "volumetric",
        "sheet_round_K1",
        "sheet_round_K1",
        "sheet_round_K1",
        "sheet_round_K1",
        "sheet_round_K0",
    ],
    [
        "T_eye",
        "T_J1_L_U",
        "T_J1_L_D",
        "T_J2_L_U",
        "T_J2_L_D",
        "T_CO_L",
    ],
)
s = build_shape(inputs)
count += 1
# Those three tilted as well
