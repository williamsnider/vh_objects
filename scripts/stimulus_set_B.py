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
    ROUND_RADIUS,
)

FAIRING_DISTANCE = 3
POST_Z_SHIFT = 0

uu = 75
vv = 75

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


app_round_concave = trimesh.primitives.creation.icosphere(3, radius=ROUND_RADIUS)
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
base_sheet = round_cs_cp * ROUND_RADIUS
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
NUM_POINT_CS = NUM_CS * 2

# Calculate widths
point_x = np.linspace(0, APPENDAGE_LENGTH, 3)
point_y = POINT_RADII  # 3 radii determine polynomial form
point_poly = np.polyfit(point_x, point_y, 2)

# But sample along 4th position to ensure smooth transition into shape
xvals = np.linspace(-APPENDAGE_LENGTH * 2 / 3, APPENDAGE_LENGTH, NUM_POINT_CS)
widths = np.polyval(point_poly, xvals)

# Calculate z_levels
z_levels = np.linspace(-APPENDAGE_LENGTH * 2 / 3, APPENDAGE_LENGTH, NUM_POINT_CS)

# Assign controlpoints
cp = np.zeros((NUM_POINT_CS, 8, 3))
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
sheet_point_K2 = sheet_point_K1.copy()
sheet_point_K2.apply_transform(T_K2)

# Slice certain meshes to keep it from going through shape
extent = 100
T = np.eye(4)
T[2, 3] = -extent / 2 - 1 * X_WIDTH
app_point_convex = slice_mesh(
    app_point_convex,
    100,
    T,
)

sheet_point_K0 = slice_mesh(sheet_point_K0, extent, T)
sheet_point_K1 = slice_mesh(sheet_point_K1, extent, T)
sheet_point_K2 = slice_mesh(sheet_point_K2, extent, T)
sheet_leaf_K0 = slice_mesh(sheet_leaf_K0, extent, T)
sheet_leaf_K1 = slice_mesh(sheet_leaf_K1, extent, T)
sheet_leaf_K2 = slice_mesh(sheet_leaf_K2, extent, T)
sheet_round_K0 = slice_mesh(sheet_round_K0, extent, T)
sheet_round_K1 = slice_mesh(sheet_round_K1, extent, T)
sheet_round_K2 = slice_mesh(sheet_round_K2, extent, T)

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
J2_pos = np.array([SEGMENT_LENGTH - X_WIDTH - 1, 0, 0])
CO_xyz = np.array([SEGMENT_LENGTH - X_WIDTH / 2, 0, 0])

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
count = 0


def build_shape(inputs):
    mesh_list = [mesh_dict[n].copy() for n in inputs[0]]
    T_list = [T_dict[n] for n in inputs[1]]
    boolean_list = inputs[2]
    label = "B" + str(inputs[3].zfill(3))

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

# Thin backbone alone
# Assign combination of inputs
comb = [
    ["thin"],
    ["T_eye"],
    [None],
    str(count),
    "",
    SAVE_DIR,
    "T_eye",
    FAIRING_DISTANCE,
    POST_Z_SHIFT,
]
combs.append(comb)
count += 1

# J1 and J2 and Collinear
for J_app in ["app1", "app2", "app3", "app4"]:
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

            for hox in [False, True]:

                # Transformation matrices
                T_J1 = "T_thin_J1_F_U"
                T_J2 = "T_thin_J2_F_U"
                T_CO = "T_CO_U"
                T_J1_hox = T_J1[:-1] + "D"
                T_J2_hox = T_J2[:-1] + "D"

                # Construct mesh and T_list
                mesh_list = ["thin"]
                T_list = ["T_eye"]

                # Add CO
                if withCO == True:
                    mesh_list.append(CO_app)
                    T_list.append(T_CO)

                # Add J1
                if withJ1 == True:
                    mesh_list.append(J_app)
                    T_list.append(T_J1)

                    if hox == True:
                        mesh_list.append(J_app)
                        T_list.append(T_J1_hox)

                # Add J2
                if withJ2 == True:
                    mesh_list.append(J_app)
                    T_list.append(T_J2)

                    if hox == True:
                        mesh_list.append(J_app)
                        T_list.append(T_J2_hox)

                # Prevent duplicates for shapes by not looping for CO if not present
                if withCO == False and CO_app != "app1":
                    continue

                boolean_list = ["union" for _ in mesh_list]

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

#################################
### Case: Volumetric Backbone ###
#################################

# Volumetric without appendages
comb = [
    ["volumetric"],
    ["T_eye"],
    [None],
    str(count),
    "",
    SAVE_DIR,
    "T_eye",
    FAIRING_DISTANCE,
    POST_Z_SHIFT,
]
combs.append(comb)
count += 1

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

#########################################
### Case: Sheets with Collinear (K=0) ###
#########################################

# J1 and J2 and Collinear
for J_app in [
    "sheet_round_K0",
    "sheet_round_K1",
    "sheet_round_K2",
    "sheet_point_K0",
    "sheet_point_K1",
    "sheet_point_K2",
    "sheet_leaf_K0",
    "sheet_leaf_K1",
    "sheet_leaf_K2",
]:
    for CO_app in ["sheet_round_K0", "sheet_point_K0", "sheet_leaf_K0"]:

        for J_ori in ["L_U", "F_U"]:

            for CO_ori in ["L", "U"]:

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

                    # Constraint: thin side of sheet must be parallel to medial axis of base shape for J2
                    if withJ2 == True and J_ori == "F_U":
                        continue

                    # Constraint: J1/J2 + Collinear has 1 curvature profile
                    if (
                        (withJ2 == True or withJ1 == True)
                        and withCO == True
                        and any(["_K1" in J_app, "_K2" in J_app])
                    ):
                        continue

                    for hox in [False, True]:

                        # Transformation matrices
                        T_J1 = "T_volu_J1_" + J_ori
                        T_J2 = "T_volu_J2_" + J_ori
                        T_CO = "T_CO_" + CO_ori
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
                        if withCO == False and (
                            CO_app != "sheet_round_K0" or CO_ori != "L"
                        ):
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

                        combs.append(comb)
                        count += 1


##########################################
### Case: Sheets with Collinear (K!=0) ###
##########################################

# J1 and J2 and Collinear
for J_app in [
    "sheet_round_K1",
    "sheet_point_K1",
    "sheet_leaf_K1",
]:
    for CO_app in ["sheet_round_K1", "sheet_point_K1", "sheet_leaf_K1"]:

        for J_ori in ["F_U", "L_U"]:

            for CO_ori in ["U", "L", "D"]:

                # Iterate through J1+CO, J2+CO, J1+J2+CO
                for J1_J2_CO in [
                    [True, False, True],
                    [False, True, True],
                    [True, True, True],
                ]:

                    withJ1, withJ2, withCO = J1_J2_CO

                    # Constraint: thin side of sheet must be parallel to medial axis of base shape for J2
                    if withJ2 == True and J_ori == "F_U":
                        continue

                    # # Constraint: J1/J2 + Collinear has 1 curvature profile
                    # if (withJ2 == True or withJ1 == True) and (withCO == True):
                    #     continue

                    for hox in [False, True]:

                        # Transformation matrices
                        T_J1 = "T_volu_J1_" + J_ori
                        T_J2 = "T_volu_J2_" + J_ori
                        T_CO = "T_CO_" + CO_ori
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

                        # # Prevent duplicates for shapes by not looping for CO if not present
                        # if withCO == False and (CO_app != "sheet_round_K0" or CO_ori != "L"):
                        #     continue

                        # Skip symmetric shape
                        if hox == True and CO_ori == "D":
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

                        combs.append(comb)
                        count += 1

########################
### Construct Shapes ###
########################


# for comb in combs[354:355]:
#     build_shape(comb)

if __name__ == "__main__":

    with Pool() as pool:
        mapped_values = list(
            tqdm(pool.imap_unordered(build_shape, combs), total=len(combs))
        )
