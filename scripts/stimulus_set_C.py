# Linear segment
import copy
import numpy as np
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.utilities import (
    approximate_arc,
    make_mesh,
    make_surface,
    calc_hemisphere_controlpoints,
    angle_between,
    fuse_meshes,
)
from objects.shape import Shape
from scripts.sheets import construct_sheet, bend_sheet, make_base_cp
from scripts.hemi import calc_sphere_controlpoints
import trimesh
from scripts.sheets import plot_arr
from scipy.spatial.transform.rotation import Rotation

# TODO: Check RADII
# TODO: Ensure parity between orthogonal and colinear components -- probably best to fuse complete components
# TODO: Think about slanting everything away for ease of 3D printing (without supports)
# TODO: Orient along z-axis

### Parameters ###
NUM_CP_PER_BACKBONE = 5
SEGMENT_LENGTH = 25
NUM_CS = 11
X_WIDTH = 6  # base radius off which other features are derived
VOLUMETRIC_RADII = np.array(
    [X_WIDTH, 2 * X_WIDTH, X_WIDTH]
)  # Middle cs radius is this times larger than edge cs
SHEET_THICKNESS = 3
NUM_CP_PER_BASE_SHEET = 16
NUM_CS_PER_SHEET = 11

POINT_RADII = np.array([X_WIDTH, 0.4 * X_WIDTH, 0.1 * X_WIDTH])
LEAF_RADII = np.array([0.75 * X_WIDTH, 1.5 * X_WIDTH, 0.25 * X_WIDTH])
APPENDAGE_LENGTH = 15
POINT_ROUNDOVER_OFFSET = SHEET_THICKNESS / 3
# SPHERE_RADIUS = 15

# Derived parameters
# cs_radii = np.arange(3) * X_WIDTH
# appendage_length = SEGMENT_LENGTH / 2

##############################
### Cross Section Profiles ###
##############################

# Circular cross sections
t = np.linspace(0, 2 * np.pi, 8, endpoint=False).reshape(-1, 1)
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
x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH * 2
y = VOLUMETRIC_RADII
volumetric_poly = np.polyfit(x, y, 2)
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

# Backbone for bending sheets
b_cp = approximate_arc(np.pi / 2, APPENDAGE_LENGTH, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
backbone_appendage = Backbone(b_cp, reparameterize=True)

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
sheet_round_K0 = make_mesh(surf, 100, 100)

# Bend round sheet
bent_cp = bend_sheet(cp, backbone_x_width, X_WIDTH)
surf = make_surface(bent_cp)
sheet_round_K1 = make_mesh(surf, 100, 100)
sheet_round_K2 = sheet_round_K1.copy()
sheet_round_K2.apply_transform(T_K2)


# Leaf sheet
num_edge_cp = 7
base_round_cp = 3
top_round_cp = 1
x = np.linspace(0, 1, 3) * APPENDAGE_LENGTH
y = LEAF_RADII
poly = np.polyfit(x, y, 2)
leaf_cp = make_base_cp(poly, x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = leaf_cp.mean(axis=0)
leaf_cp = leaf_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(leaf_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
surf = make_surface(cp)
sheet_leaf_K0 = make_mesh(surf, 100, 100)

# Bent leaf sheet
bent_cp = bend_sheet(cp, backbone_appendage, x[2] - x[0])
surf = make_surface(bent_cp)
sheet_leaf_K1 = make_mesh(
    surf, 100, 100
)  # TODO: Has artifact, fix after deciding on thickness/size
sheet_leaf_K2 = sheet_leaf_K1.copy()
sheet_leaf_K2.apply_transform(T_K2)
# Point sheet

# Calculate widths
px = np.linspace(0, APPENDAGE_LENGTH, 3)
py = POINT_RADII
poly = np.polyfit(px, py, 2)
xvals = np.linspace(0, APPENDAGE_LENGTH, NUM_CS)
widths = np.polyval(poly, xvals)

# Calculate z_levels
z_levels = np.linspace(0, APPENDAGE_LENGTH, NUM_CS)

# Assign controlpoints
cp = np.zeros((NUM_CS, 8, 3))
for i, width in enumerate(widths):

    if width == 0:
        inner = np.zeros((8, 2))
    else:
        inner = np.array(
            [
                [SHEET_THICKNESS / 2, 0],
                [SHEET_THICKNESS / 2, width / 2],
                [0, width / 2 + POINT_ROUNDOVER_OFFSET],
                [-SHEET_THICKNESS / 2, width / 2],
                [-SHEET_THICKNESS / 2, 0],
                [-SHEET_THICKNESS / 2, -width / 2],
                [0, -width / 2 - POINT_ROUNDOVER_OFFSET],
                [SHEET_THICKNESS / 2, -width / 2],
            ]
        )

    xyz = np.hstack([inner, z_levels[i] * np.ones((inner.shape[0], 1))])

    cp[i, :, :] = xyz

# Roundover edges

side_y = POINT_RADII + POINT_ROUNDOVER_OFFSET
side_poly = np.polyfit(x, side_y, 2)
bot = calc_hemisphere_controlpoints(
    cp[0],
    np.array([0, 0, 1]),
    cp[0].mean(axis=0),
    side_poly,
    x[0],
    morph_to_ellipse=True,
)
top = calc_hemisphere_controlpoints(
    cp[-1],
    np.array([0, 0, 1]),
    cp[-1].mean(axis=0),
    side_poly,
    x[-1],
    morph_to_ellipse=True,
)
cp = np.vstack([bot, cp, top[-2::-1]])
surf = make_surface(cp)
sheet_point_K0 = make_mesh(surf, 100, 100)

bent_cp = bend_sheet(cp, backbone_appendage, x[2] - x[0])
surf = make_surface(bent_cp)
sheet_point_K1 = make_mesh(
    surf, 100, 100
)  # TODO: Has artifact, fix after deciding on thickness/size
# sheet_point_bent.show(smooth=False)
sheet_point_K2 = sheet_point_K1.copy()
sheet_point_K2.apply_transform(T_K2)
# Sphere

num_cp = 11
t = np.linspace(0, 2 * np.pi, 16, endpoint=False).reshape(-1, 1)
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

surf = make_surface(cp)
sphere = make_mesh(surf, 100, 100)
max_bbox = trimesh.creation.box([100, 45, 45])
inch_sphere = trimesh.creation.icosphere(3, 25.4 / 2)

mesh_list = [
    sheet_round_K0,
    sheet_round_K1,
    sheet_round_K2,
    sheet_leaf_K0,
    sheet_leaf_K1,
    sheet_leaf_K2,
    sheet_point_K0,
    sheet_point_K1,
    sheet_point_K2,
    sphere,
    volumetric.mesh,
    max_bbox,
    inch_sphere,
]

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


vec_axial_component = np.array([1, 0, 0])
vec_to_J1 = volumetric.B(0.5)
vec_to_J1_orth = np.cross(vec_to_J1, vec_axial_component)
vec_to_J2 = volumetric.B(1.0)
vec_to_J2_orth = np.cross(vec_to_J2, vec_axial_component)

J1_xyz_U = find_line_mesh_intersection(volumetric.mesh, vec_to_J1, volumetric.r(0.5))
J1_xyz_D = find_line_mesh_intersection(volumetric.mesh, -vec_to_J1, volumetric.r(0.5))
J2_xyz_U = find_line_mesh_intersection(volumetric.mesh, vec_to_J2, volumetric.r(1.0))
J2_xyz_D = find_line_mesh_intersection(volumetric.mesh, -vec_to_J2, volumetric.r(1.0))
CO_xyz = find_line_mesh_intersection(
    volumetric.mesh, vec_axial_component, volumetric.r(1.0)
)

R_F_U = np.eye(3)
R_L_U = R_F_U @ Rotation.from_rotvec(np.pi / 2 * vec_to_J1).as_matrix()
R_B_U = R_L_U @ Rotation.from_rotvec(np.pi / 2 * vec_to_J1).as_matrix()
R_R_U = R_B_U @ Rotation.from_rotvec(np.pi / 2 * vec_to_J1).as_matrix()

R_F_D = R_F_U @ Rotation.from_rotvec(np.pi * vec_axial_component).as_matrix()
R_L_D = R_F_D @ Rotation.from_rotvec(-np.pi / 2 * vec_to_J1).as_matrix()
R_B_D = R_L_D @ Rotation.from_rotvec(-np.pi / 2 * vec_to_J1).as_matrix()
R_R_D = R_B_D @ Rotation.from_rotvec(-np.pi / 2 * vec_to_J1).as_matrix()

# Collinear rotations
R_U = R_F_D @ Rotation.from_rotvec(np.pi / 2 * vec_to_J1_orth).as_matrix()
R_L = R_U @ Rotation.from_rotvec(-np.pi / 2 * vec_to_J1).as_matrix()
R_D = R_L @ Rotation.from_rotvec(-np.pi / 2 * vec_to_J1).as_matrix()
R_R = R_D @ Rotation.from_rotvec(-np.pi / 2 * vec_to_J1).as_matrix()


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

# J1

# 3 radius profiles
# 3 curvature profiles
# 2 hox heads
# 2 orientations
for appendage_type in ["sheet_round", "sheet_leaf", "sheet_point"]:
    for curvature_profile in ["K0", "K1", "K2"]:

        appendage = eval(appendage_type + "_" + curvature_profile).copy()

        for T1 in ["T_J1_F_U", "T_J1_L_U"]:
            for hox in [False, True]:

                mesh_list = [volumetric.mesh]

                mesh = appendage.copy()
                mesh.apply_transform(eval(T1))
                mesh_list.append(mesh)

                if hox == True:
                    T2 = T1[:-1] + "D"
                    hox_mesh = appendage.copy()
                    hox_mesh.apply_transform(eval(T2))
                    mesh_list.append(hox_mesh)

                # Fuse meshes
                meshA = mesh_list[0]
                for meshB in mesh_list[1:]:
                    meshA = fuse_meshes(meshA, meshB, 2, "union")
                meshA.show(smooth=False)
                # scene = trimesh.Scene()
                # scene.add_geometry(mesh_list)
                # scene.show()


mesh = sheet_point_K0

mesh_list = [
    volumetric.mesh,
    mesh.copy().apply_transform(T_J1_B_U),
    mesh.copy().apply_transform(T_J1_L_U),
    mesh.copy().apply_transform(T_CO_U),
    mesh.copy().apply_transform(T_CO_R),
    # mesh.copy().apply_transform(T_CO_D),
    # mesh.copy().apply_transform(T_CO_L),
]
for m in mesh_list:
    scene.add_geometry(m)
scene.show()
