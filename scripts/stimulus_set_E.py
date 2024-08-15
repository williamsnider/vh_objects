import numpy as np
from vh_objects.utilities import make_mesh, make_surface
from scripts.sheets import construct_sheet, bend_sheet, make_base_cp, plot_arr
from vh_objects.backbone import Backbone
from vh_objects.utilities import approximate_arc
from vh_objects.shape import Shape
from scipy.spatial.transform import Rotation
import trimesh
from scripts.stim_set_common import (
    create_scene,
    load_cap,
    UU,
    VV,
    K_PERPENDICULAR,
    SHEET_THICKNESS,
    num_edge_cp,
    base_round_cp,
    top_round_cp,
    NUM_CS_PER_SHEET,
    export_shape,
)
import copy
from trimesh.transformations import rotation_matrix as rotvec2T
from pathlib import Path


def center_xy(mesh):
    mesh_copy = copy.deepcopy(mesh)

    bounds = mesh_copy.bounds

    # Center XY
    center = (bounds[0] + bounds[1]) / 2

    # Align to xy plane
    mesh_copy.apply_translation(-np.array([center[0], center[1], bounds[0, 2]]))

    return mesh_copy


##################
### TO DO LIST ###
##################

# Have better control of roundovers so that the resulting sheet is an exact length
# Fuse shapes all at once (have the fairing process work on several, not one at a time)

##################
### Parameters ###
##################
mesh_fairing_distance = 1
K_PERPENDICULAR_SMALL = 1 / 25

T_K2 = np.eye(4)
T_K2[:3, :3] = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()


# Have bent sheet pointing upwards
T_K1 = np.eye(4)
R = Rotation.from_rotvec(-np.pi / 4 * np.array([0, 1, 0])).as_matrix()
T_K1[:3, :3] = R

# Ellipse sheet
ellipse_length = 25
ellipse_x = np.linspace(0, 1, 3) * (ellipse_length)
ellipse_y = np.array([20, 20, 20])
ellipse_poly = np.polyfit(ellipse_x, ellipse_y, 2)
ellipse_cp = make_base_cp(ellipse_poly, ellipse_x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = ellipse_cp.mean(axis=0)
ellipse_cp = ellipse_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(ellipse_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
cp[:, :, 2] -= cp[:, :, 2].min()  # Maybe a better way to do this
surf = make_surface(cp)
sheet_ellipse_K0 = make_mesh(surf, UU, VV)
sheet_ellipse_K0 = center_xy(sheet_ellipse_K0)

# Ellipse with positive perpendicular curvature
ellipse_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(0.0001, ellipse_height, 5)  # TODO: Fix this - veery ugly
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K0 = Backbone(b_cp, reparameterize=True)

bent_cp = bend_sheet(cp, b_appendage_K0, ellipse_height, K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_ellipse_K0p = make_mesh(surf, UU, VV)
sheet_ellipse_K0p = center_xy(sheet_ellipse_K0p)

ellipse_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(np.pi / 2, ellipse_height, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

# Bent ellipse
bent_cp = bend_sheet(cp, b_appendage_K1, ellipse_height, 0)
surf = make_surface(bent_cp)
sheet_ellipse_K1 = make_mesh(surf, UU, VV)
sheet_ellipse_K1.apply_transform(T_K1)
sheet_ellipse_K1 = center_xy(sheet_ellipse_K1)

# Bent ellipse with positive perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, ellipse_height, K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_ellipse_K1p = make_mesh(surf, UU, VV)
sheet_ellipse_K1p.apply_transform(T_K1)
sheet_ellipse_K1p = center_xy(sheet_ellipse_K1p)


# Bent ellipse with negative perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, ellipse_height, -K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_ellipse_K1n = make_mesh(surf, UU, VV)
sheet_ellipse_K1n.apply_transform(T_K1)
sheet_ellipse_K1n = center_xy(sheet_ellipse_K1n)


# Teardrop
teardrop_length = 35
teardrop_x = np.linspace(0, 1, 3) * (teardrop_length)
teardrop_y = np.array([20, 15, 1])
teardrop_poly = np.polyfit(teardrop_x, teardrop_y, 2)
teardrop_cp = make_base_cp(teardrop_poly, teardrop_x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = teardrop_cp.mean(axis=0)
teardrop_cp = teardrop_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(teardrop_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
cp[:, :, 2] -= cp[:, :, 2].min()  # Maybe a better way to do this
surf = make_surface(cp)
sheet_teardrop_K0 = make_mesh(surf, UU, VV)
sheet_teardrop_K0 = center_xy(sheet_teardrop_K0)

# Teardrop with positive perpendicular curvature
teardrop_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(0.0001, teardrop_height, 5)  # TODO: Fix this - veery ugly
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K0 = Backbone(b_cp, reparameterize=True)

bent_cp = bend_sheet(cp, b_appendage_K0, teardrop_height, K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_teardrop_K0p = make_mesh(surf, UU, VV)
sheet_teardrop_K0p = center_xy(sheet_teardrop_K0p)

teardrop_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(np.pi / 2, teardrop_height, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

# Bent teardrop
bent_cp = bend_sheet(cp, b_appendage_K1, teardrop_height, 0)
surf = make_surface(bent_cp)
sheet_teardrop_K1 = make_mesh(surf, UU, VV)
sheet_teardrop_K1.apply_transform(T_K1)
sheet_teardrop_K1 = center_xy(sheet_teardrop_K1)

# Bent teardrop with positive perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, teardrop_height, K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_teardrop_K1p = make_mesh(surf, UU, VV)
sheet_teardrop_K1p.apply_transform(T_K1)
sheet_teardrop_K1p = center_xy(sheet_teardrop_K1p)

# Bent teardrop with negative perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, teardrop_height, -K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_teardrop_K1n = make_mesh(surf, UU, VV)
sheet_teardrop_K1n.apply_transform(T_K1)
sheet_teardrop_K1n = center_xy(sheet_teardrop_K1n)

# Flip teardrops
T_FLIP = np.eye(4)
T_FLIP[:3, :3] = Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
sheet_teardropf_K0 = sheet_teardrop_K0.copy()
sheet_teardropf_K0.apply_transform(T_FLIP)
sheet_teardropf_K0.apply_translation([0, 0, -sheet_teardropf_K0.bounds[0, 2]])
sheet_teardropf_K0 = center_xy(sheet_teardropf_K0)
sheet_teardropf_K0p = sheet_teardrop_K0p.copy()
sheet_teardropf_K0p.apply_transform(T_FLIP)
sheet_teardropf_K0p.apply_translation([0, 0, -sheet_teardropf_K0p.bounds[0, 2]])
sheet_teardropf_K0p = center_xy(sheet_teardropf_K0p)
sheet_teardropf_K1 = sheet_teardrop_K1.copy()
sheet_teardropf_K1.apply_transform(T_FLIP)
sheet_teardropf_K1.apply_translation([0, 0, -sheet_teardropf_K1.bounds[0, 2]])
sheet_teardropf_K1 = center_xy(sheet_teardropf_K1)
sheet_teardropf_K1p = sheet_teardrop_K1p.copy()
sheet_teardropf_K1p.apply_transform(T_FLIP)
sheet_teardropf_K1p.apply_translation([0, 0, -sheet_teardropf_K1p.bounds[0, 2]])
sheet_teardropf_K1p = center_xy(sheet_teardropf_K1p)
sheet_teardropf_K1n = sheet_teardrop_K1n.copy()
sheet_teardropf_K1n.apply_transform(T_FLIP)
sheet_teardropf_K1n.apply_translation([0, 0, -sheet_teardropf_K1n.bounds[0, 2]])
sheet_teardropf_K1n = center_xy(sheet_teardropf_K1n)

# Rectangular Sheet
rect_cp = ellipse_cp.copy()
rect_cp[:, 2] /= 1.25
rect_cp[:top_round_cp, 2] = rect_cp[:, 2].min()
rect_cp[top_round_cp + num_edge_cp : top_round_cp + num_edge_cp + top_round_cp, 2] = rect_cp[:, 2].max()
rect_cp[top_round_cp : top_round_cp + num_edge_cp, 2] = np.linspace(
    rect_cp[:, 2].min(), rect_cp[:, 2].max(), num_edge_cp
)
rect_cp[top_round_cp + num_edge_cp + top_round_cp :, 2] = np.linspace(
    rect_cp[:, 2].max(), rect_cp[:, 2].min(), num_edge_cp
)
mean_xyz = rect_cp.mean(axis=0)
rect_cp = rect_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(rect_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
cp[:, :, 2] -= cp[:, :, 2].min()  # Maybe a better way to do this
surf = make_surface(cp)
sheet_rect_K0 = make_mesh(surf, UU, VV)
sheet_rect_K0 = center_xy(sheet_rect_K0)

# Rectangular with positive perpendicular curvature
rect_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(0.0001, rect_height, 5)  # TODO: Fix this - veery ugly
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K0 = Backbone(b_cp, reparameterize=True)

bent_cp = bend_sheet(cp, b_appendage_K0, rect_height, K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_rect_K0p = make_mesh(surf, UU, VV)
sheet_rect_K0p = center_xy(sheet_rect_K0p)

rect_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(np.pi / 2, rect_height, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)


# Bent rectangular
bent_cp = bend_sheet(cp, b_appendage_K1, rect_height, 0)
surf = make_surface(bent_cp)
sheet_rect_K1 = make_mesh(surf, UU, VV)
sheet_rect_K1.apply_transform(T_K1)
sheet_rect_K1 = center_xy(sheet_rect_K1)

# Bent rectangular with positive perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, rect_height, K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_rect_K1p = make_mesh(surf, UU, VV)
sheet_rect_K1p.apply_transform(T_K1)
sheet_rect_K1p = center_xy(sheet_rect_K1p)

# Bent rectangular with negative perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, rect_height, -K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_rect_K1n = make_mesh(surf, UU, VV)
sheet_rect_K1n.apply_transform(T_K1)
sheet_rect_K1n = center_xy(sheet_rect_K1n)

# Circle
circle_radius = 20
circle_theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
circle_x = circle_radius * np.cos(circle_theta)
circle_y = circle_radius * np.sin(circle_theta)
circle_cp = np.vstack([np.zeros_like(circle_x), circle_x, circle_y]).T
mean_xyz = circle_cp.mean(axis=0)
circle_cp = circle_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(circle_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
cp[:, :, 2] -= cp[:, :, 2].min()  # Maybe a better way to do this
surf = make_surface(cp)
sheet_circle_K0 = make_mesh(surf, UU, VV)
sheet_circle_K0 = center_xy(sheet_circle_K0)

# Circle with positive perpendicular curvature
circle_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(0.0001, circle_height, 5)  # TODO: Fix this - veery ugly
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K0 = Backbone(b_cp, reparameterize=True)

bent_cp = bend_sheet(cp, b_appendage_K0, circle_height, K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_circle_K0p = make_mesh(surf, UU, VV)
sheet_circle_K0p = center_xy(sheet_circle_K0p)

circle_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(np.pi / 2, circle_height, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

# Bent circle
bent_cp = bend_sheet(cp, b_appendage_K1, circle_height, 0)
surf = make_surface(bent_cp)
sheet_circle_K1 = make_mesh(surf, UU, VV)
sheet_circle_K1.apply_transform(T_K1)
sheet_circle_K1 = center_xy(sheet_circle_K1)

# Bent circle with positive perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, circle_height, K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_circle_K1p = make_mesh(surf, UU, VV)
sheet_circle_K1p.apply_transform(T_K1)
sheet_circle_K1p = center_xy(sheet_circle_K1p)

# Bent circle with negative perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, circle_height, -K_PERPENDICULAR)
surf = make_surface(bent_cp)
sheet_circle_K1n = make_mesh(surf, UU, VV)
sheet_circle_K1n.apply_transform(T_K1)
sheet_circle_K1n = center_xy(sheet_circle_K1n)

# Small Ellipse
small_ellipse_length = 20
small_ellipse_x = np.linspace(0, 1, 3) * (small_ellipse_length)
small_ellipse_y = np.array([7.5, 7.5, 7.5])
small_ellipse_poly = np.polyfit(small_ellipse_x, small_ellipse_y, 2)
small_ellipse_cp = make_base_cp(small_ellipse_poly, small_ellipse_x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = small_ellipse_cp.mean(axis=0)
small_ellipse_cp = small_ellipse_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(small_ellipse_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
cp[:, :, 2] -= cp[:, :, 2].min()  # Maybe a better way to do this
surf = make_surface(cp)
sheet_small_ellipse_K0 = make_mesh(surf, UU, VV)
sheet_small_ellipse_K0 = center_xy(sheet_small_ellipse_K0)

# Small Ellipse with positive perpendicular curvature
small_ellipse_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(0.0001, small_ellipse_height, 5)  # TODO: Fix this - veery ugly
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K0 = Backbone(b_cp, reparameterize=True)

bent_cp = bend_sheet(cp, b_appendage_K0, small_ellipse_height, K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_ellipse_K0p = make_mesh(surf, UU, VV)
sheet_small_ellipse_K0p = center_xy(sheet_small_ellipse_K0p)

small_ellipse_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(np.pi / 2, small_ellipse_height, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

# Bent small ellipse
bent_cp = bend_sheet(cp, b_appendage_K1, small_ellipse_height, 0)
surf = make_surface(bent_cp)
sheet_small_ellipse_K1 = make_mesh(surf, UU, VV)
sheet_small_ellipse_K1.apply_transform(T_K1)
sheet_small_ellipse_K1 = center_xy(sheet_small_ellipse_K1)

# Bent small ellipse with positive perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, small_ellipse_height, K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_ellipse_K1p = make_mesh(surf, UU, VV)
sheet_small_ellipse_K1p.apply_transform(T_K1)
sheet_small_ellipse_K1p = center_xy(sheet_small_ellipse_K1p)

# Bent small ellipse with negative perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, small_ellipse_height, -K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_ellipse_K1n = make_mesh(surf, UU, VV)
sheet_small_ellipse_K1n.apply_transform(T_K1)
sheet_small_ellipse_K1n = center_xy(sheet_small_ellipse_K1n)


# Small teardrop
small_teardrop_length = 22
small_teardrop_x = np.linspace(0, 1, 3) * (small_teardrop_length)
small_teardrop_y = np.array([7.5, 5, 1])
small_teardrop_poly = np.polyfit(small_teardrop_x, small_teardrop_y, 2)
small_teardrop_cp = make_base_cp(small_teardrop_poly, small_teardrop_x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = small_teardrop_cp.mean(axis=0)
small_teardrop_cp = small_teardrop_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(small_teardrop_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
cp[:, :, 2] -= cp[:, :, 2].min()  # Maybe a better way to do this
surf = make_surface(cp)
sheet_small_teardrop_K0 = make_mesh(surf, UU, VV)

# Small teardrop with positive perpendicular curvature
small_teardrop_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(0.0001, small_teardrop_height, 5)  # TODO: Fix this - veery ugly
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K0 = Backbone(b_cp, reparameterize=True)

bent_cp = bend_sheet(cp, b_appendage_K0, small_teardrop_height, K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_teardrop_K0p = make_mesh(surf, UU, VV)

small_teardrop_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(np.pi / 2, small_teardrop_height, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

# Bent small teardrop
bent_cp = bend_sheet(cp, b_appendage_K1, small_teardrop_height, 0)
surf = make_surface(bent_cp)
sheet_small_teardrop_K1 = make_mesh(surf, UU, VV)
sheet_small_teardrop_K1.apply_transform(T_K1)

# Bent small teardrop with positive perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, small_teardrop_height, K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_teardrop_K1p = make_mesh(surf, UU, VV)
sheet_small_teardrop_K1p.apply_transform(T_K1)

# Bent small teardrop with negative perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, small_teardrop_height, -K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_teardrop_K1n = make_mesh(surf, UU, VV)
sheet_small_teardrop_K1n.apply_transform(T_K1)


# Small teardropf
sheet_small_teardropf_K0 = sheet_small_teardrop_K0.copy()
sheet_small_teardropf_K0.apply_transform(T_FLIP)
sheet_small_teardropf_K0.apply_translation([0, 0, -sheet_small_teardropf_K0.bounds[0, 2]])
sheet_small_teardropf_K0 = center_xy(sheet_small_teardropf_K0)
sheet_small_teardropf_K0p = sheet_small_teardrop_K0p.copy()
sheet_small_teardropf_K0p.apply_transform(T_FLIP)
sheet_small_teardropf_K0p.apply_translation([0, 0, -sheet_small_teardropf_K0p.bounds[0, 2]])
sheet_small_teardropf_K0p = center_xy(sheet_small_teardropf_K0p)
sheet_small_teardropf_K1 = sheet_small_teardrop_K1.copy()
sheet_small_teardropf_K1.apply_transform(T_FLIP)
sheet_small_teardropf_K1.apply_translation([0, 0, -sheet_small_teardropf_K1.bounds[0, 2]])
sheet_small_teardropf_K1 = center_xy(sheet_small_teardropf_K1)
sheet_small_teardropf_K1p = sheet_small_teardrop_K1p.copy()
sheet_small_teardropf_K1p.apply_transform(T_FLIP)
sheet_small_teardropf_K1p.apply_translation([0, 0, -sheet_small_teardropf_K1p.bounds[0, 2]])
sheet_small_teardropf_K1p = center_xy(sheet_small_teardropf_K1p)
sheet_small_teardropf_K1n = sheet_small_teardrop_K1n.copy()
sheet_small_teardropf_K1n.apply_transform(T_FLIP)
sheet_small_teardropf_K1n.apply_translation([0, 0, -sheet_small_teardropf_K1n.bounds[0, 2]])
sheet_small_teardropf_K1n = center_xy(sheet_small_teardropf_K1n)


# small rect
small_rect_cp = small_ellipse_cp.copy()
small_rect_cp[:, 2] /= 1.25
small_rect_cp[:top_round_cp, 2] = small_rect_cp[:, 2].min()
small_rect_cp[top_round_cp + num_edge_cp : top_round_cp + num_edge_cp + top_round_cp, 2] = small_rect_cp[:, 2].max()
small_rect_cp[top_round_cp : top_round_cp + num_edge_cp, 2] = np.linspace(
    small_rect_cp[:, 2].min(), small_rect_cp[:, 2].max(), num_edge_cp
)
small_rect_cp[top_round_cp + num_edge_cp + top_round_cp :, 2] = np.linspace(
    small_rect_cp[:, 2].max(), small_rect_cp[:, 2].min(), num_edge_cp
)
mean_xyz = small_rect_cp.mean(axis=0)
small_rect_cp = small_rect_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(small_rect_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
cp[:, :, 2] -= cp[:, :, 2].min()  # Maybe a better way to do this
surf = make_surface(cp)
sheet_small_rect_K0 = make_mesh(surf, UU, VV)
sheet_small_rect_K0 = center_xy(sheet_small_rect_K0)

# Small rect with positive perpendicular curvature
small_rect_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(0.0001, small_rect_height, 5)  # TODO: Fix this - veery ugly
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis

b_appendage_K0 = Backbone(b_cp, reparameterize=True)

bent_cp = bend_sheet(cp, b_appendage_K0, small_rect_height, K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_rect_K0p = make_mesh(surf, UU, VV)
sheet_small_rect_K0p = center_xy(sheet_small_rect_K0p)

small_rect_height = cp[:, :, 2].max() - cp[:, :, 2].min()
b_cp = approximate_arc(np.pi / 2, small_rect_height, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

# Bent small rect
bent_cp = bend_sheet(cp, b_appendage_K1, small_rect_height, 0)
surf = make_surface(bent_cp)
sheet_small_rect_K1 = make_mesh(surf, UU, VV)
sheet_small_rect_K1.apply_transform(T_K1)
sheet_small_rect_K1 = center_xy(sheet_small_rect_K1)

# Bent small rect with positive perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, small_rect_height, K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_rect_K1p = make_mesh(surf, UU, VV)
sheet_small_rect_K1p.apply_transform(T_K1)
sheet_small_rect_K1p = center_xy(sheet_small_rect_K1p)

# Bent small rect with negative perpendicular curvature
bent_cp = bend_sheet(cp, b_appendage_K1, small_rect_height, -K_PERPENDICULAR_SMALL)
surf = make_surface(bent_cp)
sheet_small_rect_K1n = make_mesh(surf, UU, VV)
sheet_small_rect_K1n.apply_transform(T_K1)
sheet_small_rect_K1n = center_xy(sheet_small_rect_K1n)


# Create dictionary
mesh_dict = {
    "cap": load_cap(),
    "sheet_ellipse_K0": sheet_ellipse_K0,
    "sheet_ellipse_K0p": sheet_ellipse_K0p,
    "sheet_ellipse_K1": sheet_ellipse_K1,
    "sheet_ellipse_K1p": sheet_ellipse_K1p,
    "sheet_ellipse_K1n": sheet_ellipse_K1n,
    "sheet_teardrop_K0": sheet_teardrop_K0,
    "sheet_teardrop_K0p": sheet_teardrop_K0p,
    "sheet_teardrop_K1": sheet_teardrop_K1,
    "sheet_teardrop_K1p": sheet_teardrop_K1p,
    "sheet_teardrop_K1n": sheet_teardrop_K1n,
    "sheet_teardropf_K0": sheet_teardropf_K0,
    "sheet_teardropf_K0p": sheet_teardropf_K0p,
    "sheet_teardropf_K1": sheet_teardropf_K1,
    "sheet_teardropf_K1p": sheet_teardropf_K1p,
    "sheet_teardropf_K1n": sheet_teardropf_K1n,
    "sheet_rect_K0": sheet_rect_K0,
    "sheet_rect_K0p": sheet_rect_K0p,
    "sheet_rect_K1": sheet_rect_K1,
    "sheet_rect_K1p": sheet_rect_K1p,
    "sheet_rect_K1n": sheet_rect_K1n,
    "sheet_circle_K0": sheet_circle_K0,
    "sheet_circle_K0p": sheet_circle_K0p,
    "sheet_circle_K1": sheet_circle_K1,
    "sheet_circle_K1p": sheet_circle_K1p,
    "sheet_circle_K1n": sheet_circle_K1n,
    "small_ellipse_K0": sheet_small_ellipse_K0,
    "small_ellipse_K0p": sheet_small_ellipse_K0p,
    "small_ellipse_K1": sheet_small_ellipse_K1,
    "small_ellipse_K1p": sheet_small_ellipse_K1p,
    "small_ellipse_K1n": sheet_small_ellipse_K1n,
    "small_teardrop_K0": sheet_small_teardrop_K0,
    "small_teardrop_K0p": sheet_small_teardrop_K0p,
    "small_teardrop_K1": sheet_small_teardrop_K1,
    "small_teardrop_K1p": sheet_small_teardrop_K1p,
    "small_teardrop_K1n": sheet_small_teardrop_K1n,
    "small_teardropf_K0": sheet_small_teardropf_K0,
    "small_teardropf_K0p": sheet_small_teardropf_K0p,
    "small_teardropf_K1": sheet_small_teardropf_K1,
    "small_teardropf_K1p": sheet_small_teardropf_K1p,
    "small_teardropf_K1n": sheet_small_teardropf_K1n,
    "small_rect_K0": sheet_small_rect_K0,
    "small_rect_K0p": sheet_small_rect_K0p,
    "small_rect_K1": sheet_small_rect_K1,
    "small_rect_K1p": sheet_small_rect_K1p,
    "small_rect_K1n": sheet_small_rect_K1n,
}

# create_scene(mesh_dict)
s_list = []

####################
### LARGE SHEETS ###
####################

large_sheets = [
    "sheet_ellipse_K0",
    "sheet_ellipse_K0p",
    "sheet_ellipse_K1",
    "sheet_ellipse_K1p",
    "sheet_ellipse_K1n",
    "sheet_teardrop_K0",
    "sheet_teardrop_K0p",
    "sheet_teardrop_K1",
    "sheet_teardrop_K1p",
    "sheet_teardrop_K1n",
    "sheet_teardropf_K0",
    "sheet_teardropf_K0p",
    "sheet_teardropf_K1",
    "sheet_teardropf_K1p",
    "sheet_teardropf_K1n",
    "sheet_rect_K0",
    "sheet_rect_K0p",
    "sheet_rect_K1",
    "sheet_rect_K1p",
    "sheet_rect_K1n",
]

for mesh_name in large_sheets:
    mesh_names = [mesh_name]
    centroid = mesh_dict[mesh_names[0]].centroid

    mesh_list = [mesh_dict[mesh_name].copy() for mesh_name in mesh_names]
    for m in mesh_list:
        m.apply_translation(-centroid)  # So that rotations occur about centroid

    th = 0
    T = rotvec2T(th, [0, 1, 0])
    T[2, 3] = centroid[2] - 5
    T_list = [T]

    op_list = ["union"] * len(mesh_list)
    description = "straight"
    label = "D006"

    # Add in cap
    mesh_list.append(mesh_dict["cap"])
    T = np.eye(4)
    T[2, 3] = -3
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

###################################
### SMALL SHEETS WITH ROTATIONS ###
###################################

small_sheets = [
    "sheet_circle_K0",
    "sheet_circle_K0p",
    "sheet_circle_K1",
    "sheet_circle_K1p",
    "sheet_circle_K1n",
    "small_ellipse_K0",
    "small_ellipse_K0p",
    "small_ellipse_K1",
    "small_ellipse_K1p",
    "small_ellipse_K1n",
    "small_teardrop_K0",
    "small_teardrop_K0p",
    "small_teardrop_K1",
    "small_teardrop_K1p",
    "small_teardrop_K1n",
    "small_teardropf_K0",
    "small_teardropf_K0p",
    "small_teardropf_K1",
    "small_teardropf_K1p",
    "small_teardropf_K1n",
    "small_rect_K0",
    "small_rect_K0p",
    "small_rect_K1",
    "small_rect_K1p",
    "small_rect_K1n",
]


for mesh_name in small_sheets:
    mesh_names = [mesh_name]
    centroid = mesh_dict[mesh_names[0]].centroid

    # Straight up
    for i in range(5):

        # Skip redundant thetas for flat
        if "K0" in mesh_name and "K0p" not in mesh_name and "K0n" not in mesh_name and i > 2:
            continue

        mesh_list = [mesh_dict[mesh_name].copy() for mesh_name in mesh_names]
        for m in mesh_list:
            m.apply_translation(-centroid)  # So that rotations occur about centroid

        th = np.linspace(-np.pi / 2, np.pi / 2, 5)[i]
        T = rotvec2T(th, [0, 1, 0])
        T[2, 3] = centroid[2] - 5
        T_list = [T]

        op_list = ["union"] * len(mesh_list)
        description = "straight"
        label = "D006"

        # Add in cap
        mesh_list.append(mesh_dict["cap"])
        T = np.eye(4)
        T[2, 3] = -3
        T_list.append(T)
        op_list.append("union")

        if i != 2:
            post = trimesh.creation.cylinder(radius=5.5, height=centroid[2] - 5 + 4, sections=7)
            post.apply_translation([0, 0, -post.bounds[0, 2] - 3.5])
            mesh_list.append(post)
            T_list.append(np.eye(4))
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

    for i in range(1, 3):

        # Skip circle because symmetric
        if "circle_K0" in mesh_name:
            continue

        mesh_list = [mesh_dict[mesh_name].copy() for mesh_name in mesh_names]
        for m in mesh_list:
            m.apply_translation(-centroid)  # So that rotations occur about centroid

        th = np.linspace(0, np.pi / 2, 3)[i]
        T = rotvec2T(th, [1, 0, 0])
        T[2, 3] = centroid[2] - 5
        T_list = [T]

        op_list = ["union"] * len(mesh_list)
        description = "straight"
        label = "D006"

        # Add in cap
        mesh_list.append(mesh_dict["cap"])
        T = np.eye(4)
        T[2, 3] = -3
        T_list.append(T)
        op_list.append("union")

        post = trimesh.creation.cylinder(radius=5.5, height=centroid[2] - 6, sections=7)
        post.apply_translation([0, 0, -post.bounds[0, 2] - 3.5])
        mesh_list.append(post)
        T_list.append(np.eye(4))
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


save_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet")
for i, s in enumerate(s_list):
    export_shape(s, save_dir, f"sheet_{str(i).zfill(3)}")
