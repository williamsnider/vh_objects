import trimesh
import numpy  as np


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

######################################
### Base Components and Appendages ###
######################################


# def slice_mesh(mesh, extent, T):
#     mesh = mesh.copy()
#     slicer = trimesh.primitives.Box(
#         extents=np.array([extent, extent, extent]), transform=T
#     )
#     split_mesh, _ = calc_mesh_boolean_and_edges(mesh, slicer, "difference")

#     return split_mesh


# thin = Shaft(
#     SEGMENT_LENGTH,
#     VOLUMETRIC_RADII[0],
#     VOLUMETRIC_RADII[0],
#     VOLUMETRIC_RADII[0],
#     theta=0,
#     lengthtype="two_hemi",
#     num_cs=NUM_CS,
#     num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
# )

# volumetric = Shaft(
#     SEGMENT_LENGTH,
#     VOLUMETRIC_RADII[0],
#     VOLUMETRIC_RADII[1],
#     VOLUMETRIC_RADII[2],
#     theta=0,
#     lengthtype="two_hemi",
#     num_cs=NUM_CS,
#     num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
# )

# # Transformation matrix so that shafts are pointing towards +Z axis
# T_point_z = np.eye(4)
# T_point_z[:3, :3] = Rotation.from_euler("xyz", np.array([0, -np.pi / 2, 0])).as_matrix()



# def plot_cp_and_backbone(cp, backbone):
#     import matplotlib.pyplot as plt

#     ax = plt.figure().add_subplot(projection="3d")
#     arr = cp
#     for i in range(arr.shape[0]):
#         ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")

#     # Plot backbone
#     t = np.linspace(0, 1, 100)
#     bx = backbone.r(t)[:, 0]
#     by = backbone.r(t)[:, 1]
#     bz = backbone.r(t)[:, 2]
#     ax.plot(bx, by, bz, "r-")

#     # Set scale
#     xs = np.concatenate([arr[:, :, 0].ravel(), bx.ravel()])
#     ys = np.concatenate([arr[:, :, 1].ravel(), by.ravel()])
#     zs = np.concatenate([arr[:, :, 2].ravel(), bz.ravel()])
#     ax.set_box_aspect(
#         (np.ptp(xs), np.ptp(ys), np.ptp(zs))
#     )  # aspect ratio is 1:1:1 in data space
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     plt.show()

# APP_ANGLE = np.pi/2

# def bend_app(app, app_length):
#     app_K1_b_length = app_length - app.rdist
#     app_K1_b_cp  = approximate_arc(APP_ANGLE, app_K1_b_length, 5)
#     app_K1_b_cp = app_K1_b_cp[:, [1, 2, 0]]  # Reorder
#     app_K1_b_cp[:, 0] *= -1  # Flip direction across yz axis
#     app_K1_b = Backbone(app_K1_b_cp, reparameterize=True)

#     cp = app.cp.copy()
#     cp[:,:,2] *= -1
#     bent_cp = bend_sheet(cp, app_K1_b, app_K1_b_length)
#     surf = make_surface(bent_cp)
#     app_K1 = make_mesh(surf, uu, vv)
#     app_K1.faces = app_K1.faces[:, [0, 2, 1]]  # Flip faces to fix winding
#     return app_K1



capsule_diameter = 5
capsule_length = 20

NUM_CS = 21
capsule_K0 = Shaft(capsule_length, 1.0 * capsule_diameter, 1.0 * capsule_diameter, 1.0*capsule_diameter, theta=0, lengthtype="one_hemi", num_cs=NUM_CS, num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,)

# TODO: Fix the capsule length
capsule_K1_length = capsule_length*1.7
capsule_K1 = Shaft(capsule_K1_length, 1.0 * capsule_diameter, 1.0 * capsule_diameter, 1.0*capsule_diameter, theta=np.pi/2, lengthtype="one_hemi", num_cs=NUM_CS, num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,)

# Tranform to be pointing upwards
T_point_z = np.eye(4)
T_point_z[:3, :3] = Rotation.from_euler("xyz", np.array([0, -np.pi / 2, 0])).as_matrix()
capsule_K0.mesh.apply_transform(T_point_z)
capsule_K1.mesh.apply_transform(T_point_z)


# Base cylinder
base_cylinder_radius = 20
base_cylinder_height = 0.5
base_cylinder = trimesh.creation.cylinder(radius=base_cylinder_radius, height=base_cylinder_height, sections=100)


scene = trimesh.Scene(base_cylinder)

# Claw stimulus
mesh_list = []
num_meshes = 3
for i in range(num_meshes):

    mesh = copy.deepcopy(capsule_K1.mesh)

    # Rotate about z-axis
    T = trimesh.transformations.rotation_matrix(np.linspace(0, 2*np.pi,3, endpoint=False)[i], [0, 0, 1])
    mesh.apply_transform(T)
    mesh.apply_scale(1+0.001*i)


    mesh_list.append(mesh)


s = Shape(mesh_list, [np.eye(4) for _ in range(num_meshes)], ["union"]*num_meshes, " ", "test","test", mesh_fairing_distance= 2)

scene.show()
