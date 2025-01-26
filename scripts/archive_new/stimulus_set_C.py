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
    calc_hemisphere_controlpoints,
    angle_between,
    calc_mesh_boolean_and_edges,
    angle_between,
)
from scripts.sheets_utilities import construct_sheet, bend_sheet, make_base_cp
import trimesh
from scipy.spatial.transform.rotation import Rotation
from objects.shaft import Shaft
from scripts.archive_new.stimulus_set_params import (
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
)
from scripts.stim_set_common import mesh_dict, volumetric, thin
from pathlib import Path

MESH_FAIRING_DISTANCE = 2
POST_FAIRING_DISTANCE = 2
POST_Z_SHIFT = 0
SAVE_DIR = Path(SAVE_DIR, "stimulus_set_C")




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
vec_to_J1 = np.array([0, np.sin(x_th), np.cos(x_th)])
vec_to_J1_orth = np.cross(vec_to_J1, vec_axial_component)
vec_to_J2 = np.array([0, np.sin(x_th), np.cos(x_th)])
vec_to_J2_orth = np.cross(vec_to_J2, vec_axial_component)

J1_volu_pos = volumetric.backbone.r(1 / 3)
J2_volu_pos = volumetric.backbone.r(2 / 3)
J1_thin_pos = volumetric.backbone.r(5/12)
J2_thin_pos = np.array([SEGMENT_LENGTH - X_WIDTH, 0, 0])
# J2_pos =
CO_xyz = np.array([SEGMENT_LENGTH - X_WIDTH / 2, 0, 0])

# Volumetric
J1_volu_xyz_U = (
    find_line_mesh_intersection(volumetric.mesh, vec_to_J1, J1_volu_pos)
    + XYZ_OFFSET * vec_to_J1
)
# J1_volu_xyz_D = (
#     find_line_mesh_intersection(volumetric.mesh, -vec_to_J1, J1_volu_pos)
#     + -XYZ_OFFSET * vec_to_J1
# )
J1_volu_xyz_D = J1_volu_xyz_U * np.array([1,1,-1])
J2_volu_xyz_U = J1_volu_xyz_U + np.array([2*(SEGMENT_LENGTH/2-J1_volu_xyz_U[0]), 0, 0])
J2_volu_xyz_D = J2_volu_xyz_U *np.array([1,1,-1])
# J2_volu_xyz_U = (
#     find_line_mesh_intersection(volumetric.mesh, vec_to_J2, J2_volu_pos)
#     + XYZ_OFFSET * vec_to_J2
# )
# J2_volu_xyz_D = (
#     find_line_mesh_intersection(volumetric.mesh, -vec_to_J2, J2_volu_pos)
#     + -XYZ_OFFSET * vec_to_J2
# )

# Thin
J1_thin_xyz_U = (
    find_line_mesh_intersection(thin.mesh, vec_to_J1, J1_thin_pos)
    + XYZ_OFFSET * vec_to_J1
)
# J1_thin_xyz_D = J1_thin_xyz_U * np.array([1,1,-1])
# J2_thin_xyz_U = J1_thin_xyz_U + np.array([2*(SEGMENT_LENGTH/2-J1_thin_xyz_U[0]), 0, 0])
# J2_thin_xyz_D = J2_thin_xyz_U * np.array([1,1,-1])
J1_thin_xyz_D = (
    find_line_mesh_intersection(thin.mesh, -vec_to_J1, J1_thin_pos)
    + -XYZ_OFFSET * vec_to_J1
)
J2_thin_xyz_U = (
    find_line_mesh_intersection(thin.mesh, vec_to_J2, J2_thin_pos)
    + XYZ_OFFSET * vec_to_J2
)
J2_thin_xyz_D = (
    find_line_mesh_intersection(thin.mesh, -vec_to_J2, J2_thin_pos)
    + -XYZ_OFFSET * vec_to_J2
)

# Volumetric rotations to match curvature of surface
vec1 = np.array([0, 0, 1])
vec2 = volumetric.mesh.vertex_normals[
    (volumetric.mesh.vertices == J1_volu_xyz_U).all(axis=1).argmax()
]
angle = angle_between(vec1, vec2)
y_th = angle  # np.pi / 4  # np.pi / 9

# Rotate volumetric J1 appendages to be in line with slope of volume (y_th below)
R_volu_J1_F_U = Rotation.from_euler(
    "zyx", np.array([0 * np.pi / 2, -y_th, -x_th])
).as_matrix()
R_volu_J1_L_U = Rotation.from_euler(
    "zyx", np.array([1 * np.pi / 2, -y_th, -x_th])
).as_matrix()
R_volu_J1_B_U = Rotation.from_euler(
    "zyx", np.array([2 * np.pi / 2, -y_th, -x_th])
).as_matrix()
R_volu_J1_R_U = Rotation.from_euler(
    "zyx", np.array([3 * np.pi / 2, -y_th, -x_th])
).as_matrix()
R_volu_J1_F_D = Rotation.from_euler(
    "zyx", np.array([0 * np.pi / 2, -y_th, -x_th + np.pi])
).as_matrix()
R_volu_J1_L_D = Rotation.from_euler(
    "zyx", np.array([3 * np.pi / 2, -y_th, -x_th + np.pi])
).as_matrix()
R_volu_J1_B_D = Rotation.from_euler(
    "zyx", np.array([2 * np.pi / 2, -y_th, -x_th + np.pi])
).as_matrix()
R_volu_J1_R_D = Rotation.from_euler(
    "zyx", np.array([1 * np.pi / 2, -y_th, -x_th + np.pi])
).as_matrix()
R_volu_J2_F_U = Rotation.from_euler(
    "zyx", np.array([0 * np.pi / 2, y_th, -x_th])
).as_matrix()
R_volu_J2_L_U = Rotation.from_euler(
    "zyx", np.array([1 * np.pi / 2, y_th, -x_th])
).as_matrix()
R_volu_J2_B_U = Rotation.from_euler(
    "zyx", np.array([2 * np.pi / 2, y_th, -x_th])
).as_matrix()
R_volu_J2_R_U = Rotation.from_euler(
    "zyx", np.array([3 * np.pi / 2, y_th, -x_th])
).as_matrix()
R_volu_J2_F_D = Rotation.from_euler(
    "zyx", np.array([0 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_volu_J2_L_D = Rotation.from_euler(
    "zyx", np.array([3 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_volu_J2_B_D = Rotation.from_euler(
    "zyx", np.array([2 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()
R_volu_J2_R_D = Rotation.from_euler(
    "zyx", np.array([1 * np.pi / 2, y_th, -x_th + np.pi])
).as_matrix()

R_thin_F_U = Rotation.from_euler("zyx", np.array([0 * np.pi / 2, 0, -x_th])).as_matrix()
R_thin_L_U = Rotation.from_euler("zyx", np.array([1 * np.pi / 2, 0, -x_th])).as_matrix()
R_thin_B_U = Rotation.from_euler("zyx", np.array([2 * np.pi / 2, 0, -x_th])).as_matrix()
R_thin_R_U = Rotation.from_euler("zyx", np.array([3 * np.pi / 2, 0, -x_th])).as_matrix()

R_thin_F_D = Rotation.from_euler(
    "zyx", np.array([0 * np.pi / 2, 0, -x_th + np.pi])
).as_matrix()
R_thin_L_D = Rotation.from_euler(
    "zyx", np.array([3 * np.pi / 2, 0, -x_th + np.pi])
).as_matrix()
R_thin_B_D = Rotation.from_euler(
    "zyx", np.array([2 * np.pi / 2, 0, -x_th + np.pi])
).as_matrix()
R_thin_R_D = Rotation.from_euler(
    "zyx", np.array([1 * np.pi / 2, 0, -x_th + np.pi])
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


# R_F = Rotation.from_euler("xyz", [0, angle, 0]).as_matrix()
# R_R = Rotation.from_euler("xyz", [0, -angle, 0]).as_matrix()


def calc_T(R, xyz):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


# J1, direction of curvature (forward, left, right, back), up/down
T_volu_J1_B_U = calc_T(R_volu_J1_B_U, J1_volu_xyz_U)
T_volu_J1_B_D = calc_T(R_volu_J1_B_D, J1_volu_xyz_D)
T_volu_J1_L_U = calc_T(R_volu_J1_L_U, J1_volu_xyz_U)
T_volu_J1_L_D = calc_T(R_volu_J1_L_D, J1_volu_xyz_D)
T_volu_J1_F_U = calc_T(R_volu_J1_F_U, J1_volu_xyz_U)
T_volu_J1_F_D = calc_T(R_volu_J1_F_D, J1_volu_xyz_D)
T_volu_J1_R_U = calc_T(R_volu_J1_R_U, J1_volu_xyz_U)
T_volu_J1_R_D = calc_T(R_volu_J1_R_D, J1_volu_xyz_D)

# J2
T_volu_J2_B_U = calc_T(R_volu_J2_B_U, J2_volu_xyz_U)
T_volu_J2_B_D = calc_T(R_volu_J2_B_D, J2_volu_xyz_D)
T_volu_J2_L_U = calc_T(R_volu_J2_L_U, J2_volu_xyz_U)
T_volu_J2_L_D = calc_T(R_volu_J2_L_D, J2_volu_xyz_D)
T_volu_J2_F_U = calc_T(R_volu_J2_F_U, J2_volu_xyz_U)
T_volu_J2_F_D = calc_T(R_volu_J2_F_D, J2_volu_xyz_D)
T_volu_J2_R_U = calc_T(R_volu_J2_R_U, J2_volu_xyz_U)
T_volu_J2_R_D = calc_T(R_volu_J2_R_D, J2_volu_xyz_D)

# J1, direction of curvature (forward, left, right, back), up/down
T_thin_J1_B_U = calc_T(R_thin_B_U, J1_thin_xyz_U)
T_thin_J1_B_D = calc_T(R_thin_B_D, J1_thin_xyz_D)
T_thin_J1_L_U = calc_T(R_thin_L_U, J1_thin_xyz_U)
T_thin_J1_L_D = calc_T(R_thin_L_D, J1_thin_xyz_D)
T_thin_J1_F_U = calc_T(R_thin_F_U, J1_thin_xyz_U)
T_thin_J1_F_D = calc_T(R_thin_F_D, J1_thin_xyz_D)
T_thin_J1_R_U = calc_T(R_thin_R_U, J1_thin_xyz_U)
T_thin_J1_R_D = calc_T(R_thin_R_D, J1_thin_xyz_D)

# J2
T_thin_J2_B_U = calc_T(R_thin_B_U, J2_thin_xyz_U)
T_thin_J2_B_D = calc_T(R_thin_B_D, J2_thin_xyz_D)
T_thin_J2_L_U = calc_T(R_thin_L_U, J2_thin_xyz_U)
T_thin_J2_L_D = calc_T(R_thin_L_D, J2_thin_xyz_D)
T_thin_J2_F_U = calc_T(R_thin_F_U, J2_thin_xyz_U)
T_thin_J2_F_D = calc_T(R_thin_F_D, J2_thin_xyz_D)
T_thin_J2_R_U = calc_T(R_thin_R_U, J2_thin_xyz_U)
T_thin_J2_R_D = calc_T(R_thin_R_D, J2_thin_xyz_D)


# Collinear
T_CO_U = calc_T(R_U, CO_xyz)
T_CO_L = calc_T(R_L, CO_xyz)
T_CO_D = calc_T(R_D, CO_xyz)
T_CO_R = calc_T(R_R, CO_xyz)

# T_final shift x to align with post
T_final = np.eye(4)
T_final[0,3] = -X_WIDTH
combs = []


T_dict = {
    "T_eye": np.eye(4),
    "T_final": T_final,
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
    label = "C" + str(inputs[3].zfill(3))

    description = inputs[4]
    save_dir = inputs[5]
    T_final = T_dict[inputs[6]]
    mesh_fairing_distance = inputs[7]
    post_fairing_distance = inputs[8]
    post_z_shift = inputs[9]
    fair_box = inputs[10]


    s = Shape(
        mesh_list = mesh_list,
        T_list = T_list,
        boolean_list = boolean_list,
        label = label,
        description = description,
        save_dir = save_dir,
        T_final = T_final,
        mesh_fairing_distance = mesh_fairing_distance,
        post_fairing_distance = post_fairing_distance, 
        post_z_shift = post_z_shift,
        fair_box = fair_box,
    )



    # scene = trimesh.Scene()
    # box = trimesh.primitives.creation.box(np.array([5, 50, 50]))
    # box.apply_translation([SEGMENT_LENGTH / 2, 0, 0])
    # scene.add_geometry([s.mesh_with_interface, box])
    # scene.show()
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
    "T_final",
    MESH_FAIRING_DISTANCE,
    POST_FAIRING_DISTANCE,
    POST_Z_SHIFT,
    None,
]
combs.append(comb)
count += 1

# J1 and J2 and Collinear
for J_app in ["app7", "app8", "app9", "app10"]:
    for CO_app in ["app7", "app8", "app9", "app10"]:

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
                if withCO == False and CO_app != "app7":
                    continue

                boolean_list = ["union" for _ in mesh_list]

                # Add fairing box to remove bumps
                if withJ2 == True and withCO == False:
                    box = trimesh.primitives.creation.box(extents=BOX_EXTENTS)
                    box.apply_translation(BOX_TRANSLATION)
                else:
                    box = None

                # Assign combination of inputs
                comb = [
                    mesh_list,
                    T_list,
                    boolean_list,
                    str(count),
                    "",
                    SAVE_DIR,
                    "T_final",
                    MESH_FAIRING_DISTANCE,
                    POST_FAIRING_DISTANCE,
                    POST_Z_SHIFT,
                    box,
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
    "T_final",
    MESH_FAIRING_DISTANCE,
    POST_FAIRING_DISTANCE,
    POST_Z_SHIFT,
    None,
]
combs.append(comb)
count += 1

# J1 and J2 and Collinear
for J_app in [
    "app7",
    "app8",
    "app9",
    "app10",
    "app_point_convex",
    "app_point_concave",
    "app_round_convex",
    "app_round_concave",
]:
    for CO_app in ["app7", "app8", "app9", "app10"]:

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
                if withCO == False and CO_app != "app7":
                    continue

                # # Add fairing box to remove bumps
                # if withJ2 == True and withCO == False:
                #     box = trimesh.primitives.creation.box(extents=BOX_EXTENTS)
                #     box.apply_translation(BOX_TRANSLATION)
                # else:
                #     box = None
                box = None

                # Assign combination of inputs
                comb = [
                    mesh_list,
                    T_list,
                    boolean_list,
                    str(count),
                    "",
                    SAVE_DIR,
                    "T_final",
                    MESH_FAIRING_DISTANCE,
                    POST_FAIRING_DISTANCE,
                    POST_Z_SHIFT,
                    box,
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

                        # Add fairing box to remove bumps
                        # if withJ2 == True and withCO == False:
                        #     box = trimesh.primitives.creation.box(extents=BOX_EXTENTS)
                        #     box.apply_translation(BOX_TRANSLATION)
                        # else:
                        #     box = None
                        box = None

                        # Assign combination of inputs
                        comb = [
                            mesh_list,
                            T_list,
                            boolean_list,
                            str(count),
                            "",
                            SAVE_DIR,
                            "T_final",
                            MESH_FAIRING_DISTANCE,
                            POST_FAIRING_DISTANCE,
                            POST_Z_SHIFT,
                            box,
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

                        # # Add fairing box to remove bumps
                        # if withJ2 == True and withCO == False:
                        #     box = trimesh.primitives.creation.box(extents=BOX_EXTENTS)
                        #     box.apply_translation(BOX_TRANSLATION)
                        # else:
                        #     box = None
                        box = None

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
                            "T_final",
                            MESH_FAIRING_DISTANCE,
                            POST_FAIRING_DISTANCE,
                            POST_Z_SHIFT,
                            box,
                        ]

                        combs.append(comb)
                        count += 1

########################
### Construct Shapes ###
########################


# for comb in combs[617:618]:
#     build_shape(comb)

if __name__ == "__main__":

    # with Pool() as pool:
    #     mapped_values = list(
    #         tqdm(pool.imap_unordered(build_shape, combs), total=len(combs))
    #     )
    for c in combs:
        build_shape(c)