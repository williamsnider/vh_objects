# Torsos
from vh_objects.shaft import Shaft
import trimesh
import numpy as np
from scripts.stim_set_common import create_scene, load_cap, UU, VV, export_shape, slightly_deform_mesh
from scipy.spatial.transform import Rotation
from scripts.sheets import make_surface, make_mesh, construct_sheet
from vh_objects.shape import Shape
from vh_objects.utilities import calc_mesh_boolean_and_edges, fuse_meshes, fair_mesh
from trimesh.transformations import rotation_matrix as rotvec2T
from scripts.make_gif import calc_dist_from_z_axis
from scripts.subdivide_box import load_subdivided_box
import copy
from pathlib import Path


def add_rotations_about_Z(num_rotations, T_base, mesh_dict, sf_type, mesh_list, T_list, op_list):
    for i in range(num_rotations):
        ang = np.linspace(0, 2 * np.pi, num_rotations, endpoint=False)[i]
        T = rotvec2T(ang, [0, 0, 1]) @ T_base
        mesh_list.append(mesh_dict[sf_type].copy())
        T_list.append(T)
        if "difference" in sf_type:
            op_list.append("difference")
        else:
            op_list.append("union")
    return mesh_list, T_list, op_list


# def calc_mohawk_T(
#     T_left_base, vec_left, closest_point, closest_point_normal
# ):  # Angle between vector and surface normal
#     angle = np.arccos(np.dot(vec_left, closest_point_normal))

#     # Flip angle if normal is downward
#     assert np.all(np.isclose(vec_left, np.array([0, 1, 0])))
#     if closest_point_normal[2] < 0:
#         angle = -angle

#     new_T = T_left_base @ rotvec2T(-angle, [0, 1, 0])
#     new_T[:3, 3] = closest_point
#     return new_T


# Tranform to be pointing upwards
T_point_z = np.eye(4)
T_point_z[:3, :3] = Rotation.from_euler("xyz", np.array([0, -np.pi / 2, 0])).as_matrix()
T_point_z[2, 3] = -10

TZ90 = rotvec2T(np.pi / 2, [0, 0, 1])

NUM_CS = 11
NUM_CP_PER_CROSS_SECTION = 50
# Inputs
torso_length = 40
torso_radius = 12
K1_theta = np.pi / 4
football_r1 = 0.5 * torso_radius
football_r2 = 1.25 * torso_radius
football_r3 = 0.5 * torso_radius
cylinder_r1 = torso_radius
cylinder_r2 = torso_radius
cylinder_r3 = torso_radius
dumbbell_r1 = torso_radius
dumbbell_r2 = 1.0 * torso_radius
dumbbell_r3 = torso_radius
mesh_fairing_distance = 1

# football K0
torso_football_K0 = Shaft(
    torso_length,
    football_r1,
    football_r2,
    football_r3,
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_football_K0.mesh.apply_transform(T_point_z)
torso_football_K0.mesh.apply_translation([0, 0, -torso_football_K0.mesh.bounds[0, 2]])

# football K1
torso_football_K1 = Shaft(
    torso_length,
    football_r1,
    football_r2,
    football_r3,
    theta=K1_theta,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_football_K1.mesh.apply_transform(T_point_z)
torso_football_K1.mesh.apply_transform(TZ90)
torso_football_K1.mesh.apply_translation([0, 0, -torso_football_K1.mesh.bounds[0, 2]])

# cylinder K0
torso_cylinder_K0 = Shaft(
    torso_length,
    cylinder_r1,
    cylinder_r2,
    cylinder_r3,
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_cylinder_K0.mesh.apply_transform(T_point_z)
torso_cylinder_K0.mesh.apply_translation([0, 0, -torso_cylinder_K0.mesh.bounds[0, 2]])

# cylinder K1
torso_cylinder_K1 = Shaft(
    torso_length,
    cylinder_r1,
    cylinder_r2,
    cylinder_r3,
    theta=K1_theta,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_cylinder_K1.mesh.apply_transform(T_point_z)
torso_cylinder_K1.mesh.apply_transform(TZ90)
torso_cylinder_K1.mesh.apply_translation([0, 0, -torso_cylinder_K1.mesh.bounds[0, 2]])

# # dumbbell K0
# torso_dumbbell_K0 = Shaft(
#     torso_length,
#     dumbbell_r1,
#     dumbbell_r2,
#     dumbbell_r3,
#     theta=0,
#     lengthtype="two_hemi",
#     num_cs=NUM_CS,
#     num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
# )
# torso_dumbbell_K0.mesh.apply_transform(T_point_z)
# torso_dumbbell_K0.mesh.apply_translation([0, 0, -torso_dumbbell_K0.mesh.bounds[0, 2]])

# # dumbbell K1
# torso_dumbbell_K1 = Shaft(
#     1 * torso_length,
#     dumbbell_r1,
#     dumbbell_r2,
#     dumbbell_r3,
#     theta=K1_theta,
#     lengthtype="two_hemi",
#     num_cs=NUM_CS,
#     num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
# )
# torso_dumbbell_K1.mesh.apply_transform(T_point_z)
# torso_dumbbell_K1.mesh.apply_transform(TZ90)
# torso_dumbbell_K1.mesh.apply_translation([0, 0, -torso_dumbbell_K1.mesh.bounds[0, 2]])

# dumbbell K0
sphere1 = trimesh.creation.icosphere(subdivisions=5, radius=torso_radius)
sphere1.apply_translation([0, 0, -sphere1.bounds[0, 2]])  # align bottom to origin
sphere2 = sphere1.copy()
sphere2.apply_translation([0, 0, torso_length - 2 * torso_radius])  # align top_to_origin
torso_dumbbell_K0 = fuse_meshes(sphere1, sphere2, 2, "union")
# torso_dumbbell_K0.show(smooth=False)


########################
### Surface Features ###
########################

# inputs
sf_radius = 5
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
    0.75 * sf_radius,
    0.5 * sf_radius,
    sf_radius_termination,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
sf_point.mesh.apply_transform(T_point_z)
sf_point.mesh.visual.face_colors = [0, 0, 255, 255]
# Shift up so that the point is sf_radius above the origin
sf_point.mesh.apply_translation([0, 0, -sf_point.mesh.bounds[1, 2] + sf_radius])
sf_point.mesh.apply_transform(rotvec2T(np.pi / 2, [0, 1, 0]))


# Ridge - circular sheet (disk)
circle_radius = sf_radius
circle_thickness = 3
circle_theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
circle_x = circle_radius * np.cos(circle_theta)
circle_y = circle_radius * np.sin(circle_theta)
circle_cp = np.vstack([np.zeros_like(circle_x), circle_x, circle_y]).T
mean_xyz = circle_cp.mean(axis=0)
circle_cp = circle_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(circle_cp, sheet_thickness=circle_thickness, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
cp[:, :, 2] -= cp[:, :, 2].min()  # Maybe a better way to do this
surf = make_surface(cp)
sf_ridge_vert_union = make_mesh(surf, UU, VV)
sf_ridge_vert_union.apply_translation(sf_ridge_vert_union.centroid * -1)
sf_ridge_vert_union.apply_transform(rotvec2T(np.pi / 2, [0, 0, 1]))
sf_ridge_vert_union.visual.face_colors = color_union

# Valley
sf_ridge_vert_difference = sf_ridge_vert_union.copy()
sf_ridge_vert_difference.visual.face_colors = color_difference

# Ridge horizontal
sf_ridge_hori_union = sf_ridge_vert_union.copy()
sf_ridge_hori_union.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

sf_ridge_hori_difference = sf_ridge_hori_union.copy()
sf_ridge_hori_difference.visual.face_colors = color_difference


# Sphere
sf_sphere_union = trimesh.creation.icosphere(subdivisions=5, radius=sf_radius)
sf_sphere_union.visual.face_colors = color_union

sf_sphere_difference = sf_sphere_union.copy()
sf_sphere_difference.visual.face_colors = color_difference


# Size meshes

size_ico = trimesh.creation.icosphere(subdivisions=5, radius=22.5)
box_edge = 27
# size_cube = load_subdivided_box([box_edge, box_edge, box_edge], 10)
size_cube = trimesh.creation.box([box_edge, box_edge, box_edge])
size_cubeR = size_cube.copy()
diagonal_vector = np.array([1, 1, 1]) / np.sqrt(3)
target_vector = np.array([0, 0, 1])
rotation_matrix = trimesh.geometry.align_vectors(diagonal_vector, target_vector)
size_cubeR.apply_transform(rotation_matrix)
size_cubeR.apply_translation(-size_cubeR.centroid)

mesh_dict = {
    "cap": load_cap(),
    "torso_football_K0": torso_football_K0.mesh,
    # "torso_football_K1": torso_football_K1.mesh,
    "torso_cylinder_K0": torso_cylinder_K0.mesh,
    # "torso_cylinder_K1": torso_cylinder_K1.mesh,
    "torso_dumbbell_K0": torso_dumbbell_K0,
    # "torso_dumbbell_K1": torso_dumbbell_K1.mesh,
    "sf_point": sf_point.mesh,
    "sf_ridge_vert_union": sf_ridge_vert_union,
    "sf_ridge_vert_difference": sf_ridge_vert_difference,
    "sf_ridge_hori_union": sf_ridge_hori_union,
    "sf_ridge_hori_difference": sf_ridge_hori_difference,
    "sf_sphere_union": sf_sphere_union,
    "sf_sphere_difference": sf_sphere_difference,
    "size_ico": size_ico,
    "size_cube": size_cube,
    "size_cubeR": size_cubeR,
}

# create_scene(mesh_dict)


def add_cap(mesh_list, T_list, op_list):
    # Add in cap
    mesh_list.append(mesh_dict["cap"].copy())
    T = np.eye(4)
    # T[2, 3] = -3
    T_list.append(T)
    op_list.append("union")
    return mesh_list, T_list, op_list


def distance_points_to_line(points, vec, origin):
    # Normalize the direction vector
    vec = vec / np.linalg.norm(vec)

    # Calculate the vector from the origin to each point
    vec_to_points = points - origin

    # Compute the cross product between the direction vector and the vectors to the points
    cross_products = np.cross(vec_to_points, vec)

    # Compute the distance as the norm of the cross products divided by the norm of the direction vector
    distances = np.linalg.norm(cross_products, axis=1)

    return distances


def find_mesh_point_closest_to_vec(mesh, vec, vec_origin):

    # Find the point on the mesh that is closest to the vector
    # vec_origin is the origin of the vector
    # vec is the vector
    # mesh is the mesh

    # Get the vertices
    vertices = mesh.vertices

    # Get the origin
    origin = vec_origin

    # Get the direction
    direction = vec

    # Get the point on the mesh that is closest to the vector
    distances = distance_points_to_line(vertices, direction, origin)
    cos_thetas = np.dot((vertices - origin) / np.linalg.norm(vertices - origin, axis=1, keepdims=True), direction)
    distances[cos_thetas < 0] = np.inf
    index = np.argmin(distances)
    closest_point = vertices[index]

    closest_point_normal = mesh.face_normals[mesh.vertex_faces[index][0]]

    # # Get the vector from the origin to the closest point
    # closest_vector = closest_point - origin

    # # Get the dot product of the direction and the closest vector
    # dot_product = np.dot(direction, closest_vector)

    # # Get the point on the mesh that is closest to the vector
    # closest_point = origin + dot_product * direction

    return closest_point, closest_point_normal


#######################
### Transformations ###
#######################

# T_left_base = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
# T_left_base[:3, 3] = [torso_radius, 0, torso_length - torso_radius]


s_list = []

# ###################
# ### Base Torsos ###
# ###################

base_torsos = [
    "torso_cylinder_K0",
    # "torso_cylinder_K1",
    "torso_football_K0",
    # "torso_football_K1",
    "torso_dumbbell_K0",
    # "torso_dumbbell_K1",
]

list_for_rotation = []
for mesh_name in base_torsos:

    mesh_list = [mesh_dict[mesh_name].copy()]
    T_list = [np.eye(4)]
    op_list = ["union"]

    # Store dumbbell so it can be rotated below
    list_for_rotation.append(copy.deepcopy([mesh_list, T_list, op_list, torso_length / 2]))

    mesh_list, T_list, op_list = add_cap(mesh_list, T_list, op_list)

    label = ""
    description = ""

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

sf_list = [
    "sf_point",
    "sf_ridge_vert_union",
    "sf_ridge_hori_union",
    "sf_sphere_union",
    "sf_ridge_vert_difference",
    "sf_ridge_hori_difference",
    "sf_sphere_difference",
]


def calc_T_given_mesh_vec_T_base(mesh, vec, vec_origin):
    closest_point, closest_point_normal = find_mesh_point_closest_to_vec(mesh, vec, vec_origin)
    # Calc T_Surface given closest_point, closest_point_normal
    TT = closest_point_normal
    Xaxis = np.array([1, 0, 0])
    TB = np.cross(TT, Xaxis)
    TB = TB / np.linalg.norm(TB)
    TN = np.cross(TB, TT)
    TN = TN / np.linalg.norm(TN)
    assert np.abs(np.dot(TT, TN)) < 1e-8
    T = np.eye(4)
    T[:3, 0] = TT
    T[:3, 1] = TN
    T[:3, 2] = TB
    T[:3, 3] = closest_point

    return T


##################
### Numerosity ###
##################

vec_left = np.array([0, 1, 0])
vec_right = -vec_left
front_origin = np.array([0, 0, torso_length - torso_radius])
back_origin = np.array([0, 0, torso_radius])
# vec_left_origin = np.array([0, 0, T_left_base[2, 3]])
# vec_head = np.array([0, 0, 1])
# vec_head_origin = np.array([0, 0, T_left_base[2, 3]])
# vec_right_origin = np.array([0, 0, T_left_base[2, 3]])

count = 0
for base_torso in [
    "torso_football_K0",
    "torso_cylinder_K0",
    "torso_dumbbell_K0",
]:

    # Calculate the frontright and backright transformation matrices
    T_frontright = calc_T_given_mesh_vec_T_base(mesh_dict[base_torso], vec_right, front_origin)
    T_backright = calc_T_given_mesh_vec_T_base(mesh_dict[base_torso], vec_right, back_origin)

    for sf_type in sf_list:

        for sf_locations in [
            "single",
            "double",
            "quad",
            "octo",
        ]:

            mesh_list = [mesh_dict[base_torso].copy()]
            T_list = [np.eye(4)]
            op_list = ["union"]

            if "single" in sf_locations:
                # Single sf at frontright
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    1, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )

            elif "double" in sf_locations:
                # Double sf at frontright, frontleft
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    2, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )
            elif "quad" in sf_locations:
                # Quad sf at frontright, frontleft, backright, backleft
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    2, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )

                mesh_list, T_list, op_list = add_rotations_about_Z(
                    2, T_backright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )

            elif "octo" in sf_locations:
                # Quad sf at frontright, frontleft, backright, backleft
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    4, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )

                mesh_list, T_list, op_list = add_rotations_about_Z(
                    4, T_backright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )
            else:
                raise ValueError("Invalid sf_locations")

            #     # Rotate about Z-axis
            #     num_rotations = 4
            #     for i in range(1, num_rotations):
            #         ang = np.linspace(0, 2 * np.pi, num_rotations, endpoint=False)[i]
            #         T = rotvec2T(ang, [0, 0, 1]) @ T_frontright
            #         mesh_list.append(mesh_dict[sf_type])
            #         T_list.append(T)
            #         if "difference" in sf_type:
            #             op_list.append("difference")
            #         else:
            #             op_list.append("union")

            # if "frontright" in sf_locations:

            #     mesh_list.append(mesh_dict[sf_type])

            #     print(closest_point, closest_point_normal)

            #     # Solve for T: T@ T_prev = T_goal

            #     # T_right = calc_mohawk_T(T_left_base, vec_left, closest_point, closest_point_normal)
            #     T_list.append(T)

            #     if "difference" in sf_type:
            #         op_list.append("difference")
            #     else:
            #         op_list.append("union")

            # if "head" in sf_locations:

            #     mesh_list.append(mesh_dict[sf_type])

            #     closest_point, closest_point_normal = find_mesh_point_closest_to_vec(
            #
            #     # Rotate about Z-axis
            #     num_rotations = 4
            #     for i in range(1, num_rotations):
            #         ang = np.linspace(0, 2 * np.pi, num_rotations, endpoint=False)[i]
            #         T = rotvec2T(ang, [0, 0, 1]) @ T_frontright
            #         mesh_list.append(mesh_dict[sf_type])
            #         T_list.append(T)
            #         if "difference" in sf_type:
            #             op_list.append("difference")
            #         else:
            #             op_list.append("union")

            # if "frontright" in sf_locations:

            #     mesh_list.append(mesh_dict[sf_type])

            #     print(closest_point, closest_point_normal)

            #     # Solve for T: T@ T_prev = T_goal

            #     # T_right = calc_mohawk_T(T_left_base, vec_left, closest_point, closest_point_normal)
            #     T_list.append(T)

            #     if "difference" in sf_type:
            #         op_list.append("difference")
            #     else:
            #         op_list.append("union")mesh_dict[base_torso], vec_head, vec_head_origin
            #     )
            #     T_head = calc_mohawk_T(T_left_base, vec_left, closest_point, closest_point_normal)
            #     T_list.append(T_head)

            #     if "difference" in sf_type:
            #         op_list.append("difference")
            #     else:
            #         op_list.append("union")

            # if "left" in sf_locations:

            #     mesh_list.append(mesh_dict[sf_type])

            #     closest_point, closest_point_normal = find_mesh_point_closest_to_vec(
            #         mesh_dict[base_torso], vec_left, vec_left_origin
            #     )
            #     T_left = calc_mohawk_T(T_left_base, vec_left, closest_point, closest_point_normal)
            #     T_list.append(T_left)

            #     if "difference" in sf_type:
            #         op_list.append("difference")
            #     else:
            #         op_list.append("union")

            # if "right" in sf_locations:

            #     mesh_list.append(mesh_dict[sf_type])

            #     closest_point, closest_point_normal = find_mesh_point_closest_to_vec(
            #         mesh_dict[base_torso], vec_right, vec_right_origin
            #     )
            #     T_right = calc_mohawk_T(T_left_base, vec_left, closest_point, closest_point_normal)
            #     T_list.append(T_right)

            #     if "difference" in sf_type:
            #         op_list.append("difference")
            #     else:
            #         op_list.append("union")

            # if sf_locations == "radial":

            #     closest_point, closest_point_normal = find_mesh_point_closest_to_vec(
            #         mesh_dict[base_torso], vec_left, vec_left_origin
            #     )
            #     T_base = T_left_base.copy()
            #     T_base[:3, 3] = closest_point

            #     for i in range(4):

            #         th = np.linspace(0, 2 * np.pi, 4, endpoint=False)[i]
            #         T = rotvec2T(th, [0, 0, 1]) @ T_base
            #         T_list.append(T)
            #         mesh_list.append(mesh_dict[sf_type])

            #     for i in range(4):
            #         if "difference" in sf_type:
            #             op_list.append("difference")
            #         else:
            #             op_list.append("union")

            # elif sf_locations == "mohawk":

            #     num_features = 4
            #     for i in range(num_features):

            #         th_pos = np.linspace(0, np.pi, num_features, endpoint=True)[i]
            #         T = rotvec2T(-th_pos, [0, 1, 0])
            #         new_vec = T[:3, :3] @ vec_left

            #         closest_point, closest_point_normal = find_mesh_point_closest_to_vec(
            #             mesh_dict[base_torso], new_vec, vec_left_origin
            #         )

            #         new_T = calc_mohawk_T(T_left_base, vec_left, closest_point, closest_point_normal)

            #         T_list.append(new_T)
            #         mesh_list.append(mesh_dict[sf_type])

            #     for i in range(num_features):
            #         if "difference" in sf_type:
            #             op_list.append("difference")
            #         else:
            #             op_list.append("union")

            mesh_list = slightly_deform_mesh(mesh_list)

            # Store dumbbell so it can be rotated below
            if base_torso == "torso_dumbbell_K0":
                list_for_rotation.append(copy.deepcopy([mesh_list, T_list, op_list, torso_length / 2]))

            mesh_list, T_list, op_list = add_cap(mesh_list, T_list, op_list)

            label = ""
            description = ""

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
            print(count)
            count += 1
            # s.mesh.show(smooth=False)


# ########################################
# ### Three but now in different plane ###
# ########################################
# for mesh_list, T_list, op_list, T_UP_SHIFT in list_for_rotation:

#     # Shift all down
#     for i in range(len(T_list)):
#         T_list[i][2, 3] -= T_UP_SHIFT

#     for i in range(len(T_list)):
#         T_list[i] = rotvec2T(-np.pi / 2, [1, 0, 0]) @ T_list[i]  # Rotate about x-axis 90 deg
#         T_list[i] = rotvec2T(-np.pi / 2, [0, 1, 0]) @ T_list[i]  # Rotate about y-axis 90 deg
#         T_list[i][2, 3] += torso_radius

#     # Add in cap
#     mesh_list.append(mesh_dict["cap"])
#     T = np.eye(4)
#     # T[2, 3] = -AC_DIAMETER / 3
#     T_list.append(T)
#     op_list.append("union")

#     # Add in post
#     # mesh_list.append(mesh_dict["ac_post_extra"])
#     T_list.append(np.eye(4))

#     op_list = ["union" for _ in range(len(mesh_list))]

#     s = Shape(mesh_list, T_list, op_list, "D006", "straight", "test", np.eye(4), mesh_fairing_distance)
#     s_list.append(s)
#     # s.mesh.show()

#     # if calc_dist_from_z_axis(s.mesh) > 22.6:
#     #     raise ValueError


mesh_fairing_distance = 0
size_meshes = [
    "size_ico",
    "size_cube",
    "size_cubeR",
]
scale_factors = np.linspace(1.0, 0.5, 4)
for mesh_name in size_meshes:

    for scale in scale_factors:
        m = mesh_dict[mesh_name].copy()
        m.apply_translation(-m.centroid)  # Center before scaling
        m.apply_scale(scale)

        # Align to z of centroid of largest scale
        m.apply_translation([0, 0, scale_factors[0] * mesh_dict[mesh_name].extents[2] / 2])
        # m.apply_translation([0, 0, -m.bounds[0, 2]])

        # Add cap
        mesh_list = [m]
        T_list = [np.eye(4)]
        op_list = ["union"]
        mesh_list, T_list, op_list = add_cap(mesh_list, T_list, op_list)

        # # Add icosphere
        # ico = trimesh.creation.icosphere(subdivisions=5, radius=5)
        # ico.apply_translation([0, 0, -ico.bounds[0, 2]])
        # mesh_list.append(ico)
        # T_list.append(np.eye(4))
        # op_list.append("union")

        # Add capsule
        # if mesh_name == "size_cubeR":
        capsule_radius = 4.999
        capsule_height = scale_factors[0] * mesh_dict[mesh_name].extents[2] / 2
        capsule = trimesh.primitives.Capsule(radius=capsule_radius, height=capsule_height, sections=64)
        capsule = trimesh.Trimesh(vertices=capsule.vertices, faces=capsule.faces)  # Allow editing
        capsule.apply_translation([0, 0, -capsule.bounds[0, 2] - capsule_radius - 0.001])
        capsule.vertices[capsule.vertices[:, 2] < 0] *= [1, 1, 0.25]
        # Scale vertices below z=0
        # capsule.apply_translation([0, 0, -capsule.bounds[0, 2]])
        mesh_list.append(capsule)
        T_list.append(np.eye(4))
        op_list.append("union")

        label = ""
        description = ""
        s = Shape(mesh_list, T_list, op_list, label, description, "test", np.eye(4), mesh_fairing_distance)
        s_list.append(s)
        s.mesh.show(smooth=False)

        if calc_dist_from_z_axis(s.mesh) > 22.6:
            print(calc_dist_from_z_axis(s.mesh))
            s.mesh.show()
            raise ValueError


list_for_size = [list_for_rotation[i] for i in range(2, len(list_for_rotation), min(len(sf_list), 3))]

# for mesh_list, T_list, op_list, T_UP_SHIFT in list_for_rotation:


save_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso")
for i, s in enumerate(s_list):
    export_shape(s, save_dir, f"torso_{str(i).zfill(3)}")
