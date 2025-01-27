# Torsos
from vh_objects.shaft import Shaft
import trimesh
import numpy as np
from scripts.stim_set_common import create_scene, load_cap, UU, VV, export_shape, slightly_deform_mesh, STL_DIR
from scipy.spatial.transform import Rotation
from scripts.sheets_utilities import make_surface, make_mesh, construct_sheet
from vh_objects.shape import Shape
from vh_objects.utilities import calc_mesh_boolean_and_edges, fuse_meshes, fair_mesh
from trimesh.transformations import rotation_matrix as rotvec2T
from scripts.make_gif import calc_dist_from_z_axis
from scripts.subdivide_box import load_subdivided_box, construct_tetrahedron
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
cone_r1 = 0.920 * torso_radius
cone_r3 = 0.1 * torso_radius
cone_r2 = 0.5 * (cone_r1 + cone_r3)
mesh_fairing_distance = 1
scale_factors = np.linspace(1.0, 0.6, 4)

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

# Cone
torso_cone_K0 = Shaft(
    torso_length,
    cone_r1,
    cone_r2,
    cone_r3,
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
torso_cone_K0.mesh.apply_transform(T_point_z)
torso_cone_K0.mesh.apply_translation([0, 0, -torso_cone_K0.mesh.bounds[0, 2]])
assert abs(torso_cone_K0.l_sphere_radius - torso_radius) < 0.1
print(torso_cone_K0.l_sphere_radius, torso_cone_K0.r_sphere_radius)
print(torso_radius)
# torso_cone_K0.mesh.show()


# ac_round_K0
post_radius = 5
ac_round_K0_shape = Shaft(
    torso_length / 2 + post_radius,
    post_radius,
    post_radius,
    post_radius,
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
ac_round_K0 = ac_round_K0_shape.mesh
T = np.eye(4)
R = Rotation.from_rotvec(np.array([0, -np.pi / 2, 0])).as_matrix()
T[:3, :3] = R
T[2, 3] -= post_radius  # Align spherical end to origin
ac_round_K0.apply_transform(T)
ac_post_extra = ac_round_K0.copy()
ac_post_extra.vertices[ac_post_extra.vertices[:, 2] < 0] *= [1, 1, 0.5]  # Multiply all z vertives below 0 by 0.5


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
    UU=uu,
    VV=vv,
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
sf_ridge_vert_union = make_mesh(surf, 25, 25)
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
sf_sphere_union = trimesh.creation.icosphere(subdivisions=4, radius=sf_radius)
sf_sphere_union.visual.face_colors = color_union

sf_sphere_difference = sf_sphere_union.copy()
sf_sphere_difference.visual.face_colors = color_difference


sf_ridge_vert_difference.apply_translation([0, 0, -0.25])  ## Shift up slightly to improve fusion
sf_sphere_difference.apply_translation([0, 0, -0.25])  ## Shift down slightly to improve fusion
print("Shifting sf_ridge_vert_difference up by 0.25")

# Size meshes

size_ico = trimesh.creation.icosphere(subdivisions=5, radius=torso_radius * 1.25)
box_edge = 2.0 * torso_radius
# size_cube = load_subdivided_box([box_edge, box_edge, box_edge], 10)
size_cube = trimesh.creation.box([box_edge, box_edge, box_edge])
size_cubeR = size_cube.copy()
# T = rotvec2T(np.pi / 3, [1, 0, 0]) @ rotvec2T(np.pi / 4, [0, 0, 1])
T = rotvec2T(np.arctan(np.sqrt(2)), [0, 1, 0]) @ rotvec2T(np.pi / 4, [0, 0, 1])
size_cubeR.apply_transform(T)
size_cubeR.apply_translation(-size_cubeR.centroid)

# Tetrahedron
size_tetr = construct_tetrahedron(box_edge / scale_factors[2])
size_tetrR = size_tetr.copy()
size_tetrR.apply_transform(
    rotvec2T(
        np.pi,
        [
            0,
            1,
            0,
        ],
    )
)
size_tetrR.apply_translation([0, 0, -size_tetrR.bounds[0, 2]])


# scene = trimesh.Scene()
# scene.add_geometry(size_cubeR)
# # Add line along z-axis
# line = trimesh.creation.box([0.5, 0.5, 100])
# line.visual.face_colors = [0, 0, 255, 255]
# scene.add_geometry(line)
# line2 = trimesh.creation.box([0.5, 100, 0.5])
# line2.visual.face_colors = [0, 255, 0, 255]
# scene.add_geometry(line2)
# scene.show()

mesh_dict = {
    "cap": load_cap(),
    "torso_football_K0": torso_football_K0.mesh,
    # "torso_football_K1": torso_football_K1.mesh,
    "torso_cylinder_K0": torso_cylinder_K0.mesh,
    # "torso_cylinder_K1": torso_cylinder_K1.mesh,
    "torso_dumbbell_K0": torso_dumbbell_K0,
    "torso_cone_K0": torso_cone_K0.mesh,
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
    "size_tetr": size_tetr,
    "size_tetrR": size_tetrR,
    "ac_post_extra": ac_post_extra,
}

create_scene(mesh_dict)


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
list_meshes_without_cap = []
list_meshes_without_cap_idx_for_scaling = []
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
    "torso_cone_K0",
]


for mesh_name in base_torsos:

    mesh_list = [mesh_dict[mesh_name].copy()]
    T_list = [np.eye(4)]
    op_list = ["union"]

    # Store dumbbell so it can be rotated below
    list_meshes_without_cap.append(mesh_dict[mesh_name].copy())
    list_meshes_without_cap_idx_for_scaling.append(
        len(list_meshes_without_cap) - 1
    )  # Add idx so that these base torsos will get scaled as well

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

vec_left = np.array([0, -1, 0])
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
    "torso_cone_K0",
]:

    # Calculate the frontright and backright transformation matrices
    T_frontright = calc_T_given_mesh_vec_T_base(mesh_dict[base_torso], vec_right, front_origin)
    T_backright = calc_T_given_mesh_vec_T_base(mesh_dict[base_torso], vec_right, back_origin)

    for sf_type in sf_list:

        for sf_locations in [
            "single",
            "double_same",
            "double_opposite",
            "quad",
            "legs",
        ]:

            mesh_list = [mesh_dict[base_torso].copy()]
            T_list = [np.eye(4)]
            op_list = ["union"]

            if "single" == sf_locations:
                # Single sf at frontright
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    1, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )
            elif "double_same" == sf_locations:
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    1, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )

                mesh_list, T_list, op_list = add_rotations_about_Z(
                    1, T_backright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )

            elif "double_opposite" == sf_locations:
                # Double sf at frontright, frontleft
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    2, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )
            elif "quad" == sf_locations:
                # Quad sf at frontright, frontleft, backright, backleft
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    2, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )

                mesh_list, T_list, op_list = add_rotations_about_Z(
                    2, T_backright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )
            elif "legs" == sf_locations:
                mesh_list, T_list, op_list = add_rotations_about_Z(
                    1, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )

                mesh_list, T_list, op_list = add_rotations_about_Z(
                    1, T_backright, mesh_dict, sf_type, mesh_list, T_list, op_list
                )
                th_shift = np.pi / 3
                for i in range(1, 3):
                    T = rotvec2T(th_shift, [0, 0, 1]) @ T_list[i]
                    mesh_list.append(mesh_list[i].copy())
                    T_list.append(T)
                    op_list.append(op_list[i])

            # elif "octo" in sf_locations:
            #     # Quad sf at frontright, frontleft, backright, backleft
            #     mesh_list, T_list, op_list = add_rotations_about_Z(
            #         4, T_frontright, mesh_dict, sf_type, mesh_list, T_list, op_list
            #     )

            #     mesh_list, T_list, op_list = add_rotations_about_Z(
            #         4, T_backright, mesh_dict, sf_type, mesh_list, T_list, op_list
            #     )
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
            mesh_list = slightly_deform_mesh(mesh_list)
            mesh_list = slightly_deform_mesh(mesh_list)

            # # Store dumbbell so it can be rotated below
            # if base_torso == "torso_dumbbell_K0":
            #     list_for_rotation.append(copy.deepcopy([mesh_list, T_list, op_list, torso_length / 2]))

            s = shape_without_cap = Shape(
                mesh_list, T_list, op_list, "D006", "straight", "test", np.eye(4), mesh_fairing_distance
            )

            # Omit cylinder for rotations because it is so similar to dumbbell when rotated
            if base_torso in ["torso_dumbbell_K0", "torso_football_K0", "torso_cone_K0"]:
                list_meshes_without_cap.append(s.mesh)

                if sf_type in ["sf_point", "sf_sphere_union", "sf_ridge_vert_union"] and sf_locations == "double":
                    list_meshes_without_cap_idx_for_scaling.append(len(list_meshes_without_cap) - 1)

            # Reset lists, add cap
            mesh_list = [s.mesh]
            T_list = [np.eye(4)]
            op_list = ["union"]
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


########################################
### Three but now in different plane ###
########################################
list_meshes_without_cap_rotated = []
for mesh in list_meshes_without_cap:

    m_copy = mesh.copy()

    # Center at origin
    m_copy.apply_translation([0, 0, -torso_length / 2])

    # Rotate
    T = rotvec2T(np.pi / 2, [0, 1, 0]) @ rotvec2T(-np.pi / 2, [1, 0, 0])
    m_copy.apply_transform(T)

    # Return to original position
    m_copy.apply_translation([0, 0, torso_length / 2])

    # Store rotated mesh for scaling below; note only ones with stored idx (above) will be scaled
    list_meshes_without_cap_rotated.append(m_copy.copy())

    mesh_list = [m_copy]
    T_list = [np.eye(4)]
    op_list = ["union"]

    # Add in cap
    mesh_list.append(mesh_dict["cap"])
    T = np.eye(4)
    T_list.append(T)
    op_list.append("union")

    # Add in post
    mesh_list.append(mesh_dict["ac_post_extra"])
    T_list.append(np.eye(4))
    op_list.append("union")
    # op_list = ["union" for _ in range(len(mesh_list))]

    s = Shape(mesh_list, T_list, op_list, "D006", "straight", "test", np.eye(4), mesh_fairing_distance)
    s_list.append(s)
    # s.mesh.show()

    # m_copy.show()


#############
### Scale ###
#############


# Scale the meshes identified above by their idx (total of 3)
for idx in list_meshes_without_cap_idx_for_scaling:
    m = list_meshes_without_cap_rotated[idx].copy()

    # Skip 1.0 since it's already been added
    for scale_factor in scale_factors[1:]:

        # Copy, translate, scale, return
        new_mesh = m.copy()
        new_mesh.apply_translation([0, 0, -torso_length / 2])
        new_mesh.apply_scale(scale_factor)
        new_mesh.apply_translation([0, 0, torso_length / 2])

        # Add cap
        mesh_list = [new_mesh]
        T_list = [np.eye(4)]
        op_list = ["union"]
        mesh_list, T_list, op_list = add_cap(mesh_list, T_list, op_list)

        # Add in post
        mesh_list.append(mesh_dict["ac_post_extra"])
        T_list.append(np.eye(4))
        op_list.append("union")

        label = ""
        description = ""

        s = Shape(mesh_list, T_list, op_list, label, description, "test", np.eye(4), mesh_fairing_distance)
        s_list.append(s)
        # s.mesh.show(smooth=False)
        # new_mesh.show()

        if calc_dist_from_z_axis(s.mesh) > 22.6:
            print(calc_dist_from_z_axis(s.mesh))
            s.mesh.show()
            raise ValueError


# Scale a sphere, cube, and rotated cube.
mesh_fairing_distance = 0
size_meshes = [
    "size_tetr",
    "size_tetrR",
    "size_ico",
    "size_cube",
    "size_cubeR",
]
for mesh_name in size_meshes:

    for scale in scale_factors:
        m = mesh_dict[mesh_name].copy()
        m.apply_translation(-m.centroid)  # Center before scalinG
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

        # Adust top of cap for small tetrahedron
        if mesh_name == "size_tetr":
            bottom_of_tetr = mesh_list[0].bounds[0, 2]
            capsule.apply_translation([0, 0, -bottom_of_tetr - 0.01])
            capsule.vertices[capsule.vertices[:, 2] > +0] *= [1, 1, 0.1]
            capsule.apply_translation([0, 0, -(-bottom_of_tetr - 0.01)])

        if mesh_name == "size_tetrR":
            top_of_tetr = mesh_list[0].bounds[1, 2]
            capsule.apply_translation([0, 0, -top_of_tetr + 0.01])
            capsule.vertices[capsule.vertices[:, 2] > 0] *= [1, 1, 0.1]
            capsule.apply_translation([0, 0, -(-top_of_tetr + 0.01)])

        # Scale vertices below z=0
        # capsule.apply_translation([0, 0, -capsule.bounds[0, 2]])
        mesh_list.append(capsule)
        T_list.append(np.eye(4))
        op_list.append("union")

        label = ""
        description = ""
        s = Shape(mesh_list, T_list, op_list, label, description, "test", np.eye(4), mesh_fairing_distance)
        s_list.append(s)
        # s.mesh.show(smooth=False)

        if calc_dist_from_z_axis(s.mesh) > 22.6:
            print(calc_dist_from_z_axis(s.mesh))
            s.mesh.show()
            raise ValueError


save_dir = Path(STL_DIR, "torso")
start_idx = 600
for i, s in enumerate(s_list):
    label = "G" + str(start_idx + i).zfill(3)
    try:
        export_shape(s, save_dir, label)
    except:
        print(f"Failed to save {label}")
        continue


# save_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso")
# for i, s in enumerate(s_list):

#     if calc_dist_from_z_axis(s.mesh) > 22.6:
#         print(calc_dist_from_z_axis(s.mesh))
#         s.mesh.show()
#         raise ValueError

#     export_shape(s, save_dir, f"torso_{str(i).zfill(3)}")
