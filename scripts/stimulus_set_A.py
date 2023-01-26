# Stimulus Set A
import numpy as np
from objects.utilities import (
    find_cp_for_desired_radius,
    approximate_arc,
    get_deformation_vertex,
    fuse_meshes,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.backbone import Backbone
from objects.shape import Shape
import trimesh
import scipy
from multiprocessing import Pool
import pickle

# Parameters
RADIUS_LIST = [6, 9, 12, 15]  # mm
BACKBONE_ANGLE_LIST = np.linspace(0, np.pi, 5)  # radians
SD_MAGNITUDE = 4  # mm - surface defromation magnitudes
SD_SIGMA = 2
SD_POSITIONS = [
    1 / 3,
    2 / 3,
]  # Distance along backbone at which surface deformations made; on interval [0,1]
LENGTH = 80  # mm; does not include post or interface
NUM_CP_PER_CROSS_SECTION = 8


# Helper variables
c = np.cos
s = np.sin


def construct_shapes(args):
    s = np.sin
    c = np.cos

    backbone_angle, radius, sd_list, label = args

    # Construct medial-axis backbone
    backbone_cp = approximate_arc(backbone_angle, LENGTH)

    # Rotate to align along +Y axis
    R = np.array(
        [
            [c(backbone_angle / 2), s(backbone_angle / 2), 0],
            [-s(backbone_angle / 2), c(backbone_angle / 2), 0],
            [0, 0, 1],
        ]
    )
    backbone_R = backbone_cp @ R
    backbone = Backbone(backbone_R, reparameterize=True)

    # Construct cross sections of shape
    cp_radius = find_cp_for_desired_radius(radius, NUM_CP_PER_CROSS_SECTION)
    cs_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(
        -1, 1
    )
    cs_cp = np.hstack((cp_radius * np.cos(cs_th), cp_radius * np.sin(cs_th)))
    cs_list = [
        CrossSection(controlpoints=cs_cp, position=position)
        for position in np.linspace(0.0, 1, 20)
    ]

    # Axial Component
    ac = AxialComponent(backbone, cs_list, smooth_with_post=False)

    # Shape
    s = Shape([ac], label=label)

    # Get correct point and normal to apply surface deformations
    point_and_normal = []
    for sd in sd_list:

        if sd == []:
            continue

        mag, t = sd
        pt1, normal1 = get_deformation_vertex(s.mesh, s.ac_list[0], t, N_rotation=np.pi)
        point_and_normal.append([pt1, normal1])

    # Apply surface deformations
    for i, sd in enumerate(sd_list):

        if sd == []:
            continue

        mag, t = sd
        pt, normal = point_and_normal[i]
        if mag != 0:
            s.apply_gaussian_deformation(pt, normal, mag, SD_SIGMA, num_smoothing=1)

    if backbone_angle <= np.pi / 2:
        INTERFACE_Y_SHIFT = 0
    else:
        INTERFACE_Y_SHIFT = s.mesh.vertices[:, 1].min() + 6
    OVERLAP_OFFSET = 1
    post_backbone_cp = np.array(
        [
            [s.mesh.vertices[:, 0].min() - 2 * OVERLAP_OFFSET, INTERFACE_Y_SHIFT, 0],
            [
                s.mesh.vertices[:, 0].min() / 2 - 1 * OVERLAP_OFFSET,
                INTERFACE_Y_SHIFT,
                0,
            ],
            [3 * OVERLAP_OFFSET, INTERFACE_Y_SHIFT, 0],
        ]
    )
    post_backbone = Backbone(post_backbone_cp, reparameterize=True)
    post_radius = 5
    post_cp_radius = find_cp_for_desired_radius(post_radius, NUM_CP_PER_CROSS_SECTION)
    post_th = np.linspace(
        0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False
    ).reshape(-1, 1)
    post_cp = np.hstack(
        (post_cp_radius * np.cos(post_th), post_cp_radius * np.sin(post_th))
    )
    post_cs_list = [
        CrossSection(controlpoints=post_cp, position=0.0),
        CrossSection(controlpoints=post_cp, position=1.0),
    ]
    post_ac = AxialComponent(post_backbone, post_cs_list, smooth_with_post=False)

    # Interface
    scene = trimesh.Scene()
    interface = trimesh.load_mesh(
        "/home/oconnorlab/code/objects/assets/Interface_0024 v2.stl"
    )
    R = trimesh.transformations.rotation_matrix(-np.pi / 2, np.array([0, 0, 1]))
    R[0, 3] = s.mesh.vertices[:, 0].min() - OVERLAP_OFFSET
    R[1, 3] = INTERFACE_Y_SHIFT  # Radius of post if 5mm
    interface = interface.apply_transform(R)

    # Fuse post and interface
    post_shape = fuse_meshes(
        post_ac.mesh, s.mesh, fairing_distance=3, operation="union"
    )
    interface_post_shape = fuse_meshes(post_shape, interface, 0, "union")
    s.mesh_with_interface = interface_post_shape

    COM = backbone.r(0.5)[0]
    # Rotate shape for easier viewing
    shape_euler = np.array([7 * np.pi / 8, 0, np.pi])
    R = np.eye(4)
    R[:3, :3] = scipy.spatial.transform.Rotation.from_euler(
        "xyz", shape_euler
    ).as_matrix()
    COM_rot = R[:3, :3] @ COM  # Rotate center of mass as well
    R[:3, 3] = -COM_rot  # Add COM to transformation matrix

    # new_mesh = mesh_rot
    # # new_mesh = new_mesh.apply_transform(R)
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.plot(
    #     new_mesh.vertices[::100, 0],
    #     new_mesh.vertices[::100, 1],
    #     new_mesh.vertices[::100, 2],
    #     "k.",
    # )
    # ax.plot(
    #     s.mesh.vertices[::100, 0],
    #     s.mesh.vertices[::100, 1],
    #     s.mesh.vertices[::100, 2],
    #     "g.",
    # )
    # ax.plot(
    #     new_mesh.center_mass[0], new_mesh.center_mass[1], new_mesh.center_mass[2], "r."
    # )
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.show()

    base_dir = "/home/oconnorlab/code/objects/sample_shapes/stimulus_set_A"
    s.save_mesh_as_png(
        base_dir + "/png",
        return_img=False,
        rotation=R,
        interface=True,
    )

    # Pickle object
    s.save_as_pickle(base_dir + "/pkl")

    print("Finished", s.label)


# List of list of list containing surface deformation magnitudes and locations (t)
sd_table = [
    [[]],
    [[SD_MAGNITUDE, 3 / 9]],
    [[SD_MAGNITUDE, 3 / 9], [SD_MAGNITUDE, 4 / 9]],
    [[SD_MAGNITUDE, 3 / 9], [SD_MAGNITUDE, 5 / 9]],
    [[SD_MAGNITUDE, 3 / 9], [SD_MAGNITUDE, 6 / 9]],
]

arg_list = []
count = 0
for j in range(len(BACKBONE_ANGLE_LIST)):
    backbone_angle = BACKBONE_ANGLE_LIST[j]
    for i in range(len(RADIUS_LIST)):
        radius = RADIUS_LIST[i]
        for sd_list in sd_table:
            arg_list.append([backbone_angle, radius, sd_list, "shape_{}".format(count)])
            count += 1

# construct_shapes(arg_list[33])
with Pool() as pool:
    pool.map(construct_shapes, arg_list)


# Load a shape
filename = "/home/oconnorlab/code/objects/sample_shapes/stimulus_set_A/pkl/shape_20.pkl"
with open(filename) as f:
    shape = pickle.load(f)
# # s = construct_shapes([np.pi, 15, 3, 3, "label_1"])
# # s.mesh_with_interface.show()
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot(
#     s.mesh.vertices[::100, 0],
#     s.mesh.vertices[::100, 1],
#     s.mesh.vertices[::100, 2],
#     "k.",
# )
# ax.plot(s.mesh.center_mass[0], s.mesh.center_mass[1], s.mesh.center_mass[2], "r.")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.show()


# Plot verts and center of mass


# broken = trimesh.repair.broken_faces(interface, color=[255, 0, 0, 255])
# interface.show(smooth=False)

# scene.add_geometry(interface)
# scene.add_geometry(s.mesh)
# scene.add_geometry(post_ac.mesh)
# scene.show()
# s.mesh.show()


# # Hook

# # Points along curve
# angle = np.pi
# dist = LENGTH
# radius = dist / angle
# t = np.linspace(angle, 0, 15, endpoint=False).reshape(-1, 1)
# t = np.flip(t)
# c = np.cos
# s = np.sin
# curve_cp = np.hstack([radius * s(t), radius * c(t), np.zeros(t.shape)])
# curve_cp -= np.array([0, radius, 0])

# # Smoothed hook backbone
# j = 4
# backbone_angle = BACKBONE_ANGLE_LIST[j]
# backbone_cp = approximate_arc(backbone_angle, LENGTH)
# s = backbone_cp[1, 0]
# s = 0
# POST_LENGTH = 15
# post_cp = np.array(
#     [[-s, POST_LENGTH, 0], [-s, POST_LENGTH * 2 / 3, 0], [-s, POST_LENGTH / 3, 0]]
# )
# comb_cp = np.concatenate([post_cp, curve_cp])
# backbone = Backbone(comb_cp, reparameterize=True)
# xy = backbone.r(np.linspace(0, 1, 100))

# # Find midpoint of curve
# samples = backbone.r(np.linspace(0, 1, 1000))
# s = np.sin
# midpoint = np.array(
#     [radius * s(backbone_angle / 2), radius * c(backbone_angle / 2) - radius, 0]
# )
# dists = np.linalg.norm(samples - midpoint, axis=1)
# idx = dists.argmin()
# mid_t = np.linspace(0, 1, 1000)[idx]

# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1)
# # axs.plot(backbone_cp[:,0], backbone_cp[:,1], "b.", linewidth=10)
# # axs.plot(post_cp[:,0], post_cp[:,1], "g.")
# axs.plot(xy[:, 0], xy[:, 1], "r-", linewidth=10)
# axs.plot(comb_cp[:, 0], comb_cp[:, 1], "k.")

# axs.set_aspect("equal")
# plt.show()


# i = 4
# radius = RADIUS_LIST[i]
# cp_radius = fit_radius(radius, NUM_CP_PER_CROSS_SECTION)
# th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
# cs_cp = np.hstack((cp_radius * np.cos(th), cp_radius * np.sin(th)))
# post_radius = fit_radius(5, NUM_CP_PER_CROSS_SECTION)
# post_cp = np.hstack((post_radius * np.cos(th), post_radius * np.sin(th)))
# curve_t = np.flip(np.linspace(1, 1 - 2 * (1 - mid_t), 20, endpoint=False))
# cs_list_curve = [
#     CrossSection(controlpoints=cs_cp, position=position) for position in curve_t
# ]
# cs_list_post = [
#     CrossSection(controlpoints=post_cp, position=position)
#     for position in np.linspace(0, (1 - 2 * (1 - mid_t)) * 1 / 2, 3, endpoint=True)
# ]
# cs_list = [*cs_list_post, *cs_list_curve]
# # Axial Component
# ac = AxialComponent(backbone, cs_list, smooth_with_post=False)
# ac.mesh.show()


# post_begin = 1 - 2 * (1 - mid_t)
# new_t = np.linspace(0, 1, 20)
# new_cs_curve = [
#     CrossSection(controlpoints=cs_cp, position=position)
#     for position in new_t[new_t >= post_begin]
# ]
# new_cs_post = [
#     CrossSection(controlpoints=post_cp, position=position)
#     for position in new_t[new_t < post_begin]
# ]
# new_ac = AxialComponent(backbone, [*new_cs_post, *new_cs_curve], smooth_with_post=False)
# new_ac.mesh.show(smooth=False)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# for i in range(ac.controlpoints.shape[0]):
#     ax.plot(
#         ac.controlpoints[i, :, 0],
#         ac.controlpoints[i, :, 1],
#         ac.controlpoints[i, :, 2],
#         "b-",
#     )
# plt.show()


# ## Fuse to interface

# # Generate ac
# new_t = np.linspace(0, 1, 20)
# cs_curve = [CrossSection(controlpoints=cs_cp, position=position) for position in new_t]
# backbone_cp = approximate_arc(backbone_angle, LENGTH)
# backbone = Backbone(backbone_cp, reparameterize=True)
# ac = AxialComponent(backbone, cs_curve, smooth_with_post=False)
# ac.mesh.show(smooth=False)

# # Shift ac so that furthest vertex along -x axis are at origin
# old_verts = ac.mesh.vertices
# bottommost_vert = old_verts[old_verts[:, 1].argmin()]
# new_verts = old_verts - bottommost_vert
# new_verts += np.array([0, 5, 0])
# ac.mesh.vertices = new_verts

# # Add in interface
# import trimesh

# scene = trimesh.Scene()
# scene.add_geometry(ac.mesh)
# interface = trimesh.load_mesh(
#     "/home/oconnorlab/code/objects/assets/Interface_0023_aligned_to_origin v3.stl"
# )
# scene.add_geometry(interface)
# scene.show()

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot(new_verts[:, 0], new_verts[:, 1], new_verts[:, 2], "k.")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.show()

# # plt.show()
# # # Backbone Angles / Curvature
# # for j in range(len(BACKBONE_ANGLE_LIST)):
# #     j = 4
# #     backbone_angle = BACKBONE_ANGLE_LIST[j]
# #     backbone_cp = approximate_arc(backbone_angle, LENGTH)
# #     c = np.cos
# #     s = np.sin
# #     R = np.array([[c(backbone_angle/2), s(backbone_angle/2), 0],[-s(backbone_angle/2), c(backbone_angle/2), 0],[0,0,1]])
# #     backbone_R = backbone_cp @ R
# #     backbone = Backbone(backbone_R, reparameterize=True)

# #     # Radii
# #     for i in range(len(RADIUS_LIST)):
# #         radius = RADIUS_LIST[i]
# #         cp_radius = fit_radius(radius, NUM_CP_PER_CROSS_SECTION)
# #         th = np.linspace(0, 2*np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1,1)
# #         cs_cp = np.hstack((cp_radius*np.cos(th), cp_radius*np.sin(th)))
# #         cs_list = [CrossSection(controlpoints=cs_cp, position=position) for position in np.linspace(0.1, 1,20)]

# #         # Axial Component
# #         ac = AxialComponent(backbone, cs_list, smooth_with_post=True)
# #         ac.mesh.show()
# #         break
# #     break
