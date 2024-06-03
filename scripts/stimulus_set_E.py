from objects.shaft import Shaft
import numpy as np
from objects.utilities import make_surface, make_mesh
import trimesh
from scipy.spatial.transform import Rotation
import copy


#################
### Utilities ###
#################
def clip_along_axis(cp, axis, thickness):

    cp_clipped = np.zeros(cp.shape)

    for cs_num in range(cp.shape[0]):

        # Skip ends
        if cs_num == 0 or cs_num == cp.shape[0] - 1:
            cp_clipped[cs_num] = cp[cs_num]
            continue

        # Calculate local coordinate system
        cs_cp = cp[cs_num, :, :]
        P0 = np.mean(cs_cp, axis=0)
        P1 = cs_cp[0]
        P2 = cs_cp[1]
        vecT = P1 - P0
        vecT /= np.linalg.norm(vecT)
        vecN = np.cross(vecT, (P2 - P0) / np.linalg.norm(P2 - P0))
        vecN /= np.linalg.norm(vecN)
        vecB = np.cross(vecT, vecN)

        # Translate to origin
        cs_cp_t = cs_cp - P0

        # Rotate to align with xz plane
        Ra = np.hstack([vecT.reshape(3, 1), vecN.reshape(3, 1), vecB.reshape(3, 1)])
        goal = np.eye(3)
        R = np.dot(goal, np.linalg.inv(Ra))
        cs_cp_R = np.dot(R, cs_cp_t.T).T

        # Shrink along axis
        cs_cp_R[:, axis] = np.clip(cs_cp_R[:, axis], -thickness / 2, thickness / 2)

        # Undo transformations
        cs_cp_n = np.dot(R.T, cs_cp_R.T).T
        cs_cp_n += P0
        cp_clipped[cs_num, :, :] = cs_cp_n
    return cp_clipped


def plot_cp(cp):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(cp.shape[0]):
        ax.plot(cp[i, :, 0], cp[i, :, 1], cp[i, :, 2], "b-*")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Set axis limits
    ax_min = np.min(cp)
    ax_max = np.max(cp)
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_zlim(ax_min, ax_max)
    plt.show()


#######################
### Make components ###
#######################

comp_dict = {}

# inputs
CAPSULE_RADIUS = 5
CAPSULE_LENGTH = 20
FLATTENED_THICKNESS = 4
NUM_CS = 5
NUM_CP_PER_CROSS_SECTION = 20
uu = 50
vv = 50
K1_THETA = np.pi / 2
CAPSULE_K1_LENGTH = CAPSULE_LENGTH * 1.7

# Tranform to be pointing upwards
T_point_z = np.eye(4)
T_point_z[:3, :3] = Rotation.from_euler("xyz", np.array([0, -np.pi / 2, 0])).as_matrix()


# capsule_K0
capsule_K0 = Shaft(
    CAPSULE_LENGTH,
    1.0 * CAPSULE_RADIUS,
    1.0 * CAPSULE_RADIUS,
    1.0 * CAPSULE_RADIUS,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
capsule_K0.mesh.apply_transform(T_point_z)
comp_dict["capsule_K0"] = capsule_K0.mesh

# Capsule_K0 flattened
capsule_K0_F1_cp = capsule_K0.cp.copy()
capsule_K0_F1_cp[:, :, 2] = np.clip(capsule_K0_F1_cp[:, :, 2], -FLATTENED_THICKNESS / 2, FLATTENED_THICKNESS / 2)
capsule_K0_F1_surf = make_surface(capsule_K0_F1_cp)
capsule_K0_F1_mesh = make_mesh(capsule_K0_F1_surf, uu, vv)
capsule_K0_F1_mesh.apply_transform(T_point_z)
comp_dict["capsule_K0_F1"] = capsule_K0_F1_mesh

# Capsule_K0 flattened and rotated
T_rot = np.eye(4)
T_rot[:3, :3] = Rotation.from_euler("xyz", np.array([0, 0, np.pi / 2])).as_matrix()
capsule_K0_F2_mesh = capsule_K0_F1_mesh.copy()
capsule_K0_F2_mesh.apply_transform(T_rot)
comp_dict["capsule_K0_F2"] = capsule_K0_F2_mesh

# Capsule_K1
capsule_K1 = Shaft(
    CAPSULE_K1_LENGTH,
    1.0 * CAPSULE_RADIUS,
    1.0 * CAPSULE_RADIUS,
    1.0 * CAPSULE_RADIUS,
    theta=K1_THETA,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
capsule_K1.mesh.apply_transform(T_point_z)
comp_dict["capsule_K1"] = capsule_K1.mesh

# Capsule_K1 flattened
capsule_K1_F1_cp = capsule_K1.cp.copy()
capsule_K1_F1_cp = clip_along_axis(capsule_K1_F1_cp, 0, FLATTENED_THICKNESS)
capsule_K1_F1_surf = make_surface(capsule_K1_F1_cp)
capsule_K1_F1_mesh = make_mesh(capsule_K1_F1_surf, uu, vv)
capsule_K1_F1_mesh.apply_transform(T_point_z)
comp_dict["capsule_K1_F1"] = capsule_K1_F1_mesh

# Capsule_K1 flattened
capsule_K1_F2_cp = capsule_K1.cp.copy()
capsule_K1_F2_cp = clip_along_axis(capsule_K1_F2_cp, 2, FLATTENED_THICKNESS)
capsule_K1_F2_surf = make_surface(capsule_K1_F2_cp)
capsule_K1_F2_mesh = make_mesh(capsule_K1_F2_surf, uu, vv)
capsule_K1_F2_mesh.apply_transform(T_point_z)
comp_dict["capsule_K1_F2"] = capsule_K1_F2_mesh

# Capsule_K1 flattened and curved in cross sectional plane
# capsule_K1_F2_K_cp = capsule_K1_F1_cp.copy()
# num_cs = capsule_K1_F2_cp.shape[0]

# # bend cs
# full_bend_angle = -np.pi / 2
# bend_angle_by_cs = np.hstack(
#     [
#         np.linspace(0, full_bend_angle, num=NUM_CS),
#         np.ones(NUM_CS) * full_bend_angle,
#         np.linspace(full_bend_angle, 0, num=NUM_CS),
#     ]
# )
# for cs_num in range(1, num_cs - 1):
#     cs_cp = capsule_K1_F2_K_cp[cs_num, :, :]

#     # transform to xy plane
#     cs_cp_centerpoint = np.mean(cs_cp, axis=0)
#     cs_cp = cs_cp - cs_cp_centerpoint
#     vec1 = np.array([0, 0, 1])
#     vec2 = cs_cp[0] - cs_cp[NUM_CP_PER_CROSS_SECTION // 2]
#     vec3 = np.cross(vec1, vec2)
#     vec3 = vec3 / np.linalg.norm(vec3)
#     vec2 = np.cross(vec3, vec1)
#     vec2 = vec2 / np.linalg.norm(vec2)
#     R_curr = np.vstack([vec1, vec2, vec3])
#     R_goal = np.eye(3)
#     R = np.dot(R_goal, np.linalg.inv(R_curr))
#     cs_cp_t = np.dot(cs_cp, R)

#     bend_angle = bend_angle_by_cs[cs_num]
#     bend_length = cs_cp_t[:, 0].max() - cs_cp_t[:, 0].min()
#     bend_radius = bend_length / bend_angle

#     num_cs_cp = cs_cp.shape[0]
#     right_pairs = np.vstack([np.arange(num_cs_cp / 4 + 1), np.arange(num_cs_cp / 2, num_cs_cp / 4 - 1, -1)])
#     left_pairs = np.vstack(
#         [
#             np.arange(NUM_CP_PER_CROSS_SECTION - 1, num_cs_cp / 4 * 3 - 1, -1),
#             np.arange(num_cs_cp / 2 + 1, num_cs_cp / 4 * 3 + 1),
#         ]
#     )
#     pair_indices = np.hstack([right_pairs, left_pairs]).astype("int")

#     cs_cp_t_bend = np.zeros_like(cs_cp_t)

#     for pair_idx in pair_indices.T:
#         pair = cs_cp_t[pair_idx, :]
#         pair_center = np.mean(pair, axis=0)
#         dist_pair_to_cs_center = np.linalg.norm(pair_center) * np.sign(pair_center[0])
#         arc_length = dist_pair_to_cs_center
#         pair_angle = arc_length / bend_radius
#         pair_new_center = np.array(
#             [bend_radius * np.sin(pair_angle), bend_radius * np.cos(pair_angle) - bend_radius, 0]
#         )
#         from scipy.spatial.transform import Rotation

#         Rz = Rotation.from_euler("z", pair_angle).as_matrix()
#         pair_at_center = pair - pair_center
#         pair_t = pair_at_center @ Rz + pair_new_center

#         cs_cp_t_bend[pair_idx, :] = pair_t

#     t = np.linspace(bend_angle / -2, bend_angle / 2, 100)
#     x = bend_radius * np.sin(t)
#     y = bend_radius * np.cos(t) - bend_radius
#     z = np.zeros_like(x)

#     cs_cp_new = cs_cp_t_bend @ R.T
#     cs_cp_new += cs_cp_centerpoint
#     capsule_K1_F2_K_cp[cs_num, :, :] = cs_cp_new
#     # plot_cp(capsule_K1_F2_K_cp)


# plot_cp(capsule_K1_F2_K_cp)

# # Make mesh
# capsule_K1_F2_K_surf = make_surface(capsule_K1_F2_K_cp)
# capsule_K1_F2_K_mesh = make_mesh(capsule_K1_F2_K_surf, uu, vv)
# capsule_K1_F2_K_mesh.apply_transform(T_point_z)
# comp_dict["capsule_K1_F2_K"] = capsule_K1_F2_K_mesh
# comp_dict["capsule_K1_F2_K"].show(smooth=False)

# import matplotlib.pyplot as plt

# ax = plt.axes(projection="3d")
# ax.plot(cs_cp_t[:, 0], cs_cp_t[:, 1], cs_cp_t[:, 2], "b-*")
# ax.plot(x, y, z, "r")
# ax.plot(cs_cp_t_bend[:, 0], cs_cp_t_bend[:, 1], cs_cp_t_bend[:, 2], "g-*")
# axmin = np.min(cs_cp_t)
# axmax = np.max(cs_cp_t)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")

# ax.set_xlim(axmin, axmax)
# ax.set_ylim(axmin, axmax)
# ax.set_zlim(axmin, axmax)

# plt.show()

# plot_cp(cs_cp_t.reshape(1, -1, 3))

# plot_cp(capsule_K1_F2_K_cp)


# Make rounded, flat, bent cross section
def construct_rounded_cs(bend_angle, thickness, full_width, num_cp_per_cs):

    width = full_width - thickness  # Width of the flat part, accounts for hemispherical edges
    th = np.linspace(0, np.pi, num_cp_per_cs).reshape(-1, 1)
    if bend_angle == 0:
        top = np.hstack(
            [
                np.linspace(-width / 2, width / 2, num_cp_per_cs).reshape(-1, 1),
                thickness / 2 * np.ones((num_cp_per_cs, 1)),
                np.zeros((num_cp_per_cs, 1)),
            ]
        )
        bottom = np.hstack(
            [
                np.linspace(-width / 2, width / 2, num_cp_per_cs).reshape(-1, 1),
                -thickness / 2 * np.ones((num_cp_per_cs, 1)),
                np.zeros((num_cp_per_cs, 1)),
            ]
        )
        right = np.hstack([thickness / 2 * np.sin(th), thickness / 2 * np.cos(th), np.zeros_like(th)])
        left = np.hstack([-thickness / 2 * np.sin(th), -thickness / 2 * np.cos(th), np.zeros_like(th)])
        right += np.array([width / 2, 0, 0])
        left += np.array([-width / 2, 0, 0])

    else:
        center_radius = CAPSULE_RADIUS / bend_angle

        t = np.linspace(-bend_angle / 2, bend_angle / 2, num_cp_per_cs).reshape(-1, 1)

        top = np.hstack(
            [
                (center_radius + thickness / 2) * np.sin(t),
                (center_radius + thickness / 2) * np.cos(t) - center_radius,
                np.zeros_like(t),
            ]
        )
        bottom = np.hstack(
            [
                (center_radius - thickness / 2) * np.sin(t),
                (center_radius - thickness / 2) * np.cos(t) - center_radius,
                np.zeros_like(t),
            ]
        )
        right = np.hstack([thickness / 2 * np.sin(th), thickness / 2 * np.cos(th), np.zeros_like(th)])
        left = np.hstack([-thickness / 2 * np.sin(th), -thickness / 2 * np.cos(th), np.zeros_like(th)])
        Tz = Rotation.from_euler("z", bend_angle / 2).as_matrix()
        right = right @ Tz
        right += np.array([center_radius * np.sin(t[-1])[0], center_radius * np.cos(t[-1])[0] - center_radius, 0])
        left = left @ Tz.T
        left += np.array([center_radius * np.sin(t[0])[0], center_radius * np.cos(t[0])[0] - center_radius, 0])

    cp = np.vstack([top[1:-1], right, bottom[-2:0:-1], left])
    return cp


bend_angle = np.pi / 2
thickness = FLATTENED_THICKNESS
width = CAPSULE_RADIUS * 2
num_cp_per_cs = 10
cp = construct_rounded_cs(bend_angle, thickness, width, num_cp_per_cs)

# arc_array = approximate_arc(np.pi, np.pi*thickness/2,5)
import matplotlib.pyplot as plt

ax = plt.axes(projection="3d")
# ax.plot(top[:, 0], top[:, 1], top[:, 2], "b-*")
# ax.plot(bottom[:, 0], bottom[:, 1], bottom[:, 2], "r-*")
# ax.plot(right[:, 0], right[:, 1], right[:, 2], "g-*")
# ax.plot(left[:, 0], left[:, 1], left[:, 2], "y-*")
ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], "k-*")
# ax.plot(arc_array[:, 0], arc_array[:, 1], arc_array[:, 2], "r-*")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
axmin = np.min(cp)
axmax = np.max(cp)
ax.set_xlim(axmin, axmax)
ax.set_ylim(axmin, axmax)
ax.set_zlim(axmin, axmax)
plt.show()

from objects.utilities import approximate_arc

NUM_MIDDLE_CS = NUM_CS * 3

cp_new = np.zeros((2 * NUM_CS + NUM_MIDDLE_CS, (num_cp_per_cs - 1) * 4, 3))
b = capsule_K1.backbone
x = np.concatenate(
    [
        capsule_K1.x[:NUM_CS],
        np.linspace(capsule_K1.x[NUM_CS], capsule_K1.x[-NUM_CS - 1], NUM_MIDDLE_CS),
        capsule_K1.x[-NUM_CS:],
    ]
)
y = np.concatenate([capsule_K1.y[:NUM_CS], np.ones(NUM_MIDDLE_CS), capsule_K1.y[-NUM_CS:]])
# y = capsule_K1.y
y = y / max(y)
y[NUM_CS : NUM_CS + NUM_MIDDLE_CS] = 1

MAX_BEND = np.pi / 2
bend_angles = np.concatenate(
    [np.linspace(0, MAX_BEND, num=NUM_CS), np.ones(NUM_MIDDLE_CS) * MAX_BEND, np.linspace(MAX_BEND, 0, num=NUM_CS)]
)
bend_angles[[1, -2]] = 0
for i in range(len(x)):

    bend_angle = bend_angles[i]
    cp = construct_rounded_cs(bend_angle, thickness, width, num_cp_per_cs)
    cp = cp[:, [2, 1, 0]]
    Tx = Rotation.from_euler("x", np.pi).as_matrix()
    cp = cp @ Tx

    base_cs = cp.copy()

    new_cs = base_cs.copy()

    # Scale according to y
    new_cs *= y[i]

    # Shift according to x
    pos = np.round((x[i] - x[0]) / (x[-1] - x[0]), 8)
    assert 0.0 <= pos <= 1.0
    T = np.eye(4)
    T[:3, 0] = b.T(pos).reshape(-1)
    T[:3, 1] = b.N(pos).reshape(-1)
    T[:3, 2] = b.B(pos).reshape(-1)
    T[:3, 3] = b.r(pos).reshape(-1)

    # Homogenous coordinates
    homo_cs = np.hstack([new_cs, np.ones((new_cs.shape[0], 1))])

    # Transform
    T_cs = homo_cs @ T.T

    # Populate
    cp_new[i] = T_cs[:, :3]

surf = make_surface(cp_new)
mesh = make_mesh(surf, 75, 75)

# mesh = mesh.apply_transform(T_point_z)
comp_dict["capsule_K1_F2_K"] = mesh

# Plot all components
scene = trimesh.Scene()
# x_trans = 0
# gap = 3
# for key in comp_dict:

#     # Translate mesh so that they don't overlap
#     mesh_copy = copy.deepcopy(comp_dict[key])
#     mesh_copy.apply_translation([-mesh_copy.bounds[0][0] + x_trans, 0, 0])  # Align left side to origin
#     scene.add_geometry(mesh_copy)
#     x_trans = mesh_copy.bounds[1][0] + gap


# Plot points in scene
pts = cp_new.reshape(-1, 3)
# pts = np.hstack([pts, np.ones([len(pts), 1])])
# pts = pts @ T_point_z
# # pts = pts[:, :3]
scene.add_geometry(comp_dict["capsule_K1_F2_K"])
scene.add_geometry(trimesh.points.PointCloud(pts))

scene.show(smooth=False)

comp_dict["capsule_K1_F2_K"].extents
comp_dict["capsule_K0"].extents
