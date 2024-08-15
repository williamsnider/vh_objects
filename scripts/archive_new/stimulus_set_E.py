from objects.shaft import Shaft
import numpy as np
from objects.utilities import make_surface, make_mesh
import trimesh
from scipy.spatial.transform import Rotation
import copy


#################
### Utilities ###
#################
# Plot all components
def plot_components(comp_dict):
    scene = trimesh.Scene()
    x_trans = 0
    gap = 3
    for key in comp_dict:

        # Translate mesh so that they don't overlap
        mesh_copy = copy.deepcopy(comp_dict[key])
        mesh_copy.apply_translation([-mesh_copy.bounds[0][0] + x_trans, 0, 0])  # Align left side to origin
        scene.add_geometry(mesh_copy)
        x_trans = mesh_copy.bounds[1][0] + gap

    scene.show(smooth=False)


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


def construct_cp_from_cs_func(cs_func, base_shaft, NUM_CS, thickness, width, MAX_BEND):
    # bend_angle = np.pi / 2
    # thickness = FLATTENED_THICKNESS
    # cp = cs_func(bend_angle, thickness, width, num_cp_per_cs)

    NUM_MIDDLE_CS = NUM_CS * 3
    num_cp_per_cs = 10
    width = CAPSULE_RADIUS * 2

    cp_new = np.zeros((2 * NUM_CS + NUM_MIDDLE_CS, (num_cp_per_cs - 1) * 4, 3))
    b = base_shaft.backbone
    x = np.concatenate(
        [
            base_shaft.x[:NUM_CS],
            np.linspace(base_shaft.x[NUM_CS], base_shaft.x[-NUM_CS - 1], NUM_MIDDLE_CS),
            base_shaft.x[-NUM_CS:],
        ]
    )
    y = np.concatenate([base_shaft.y[:NUM_CS], np.ones(NUM_MIDDLE_CS), base_shaft.y[-NUM_CS:]])
    # y = base_shaft.y
    y = y / max(y)
    y[NUM_CS : NUM_CS + NUM_MIDDLE_CS] = 1

    bend_angles = np.concatenate(
        [np.linspace(0, MAX_BEND, num=NUM_CS), np.ones(NUM_MIDDLE_CS) * MAX_BEND, np.linspace(MAX_BEND, 0, num=NUM_CS)]
    )
    bend_angles[[1, -2]] = 0
    for i in range(len(x)):

        bend_angle = bend_angles[i]
        cp = cs_func(bend_angle, thickness, width, num_cp_per_cs)
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

    return cp_new


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

# Base cylinder
base_cylinder_radius = 20
base_cylinder_height = 5
base_cylinder = trimesh.creation.cylinder(radius=base_cylinder_radius, height=base_cylinder_height, sections=20)
base_cylinder.apply_translation([0, 0, -base_cylinder_height])
comp_dict["base_cylinder"] = base_cylinder


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
capsule_K0_F1_cp = construct_cp_from_cs_func(
    construct_rounded_cs, capsule_K0, NUM_CS, FLATTENED_THICKNESS, 2 * CAPSULE_RADIUS, 0
)
capsule_K0_F1_surf = make_surface(capsule_K0_F1_cp)
capsule_K0_F1_mesh = make_mesh(capsule_K0_F1_surf, uu, vv)
capsule_K0_F1_mesh.apply_transform(T_point_z)
comp_dict["capsule_K0_F1"] = capsule_K0_F1_mesh

# # Capsule_K0 flattened
# capsule_K0_F1_cp = capsule_K0.cp.copy()
# capsule_K0_F1_cp[:, :, 2] = np.clip(capsule_K0_F1_cp[:, :, 2], -FLATTENED_THICKNESS / 2, FLATTENED_THICKNESS / 2)
# capsule_K0_F1_surf = make_surface(capsule_K0_F1_cp)
# capsule_K0_F1_mesh = make_mesh(capsule_K0_F1_surf, uu, vv)
# capsule_K0_F1_mesh.apply_transform(T_point_z)
# comp_dict["capsule_K0_F1"] = capsule_K0_F1_mesh

# # Capsule_K0 flattened and rotated
# T_rot = np.eye(4)
# T_rot[:3, :3] = Rotation.from_euler("xyz", np.array([0, 0, np.pi / 2])).as_matrix()
# capsule_K0_F2_mesh = capsule_K0_F1_mesh.copy()
# capsule_K0_F2_mesh.apply_transform(T_rot)
# comp_dict["capsule_K0_F2"] = capsule_K0_F2_mesh

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

# Capsule K1 flattened
capsule_K1_F1_cp = construct_cp_from_cs_func(
    construct_rounded_cs, capsule_K1, NUM_CS, FLATTENED_THICKNESS, 2 * CAPSULE_RADIUS, 0
)
capsule_K1_F1_surf = make_surface(capsule_K1_F1_cp)
capsule_K1_F1_mesh = make_mesh(capsule_K1_F1_surf, uu, vv)
capsule_K1_F1_mesh.apply_transform(T_point_z)
comp_dict["capsule_K1_F1"] = capsule_K1_F1_mesh


# # Capsule_K1 flattened
# capsule_K1_F1_cp = capsule_K1.cp.copy()
# capsule_K1_F1_cp = clip_along_axis(capsule_K1_F1_cp, 0, FLATTENED_THICKNESS)
# capsule_K1_F1_surf = make_surface(capsule_K1_F1_cp)
# capsule_K1_F1_mesh = make_mesh(capsule_K1_F1_surf, uu, vv)
# capsule_K1_F1_mesh.apply_transform(T_point_z)
# comp_dict["capsule_K1_F1"] = capsule_K1_F1_mesh

# # Capsule_K1 flattened
# capsule_K1_F2_cp = capsule_K1.cp.copy()
# capsule_K1_F2_cp = clip_along_axis(capsule_K1_F2_cp, 2, FLATTENED_THICKNESS)
# capsule_K1_F2_surf = make_surface(capsule_K1_F2_cp)
# capsule_K1_F2_mesh = make_mesh(capsule_K1_F2_surf, uu, vv)
# capsule_K1_F2_mesh.apply_transform(T_point_z)
# comp_dict["capsule_K1_F2"] = capsule_K1_F2_mesh


capsule_K1_F2_K_cp = construct_cp_from_cs_func(
    construct_rounded_cs, capsule_K1, NUM_CS, FLATTENED_THICKNESS, 2 * CAPSULE_RADIUS, np.pi / 2
)
capsule_K1_F2_K_surf = make_surface(capsule_K1_F2_K_cp)
capsule_K1_F2_K = make_mesh(capsule_K1_F2_K_surf, 75, 75)

capsule_K1_F2_K = capsule_K1_F2_K.apply_transform(T_point_z)
comp_dict["capsule_K1_F2_K"] = capsule_K1_F2_K
