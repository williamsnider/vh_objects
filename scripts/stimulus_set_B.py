import numpy as np
from objects.utilities import (
    find_cp_for_desired_radius,
    approximate_arc,
    get_deformation_vertex,
    fuse_meshes,
    fair_mesh,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.backbone import Backbone
from objects.shape import Shape
import trimesh
import scipy
import scipy.spatial
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import copy

### Construct projections ###
NUM_CP_PER_CROSS_SECTION = 8
NUM_CP_PER_BACKBONE = 5
LENGTH = 20
RADIUS = 2

# Base
backbone_cp = np.hstack(
    [
        np.linspace(0, 10, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
backbone = Backbone(backbone_cp, reparameterize=True)
cp_radius = find_cp_for_desired_radius(RADIUS, NUM_CP_PER_CROSS_SECTION)
cs_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(
    -1, 1
)
cs_cp = np.hstack((cp_radius * np.cos(cs_th), cp_radius * np.sin(cs_th)))

cs_list = [
    CrossSection(controlpoints=1.01 * cs_cp, position=position)
    for position in np.linspace(0.0, 0.9, 6)
]
cs_list.append(CrossSection(controlpoints=1.01 * cs_cp, position=1.0, tilt=np.pi / 100))
p_base = AxialComponent(backbone, cs_list, smooth_with_post=False)
# p_base.mesh.show(smooth=False)

# Straight
backbone_cp = np.hstack(
    [
        np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
backbone = Backbone(backbone_cp, reparameterize=True)
cp_radius = find_cp_for_desired_radius(RADIUS, NUM_CP_PER_CROSS_SECTION)
cs_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(
    -1, 1
)
cs_cp = np.hstack((cp_radius * np.cos(cs_th), cp_radius * np.sin(cs_th)))

cs_list = [
    CrossSection(controlpoints=1.000 * cs_cp, position=position)
    for position in np.linspace(0.0, 0.9, 6)
]
p_straight = AxialComponent(
    backbone,
    cs_list,
    smooth_with_post=False,
    parent_axial_component=p_base,
    position_along_parent=0.9,
)
# p_straight.mesh.show(smooth=False)

# Sharp bend
backbone_cp = np.hstack(
    [
        np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
shift = backbone_cp - backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
R = scipy.spatial.transform.Rotation.from_euler(
    "xyz", np.array([0, 0, np.pi / 4])
).as_matrix()
shift_R = shift @ R
transformed = shift_R + backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
backbone_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[
    NUM_CP_PER_BACKBONE // 2 :: -1, :
]
backbone = Backbone(backbone_cp, reparameterize=True)
cs_list = []
cs_list.extend(
    [
        CrossSection(controlpoints=1.001 * cs_cp, position=position)
        for position in [0.0, 0.2, 0.25]
    ]
)
# Scale middle cross section to prevent tearing
# Alternative would be to place middle cross section where vectors of other controlpoints meet. Must handle how to center along backbone.
S = np.eye(2)
S[0, 0] = 2
cs_list.extend(
    [
        CrossSection(controlpoints=1.001 * cs_cp @ S, position=position)
        for position in [0.5]
    ]
)
cs_list.extend(
    [
        CrossSection(controlpoints=1.001 * cs_cp, position=position)
        for position in [0.75, 0.8, 0.9]
    ]
)
p_bend_sharp = AxialComponent(
    backbone, cs_list, parent_axial_component=p_base, position_along_parent=0.9
)
# p_bend_sharp.mesh.show(smooth=False)

# Half bend
backbone_cp = np.hstack(
    [
        np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
shift = backbone_cp - backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
R = scipy.spatial.transform.Rotation.from_euler(
    "xyz", np.array([0, 0, np.pi / 2])
).as_matrix()
shift_R = shift @ R
transformed = shift_R + backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
backbone_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[
    NUM_CP_PER_BACKBONE // 2 :: -1, :
]
backbone = Backbone(backbone_cp, reparameterize=True)
cs_list = [
    CrossSection(controlpoints=1.002 * cs_cp, position=position)
    for position in [0.0, 0.3, 0.40, 0.5, 0.60, 0.7, 0.9]
]
p_bend_half = AxialComponent(
    backbone, cs_list, parent_axial_component=p_base, position_along_parent=0.9
)
# p_bend_half.mesh.show(smooth=False)

# Weak bend
backbone_cp = np.hstack(
    [
        np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
shift = backbone_cp - backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
R = scipy.spatial.transform.Rotation.from_euler(
    "xyz", np.array([0, 0, 3 * np.pi / 4])
).as_matrix()
shift_R = shift @ R
transformed = shift_R + backbone_cp[NUM_CP_PER_BACKBONE // 2, :]
backbone_cp[NUM_CP_PER_BACKBONE // 2 :, :] = transformed[
    NUM_CP_PER_BACKBONE // 2 :: -1, :
]
backbone = Backbone(backbone_cp, reparameterize=True)
cs_list = [
    CrossSection(controlpoints=1.003 * cs_cp, position=position)
    for position in [0.0, 0.3, 0.40, 0.5, 0.60, 0.7, 0.9]
]
p_bend_weak = AxialComponent(
    backbone, cs_list, parent_axial_component=p_base, position_along_parent=0.9
)
# p_bend_weak.mesh.show(smooth=False)


# Arc
backbone_cp = approximate_arc(np.pi / 2, LENGTH)
backbone_cp *= np.array([1, -1, 1])  # Flip about y axis
backbone = Backbone(backbone_cp, reparameterize=True)
cs_list = [
    CrossSection(controlpoints=1.004 * cs_cp, position=position)
    for position in np.linspace(0.0, 0.9, 10)
]
p_arc = AxialComponent(
    backbone,
    cs_list,
    smooth_with_post=False,
    parent_axial_component=p_base,
    position_along_parent=0.9,
)
# p_arc.mesh.show(smooth=False)


### Shapes ###

# s1 = Shape([p_base, p_arc, p_bend_weak])
# s2 = Shape([p_base, p_bend_weak])
s3 = Shape([p_base, p_straight, p_arc])
# s3.mesh.show()


p_list = [None, p_straight, p_bend_sharp, p_bend_half, p_bend_weak, p_arc]
import itertools

p_dict = {
    "None": None,
    "p_straight": p_straight,
    "p_bend_sharp": p_bend_sharp,
    "p_bend_half": p_bend_half,
    "p_bend_weak": p_bend_weak,
    "p_arc": p_arc,
}
combs = list(itertools.combinations_with_replacement(p_dict.keys(), 2))
angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
for p1_name, p2_name in combs:

    p1 = p_dict[p1_name]
    p2 = p_dict[p2_name]

    if (p1 == None) and (p2 == None):
        continue
    elif p1 == None:
        print(p1_name, p2_name)
        s = Shape([p_base, p2])
    elif p2 == None:
        print(p1_name, p2_name)
        s = Shape([p_base, p1])
    else:

        # Rotate projections (exclude straight projection)
        if (p1_name == "p_straight") and (p2_name == "p_straight"):
            continue
        elif (p1_name == "p_straight") or (p2_name == "p_straight"):
            print(p1_name, p2_name)
            s = Shape([p_base, p1, p2])
        else:
            for ang in angles:

                # Skip duplicate of same projection at same angle
                if (p1_name == p2_name) and (ang == 0 or ang == angles[-1]):
                    continue

                # Apply rotation
                p1_rot = copy.deepcopy(p1)
                p1_rot.euler_angles = np.array([ang, 0, 0])

                # Recalculate axial component
                p1_rot.calc_points()

                # Apply small shift (to help with boolean union)
                p1_rot.mesh.vertices *= 1.001

                print(p1_name, p2_name, ang)
                s = Shape([p_base, p1_rot, p2])
                s.mesh.show()

    s.mesh.show()

# # Plot cross sections of shape
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# cp = p_arc.controlpoints
# for i in range(8):
#     xyz = cp[:, i, :]
#     ax.plot(
#         xyz[:, 0],
#         xyz[:, 1],
#         xyz[:, 2],
#         "k-*",
#     )
# plt.show()


# T = np.eye(4)
# T[:3, 3] = [6.5, 1, 0]
# box = trimesh.primitives.Box(extents=[5, 5, 5], transform=T)

# scene = trimesh.Scene()
# scene.add_geometry(box)
# scene.add_geometry(p_bend_sharp.mesh)
# scene.show()

# # Fair mesh
# neighbors = box.contains(p_bend_sharp.mesh.vertices)
# neighbors = np.arange(0, len(neighbors))[neighbors]
# faired_mesh = fair_mesh(p_bend_sharp.mesh, neighbors, 2)
# faired_mesh.show(smooth=False)


# fig = plt.figure()
# ax = plt.axes(projection="3d")
# t = np.linspace(0, 1, 100)
# xyz = backbone.r(t)
# ax.plot(
#     xyz[:, 0],
#     xyz[:, 1],
#     xyz[:, 2],
#     "k.",
# )
# plt.show()
