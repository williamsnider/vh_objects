# Make axial component

from vh_objects.axial_component import AxialComponent
from vh_objects.cross_section import CrossSection
from vh_objects.shape import Shape
from vh_objects.backbone import Backbone
import numpy as np
from pathlib import Path
from compas_cgal.booleans import booleans
import trimesh
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np


def combine_meshes(mesh1, mesh2, operation="union"):

    # Use compas/CGAL to calculate boolean operation
    VA = mesh1.vertices.__array__()
    FA = mesh1.faces.__array__().astype("int32")
    VB = mesh2.vertices.__array__()
    FB = mesh2.faces.__array__().astype("int32")
    mesh_B = [mesh2.vertices.tolist(), mesh2.faces.tolist()]
    if operation == "union":
        mesh_C = booleans.boolean_union(VA, FA, VB, FB)
    elif operation == "difference":
        mesh_C = booleans.boolean_difference(VA, FA, VB, FB)
    else:
        raise NotImplementedError("Boolean operation must be 'union' or 'difference'.")

    # Get edges - vertices that were in neither initial mesh
    set_A = set([tuple(l) for l in VA])
    set_B = set([tuple(l) for l in VB])
    set_C = set([tuple(l) for l in mesh_C[0]])
    new_verts = (set_C - set_A) - set_B
    edge_verts_pts = np.zeros((len(new_verts), 3))
    for i, v in enumerate(new_verts):
        edge_verts_pts[i] = list(v)

    # Return as trimesh - easier to work with
    mesh = trimesh.Trimesh(
        vertices=mesh_C[0],
        faces=mesh_C[1],
    )

    # Identify which mesh each vertex in C comes from
    mesh_C_verts = mesh_C[0]
    mesh_C_verts_origin = np.zeros((len(mesh_C_verts))).astype("int")
    for i, v in enumerate(mesh_C_verts):
        if tuple(v) in set_A:
            mesh_C_verts_origin[i] = 1
        elif tuple(v) in set_B:
            mesh_C_verts_origin[i] = 2
        else:
            mesh_C_verts_origin[i] = 3

    # Use mesh_C_verts_origin as face map
    face_origin = np.zeros(mesh.faces.shape).astype("int")

    for r in range(mesh.faces.shape[0]):
        for c in range(mesh.faces.shape[1]):
            face_origin[r, c] = mesh_C_verts_origin[mesh.faces[r, c]]

    group1 = np.sum(face_origin == 1, axis=1) >= 1
    group2 = ~group1

    return mesh, group1, group2


c = np.cos
s = np.sin
num_cp = 8
r = 10

plt.rcParams.update(
    {
        "font.family": "Arial",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1,
        "lines.markersize": 10,
    }
)
sb_hex = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#D55E00",
    "purple": "#CC78BC",
    "brown": "#CA9161",
    "pink": "#FBAFE4",
    "gray": "#949494",
    "yellow": "#ECE133",
    "sky_blue": "#56B4E9",
}

import matplotlib.colors as mcolors

sb_rgb = {name: tuple(int(c * 255) for c in mcolors.to_rgb(hex_code)) for name, hex_code in sb_hex.items()}


# def save_fig(fig, path, ):


# color_m1 = [255, 0, 0, 255]
# color_m2 = [0, 0, 255, 255]
# color_j = [255, 0, 255, 255]
# color_f = [0, 255, 0, 255]

color_m1 = sb_rgb["green"] + (255,)
color_m2 = sb_rgb["sky_blue"] + (255,)
color_j = sb_rgb["purple"] + (255,)
color_f = sb_rgb["yellow"] + (255,)


linewidth_b = 5
markersize_cp = 10
figsize_half = (2.5, 2.5)
figsize_full = (5, 12)

#########################
### Smooth transition ###
#########################
cs_list = []

th_list = np.linspace(0, 2 * np.pi, num_cp, endpoint=False)
base_cp = np.array([np.cos(th_list), np.sin(th_list)]).T

cp0 = np.array([r * np.cos(th_list), r * np.sin(th_list)]).T
cs0 = CrossSection(cp0, 0.15)
cs_list.append(cs0)

# Ellipse
th_list = np.linspace(0, 2 * np.pi, num_cp, endpoint=False)
cp = np.array([np.cos(th_list), np.sin(th_list)]).T
r_list = [[1 / 2 * r, 3 / 2 * r] for _ in range(num_cp)]
r_list = np.array(r_list)
cp *= r_list
cs = CrossSection(cp, 0.85)
cs_list.append(cs)

# Create base backbone
cp = np.array(
    [
        [0, 0, 0],
        [10, 0, 0],
        [20, 0, 0],
        [30, 0, 0],
        [40, 0, 0],
        [50, 0, 0],
        [60, 0, 0],
    ]
)
backbone1 = Backbone(cp, reparameterize=True)


ac = AxialComponent(backbone1, cross_sections=cs_list)
# ac.mesh.show()
m_ac = ac.mesh


xlims = [-20, 20]
ylims = xlims

fig, ax = plt.subplots(1, 1, figsize=figsize_half)
cs = cs_list[0]
pts = cs.controlpoints
pts = np.vstack((pts, pts[0]))
ax.plot(pts[:, 0], pts[:, 1], "o--", color=sb_hex["blue"], markersize=markersize_cp)
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
fig.set_size_inches(figsize_half)
fig.savefig("cs1.svg")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=figsize_half)
cs = cs_list[1]
pts = cs.controlpoints
pts = np.vstack((pts, pts[0]))
ax.plot(pts[:, 0], pts[:, 1], "o--", color=sb_hex["orange"], markersize=markersize_cp)
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
fig.set_size_inches(figsize_half)
fig.savefig("cs2.svg")
plt.show()


def adjust_ax(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # ax.grid(False)
    # ax.axis("off")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_axis_off()
    ax.view_init(elev=11, azim=-58)
    return ax


## # Plot black
## pts = ac.controlpoints[1]
## pts = np.vstack((pts, pts[0]))
## ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "k--", markersize=10)

## pts = ac.controlpoints[-2]
## pts = np.vstack((pts, pts[0]))
## ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "k--", markersize=10)

# Plot backbone and cross sections in 3D
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111, projection="3d")
t = np.linspace(0, 1, 100)
pts_b = ac.backbone.r(t)

x_limits = [np.min(pts_b[:, 0]), np.max(pts_b[:, 0])]
y_limits = [np.min(pts_b[:, 1]), np.max(pts_b[:, 1])]
z_limits = [np.min(pts_b[:, 2]), np.max(pts_b[:, 2])]


# Plot backbone
ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], "k", linewidth=linewidth_b)

# Plot first
pts = ac.controlpoints[2]
pts = np.vstack((pts, pts[0]))
ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color=sb_hex["blue"], markersize=markersize_cp)
# pt = ac.backbone.r(0.15)[0]
# ax.plot(pt[0], pt[1], pt[2], "g.", markersize=10)


# Plot Second
pts = ac.controlpoints[-3]
pts = np.vstack((pts, pts[0]))
ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color=sb_hex["orange"], markersize=markersize_cp)
# pt = ac.backbone.r(0.85)[0]
# ax.plot(pt[0], pt[1], pt[2], "b.", markersize=10)

for pts in ac.controlpoints:
    pts = np.vstack((pts, pts[0]))
    # ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "b")

    x_limits.extend([np.min(pts[:, 0]), np.max(pts[:, 0])])
    y_limits.extend([np.min(pts[:, 1]), np.max(pts[:, 1])])
    z_limits.extend([np.min(pts[:, 2]), np.max(pts[:, 2])])

# Compute overall limits
x_min, x_max = min(x_limits), max(x_limits)
y_min, y_max = min(y_limits), max(y_limits)
z_min, z_max = min(z_limits), max(z_limits)

# Create a uniform scaling box
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
mid_x = (x_max + x_min) / 2.0
mid_y = (y_max + y_min) / 2.0
mid_z = (z_max + z_min) / 2.0

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax = adjust_ax(ax)
plt.tight_layout()
fig.set_size_inches(figsize_full)
fig.savefig("ac.svg")
plt.show()


# Show controlpoint grid
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot in increasing shades of gray
c_scale = np.linspace(0.8, 0.0, ac.controlpoints.shape[1])
idx_list = np.arange(ac.controlpoints.shape[1])
idx_list = np.roll(idx_list, -2)
c_scale = np.roll(c_scale, -2)
for j in idx_list[::-1]:
    pts = ac.controlpoints[:, j]
    # pts = np.vstack((pts, pts[0]))
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=(c_scale[j], c_scale[j], c_scale[j]), linewidth=2)

    # # Plot control points
    # ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "ko", markersize=markersize_cp)
# for j in range(ac.controlpoints.shape[1]):
#     pts = ac.controlpoints[:, j]
#     # pts = np.vstack((pts, pts[0]))
#     ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "k-")


# Plot black
pts = ac.controlpoints[1]
pts = np.vstack((pts, pts[0]))
ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=sb_hex["green"], linewidth=4)

pts = ac.controlpoints[-2]
pts = np.vstack((pts, pts[0]))
ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=sb_hex["green"], linewidth=4)

# Plot green
pts = ac.controlpoints[2]
pts = np.vstack((pts, pts[0]))
ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=sb_hex["blue"], linewidth=4)

# Plot blue
pts = ac.controlpoints[-3]
pts = np.vstack((pts, pts[0]))
ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=sb_hex["orange"], linewidth=4)


ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax = adjust_ax(ax)
plt.tight_layout()
fig.set_size_inches(figsize_full)
fig.savefig("ac_cp.svg")
plt.show()

######################
### Arc'd Backbone ###
######################

from vh_objects.shaft import Shaft

r = 10
shaft = Shaft(
    length=100,
    r1=r,
    r2=r,
    r3=r,
    theta=np.pi,
    lengthtype="two_hemi",
    num_cs=10,
    num_cp_per_cs=8,
    UU=300,
    VV=300,
)
# shaft.mesh.show()
m_shaft = shaft.mesh

# Plot backbone and cross sections
t = np.linspace(0, 1, 100)
pts_b = shaft.backbone.r(t)

pts_cp = shaft.cp

# Plot 3d points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], "k", linewidth=4)

# x_limits = [np.min(pts_b[:, 0]), np.max(pts_b[:, 0])]
# y_limits = [np.min(pts_b[:, 1]), np.max(pts_b[:, 1])]


singles = np.arange(num_cp)
pairs = [[s, (s + 1) % num_cp] for s in singles]
for pair_num, pair in enumerate(pairs[::-1]):
    for cp in pts_cp:
        pts = cp
        pts = np.vstack((pts, pts[0]))
        # ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=sb_hex["sky_blue"], linewidth=3)

        ax.plot(pts[pair, 0], pts[pair, 1], pts[pair, 2], color=sb_hex["sky_blue"], linewidth=3)

        x_limits.extend([np.min(pts[:, 0]), np.max(pts[:, 0])])
        y_limits.extend([np.min(pts[:, 1]), np.max(pts[:, 1])])
        z_limits.extend([np.min(pts[:, 2]), np.max(pts[:, 2])])

    if pair_num == num_cp // 2:
        ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], "k", linewidth=4)


# Compute overall limits
x_min, x_max = min(x_limits), max(x_limits)
y_min, y_max = min(y_limits), max(y_limits)
z_min, z_max = min(z_limits), max(z_limits)

# Create a uniform scaling box
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
mid_x = (x_max + x_min) / 2.0
mid_y = (y_max + y_min) / 2.0
mid_z = (z_max + z_min) / 2.0

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax = adjust_ax(ax)
ax.view_init(elev=109, azim=179, roll=0)
fig.set_size_inches((10, 10))
fig.savefig("shaft.svg")
plt.show()


import trimesh
import numpy as np


def add_contours_to_mesh(mesh, num_lines=10):
    # Define slicing planes (x-coordinates) for YZ plane slicing
    x_values = np.linspace(mesh.bounds[0, 0], mesh.bounds[1, 0], num_lines)

    # Create a scene
    scene = trimesh.Scene()
    scene.add_geometry(mesh)  # Add original mesh

    # Collect contour lines
    for x in x_values:
        # Define a slicing plane in YZ plane
        plane_origin = np.array([x, 0, 0])  # Varying x, keeping y and z constant
        plane_normal = np.array([1, 0, 0])  # Slicing perpendicular to X-axis

        # Intersect the mesh with the plane
        slice_result = mesh.section(plane_normal, plane_origin)

        if slice_result:
            scene.add_geometry(slice_result)  # Directly add Path3D to scene

    return scene


##################
### Fuse Above ###
##################
import trimesh


# Show both individually
m_shaft.apply_translation(-m_shaft.centroid)
m_shaft.visual.face_colors = color_m2
scene = trimesh.Scene([m_shaft])
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
scene.show(
    smooth=False,
)

# m_ac.apply_translation(-m_ac.centroid)
m_ac.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
m_ac.visual.face_colors = color_m1
scene = trimesh.Scene([m_ac])
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
scene.show(
    smooth=False,
)
# Rotate Z_about


# Transform to origin
m_shaft.apply_translation(-m_shaft.centroid)

# Transform fully down
m_shaft.apply_translation([-m_shaft.bounds[-1][0] + r, 0, 0])


# Transform copy for alignment with svg above
m_ac2 = m_ac.copy()
scene = add_contours_to_mesh(m_ac2)
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
scene.show()
m_ac2.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1]))
m_ac2.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 8, [1, 0, 0]))
m_ac2.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 8, [0, 1, 0]))


scene = trimesh.Scene([m_ac2])
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
scene.show(
    smooth=False,
)


# scene = trimesh.Scene([m_ac, m_shaft])
# scene.show()


# Fuse Mesh
from vh_objects.utilities import calc_mesh_boolean_and_edges, find_neighbors, fair_mesh

mesh1 = m_ac.copy()
mesh2 = m_shaft.copy()
operation = "union"
union_mesh, edge_verts_indices = calc_mesh_boolean_and_edges(mesh1, mesh2, operation)


# Fair mesh
HARMONIC_POWER = 2
fairing_distance = 4
edge_neighbors = find_neighbors(union_mesh, edge_verts_indices, distance=fairing_distance)
all_neighbors = edge_neighbors  # +add_verts_neighbors
faired_mesh = fair_mesh(union_mesh, all_neighbors, HARMONIC_POWER)
# faired_mesh.show(smooth=False)


m1 = m_ac.copy()
m2 = m_shaft.copy()

# Not joined
mesh, group1, group2 = combine_meshes(m1, m2, operation="union")
mesh.visual.face_colors[group1] = color_m1
mesh.visual.face_colors[group2] = color_m2
scene = trimesh.Scene([mesh])
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
print("Not Joined")
scene.show(
    smooth=False,
)

# Joined
mesh.visual.face_colors = color_j
scene = trimesh.Scene([mesh])
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
print("Joined")
scene.show(
    smooth=False,
)

# Joined with region-to-fair
v_list = all_neighbors
f_map = np.zeros(mesh.faces.shape)
for r in range(mesh.faces.shape[0]):
    for c in range(mesh.faces.shape[1]):
        f_map[r, c] = 1 if mesh.faces[r, c] in v_list else 0
f_map = np.sum(f_map, axis=1) == 3
mesh.visual.face_colors[f_map] = color_f
scene = trimesh.Scene([mesh])
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
scene.show(
    smooth=False,
)

# Faired with both labelled
mesh = faired_mesh
mesh.visual.face_colors = color_j
mesh.visual.face_colors[f_map] = color_f
scene = trimesh.Scene([mesh])
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
scene.show(
    smooth=False,
)

# Faired with solid label
mesh = faired_mesh
mesh.visual.face_colors = color_j
scene = trimesh.Scene([mesh])
scene.set_camera(angles=(0.0, 0.0, -np.pi / 2), distance=150, center=(10, 0, 0))
scene.show(
    smooth=False,
)
