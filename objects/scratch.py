import igl
import scipy as sp
import numpy as np
import os
import trimesh
import matplotlib.pyplot as plt

root_folder = os.getcwd()


v, f = igl.read_triangle_mesh(os.path.join(root_folder, "objects", "decimated-max.obj"))
v[:, [0, 2]] = v[:, [2, 0]]  # Swap X and Z axes
u = v.copy()


s = igl.read_dmat(os.path.join(root_folder, "objects", "decimated-max-selection.dmat"))
b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, v.shape[0])] if t[1] >= 0]]).T
b = np.arange(0, v.shape[0])[s == 2]


## Boundary conditions directly on deformed positions
u_bc = np.zeros((b.shape[0], v.shape[1]))
v_bc = np.zeros((b.shape[0], v.shape[1]))

for bi in range(b.shape[0]):
    v_bc[bi] = v[b[bi]]

    if s[b[bi]] == 0:  # Don't move handle 0
        u_bc[bi] = v[b[bi]]
    elif s[b[bi]] == 1:  # Move handle 1 down
        u_bc[bi] = v[b[bi]] + np.array([[0, -50, 0]])
    else:  # Move other handles forward
        u_bc[bi] = v[b[bi]] + np.array([[-25, 0, 0]])

i = 2
u_bc_anim = v_bc + i * 0.6 * (u_bc - v_bc)
d_bc = u_bc_anim - v_bc
d = igl.harmonic_weights(v, f, b, d_bc, 2)
u = v + d
# subplot(
#     u,
#     f,
#     s,
#     shading={"wireframe": False, "colormap": "tab10"},
#     s=[1, 4, i + 1],
#     data=p,
# )
mesh = trimesh.Trimesh(vertices=u, faces=f)
trimesh.repair.fix_normals(mesh)
mesh.visual.vertex_colors[b, 0] = 255
# mesh.show()


# Plot points
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=-90, azim=90)

# Entire mesh
x, y, z = mesh.vertices.T
ax.plot(x, y, z, ".", color="green")

# S Points
x, y, z = mesh.vertices[b].T
ax.plot(x, y, z, ".", color="red")

plt.show()


# # Create scene
# mesh.visual.vertex_colors[b, 0] = 255
# trimesh.Scene(mesh).show()

# @interact(deformation_field=True, step=(0.0, 2.0))
# def update(deformation_field, step=0.0):
#     # Determine boundary conditions
#     u_bc_anim = v_bc + step * (u_bc - v_bc)

#     if deformation_field:
#         d_bc = u_bc_anim - v_bc
#         d = igl.harmonic_weights(v, f, b, d_bc, 2)
#         u = v + d
#     else:
#         u = igl.harmonic_weights(v, f, b, u_bc_anim, 2)
#     p.update_object(vertices=u)
