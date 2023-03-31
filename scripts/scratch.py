import trimesh
from objects.utilities import fair_mesh
import numpy as np

# Base shape
mesh = trimesh.primitives.creation.icosphere(5)
num_verts = mesh.vertices.shape[0]

# Squash
thres = 0.5
above = mesh.vertices[:, 1] > thres
below = mesh.vertices[:, 1] < -thres

mesh.vertices[above, 1] = thres
mesh.vertices[below, 1] = -thres

# Expand to hit radius
outside_mask = above + below
inside_mask = ~outside_mask
RADIUS = 1.5
mesh.vertices[inside_mask] *= np.array([RADIUS, 1, RADIUS])

# Find radii at max length
dists = np.linalg.norm(mesh.vertices, axis=1)
edge_mask = dists > dists.max() - 0.001

# Fair
inside_not_edge = np.logical_xor(inside_mask, edge_mask)
inside_not_edge_indices = np.arange(0, num_verts)[inside_mask]
faired_mesh = fair_mesh(mesh, inside_not_edge_indices, 3)
faired_mesh.visual.vertex_colors[inside_not_edge] = [255, 0, 255, 255]
faired_mesh.show(smooth=False)
