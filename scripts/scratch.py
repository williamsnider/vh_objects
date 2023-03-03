import trimesh
import objects.utilities

# Load mesh
mesh = trimesh.load("test.stl")

# Analyze curvature
k1, k2 = objects.utilities.calc_mesh_principal_curvatures(mesh)

k1_faces = k1[mesh.faces].mean(axis=1)
k2_faces = k2[mesh.faces].mean(axis=1)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "red"])
face_min = np.min([k1_faces.min(), k2_faces.min()])
face_max = np.max([k1_faces.max(), k2_faces.max()])
c1 = (k1_faces - k1_faces.min()) / (k1_faces.max() - k1_faces.min())
c2 = (k2_faces - k2_faces.min()) / (k2_faces.max() - k2_faces.min())
cmap1 = cmap(c1)
cmap2 = cmap(c2)

mesh = mesh.copy()
mesh.visual.face_colors = cmap1

# mesh.show()


import igl
import scipy as sp
import numpy as np
from meshplot import plot, subplot, interact

import os

root_folder = os.getcwd()
v, f = igl.read_triangle_mesh("test.stl")
k = igl.gaussian_curvature(v, f)
my_plot = plot(v, f, k)
my_plot.save("my_plot.html")

v1, v2, k1, k2 = igl.principal_curvature(v, f)
h2 = 0.5 * (k1 + k2)
p = plot(v, f, h2, shading={"wireframe": False}, return_plot=True)

avg = igl.avg_edge_length(v, f) / 2.0
p.add_lines(v + v1 * avg, v - v1 * avg, shading={"line_color": "red"})
p.add_lines(v + v2 * avg, v - v2 * avg, shading={"line_color": "green"})
