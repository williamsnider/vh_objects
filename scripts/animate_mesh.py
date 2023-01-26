import numpy as np

import pyvista as pv
import scipy.spatial
import pickle

# Mesh
filename = "/home/oconnorlab/code/objects/sample_shapes/stimulus_set_A/pkl/shape_20.pkl"
with open(filename, "rb") as f:
    shape = pickle.load(f)
tri_mesh = shape.mesh_with_interface

# Shift center of mass to origin
tri_mesh.apply_translation(-shape.mesh.center_mass)

mesh = pv.wrap(shape.mesh_with_interface)
pts = mesh.points
# cpos = mesh.plot()

plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.add_mesh(
    mesh,
    lighting=True,
    show_edges=False,
)
plotter.open_gif("shape_rotation.gif")

# Write a frame. This triggers a render.
for th in np.linspace(0, 2 * np.pi, 200, endpoint=False):
    shape_euler = np.array([0, 0, th])
    R = scipy.spatial.transform.Rotation.from_euler("xyz", shape_euler).as_matrix()
    R_pts = pts @ R
    mesh.points = R_pts
    plotter.write_frame()

# Closes and finalizes movie
plotter.close()
