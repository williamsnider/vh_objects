# Make a backbone that has a smooth junction

from scripts.stimulus_set_B import app1, app2, app3, app4, thin, NUM_CS
from scripts.stimulus_set_params import X_WIDTH, SEGMENT_LENGTH
from scripts.sheets import plot_arr
from objects.utilities import make_surface, make_mesh, fuse_meshes, fair_mesh
import numpy as np
import trimesh
from scipy.spatial.transform.rotation import Rotation

fairing_distance = X_WIDTH
harmonic_power = 2
box_extents = 3 * np.array([X_WIDTH, X_WIDTH, X_WIDTH])
box_translation = np.array([SEGMENT_LENGTH + 3 * X_WIDTH / 4, 0, 0])

for app in [app1, app2, app3, app4]:
    # Find thinnest interior section
    cp = app.cp.copy()
    # dists = np.linalg.norm(cp[:, :, :2], axis=2).mean(axis=1)
    # idx = dists[NUM_CS : -NUM_CS * 3 // 2].argmin() + NUM_CS

    # # Make cylinder
    # cp[2:idx, :, :2] = cp[idx, :, :2]
    # # plot_arr(cp)

    # Make mesh
    surf = make_surface(cp)
    mesh1 = make_mesh(surf, 50, 50)
    mesh2 = mesh1.copy()

    # Fuse mesh1
    T = np.eye(4)
    T[:3, 3] = [SEGMENT_LENGTH + -X_WIDTH + 0.01, 0, -X_WIDTH]
    mesh1.apply_transform(T)
    fused1 = fuse_meshes(thin.mesh.copy(), mesh1.copy(), fairing_distance, "union")
    fused1.visual.vertex_colors = [125, 125, 125, 255]
    # Fuse mesh2
    T = np.eye(4)
    T[:3, 3] = [SEGMENT_LENGTH + -X_WIDTH + 0.01, 0, X_WIDTH]
    T[:3, :3] = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
    mesh2.apply_transform(T)
    fused2 = fuse_meshes(fused1.copy(), mesh2.copy(), fairing_distance, "union")
    fused2.visual.vertex_colors = [125, 125, 125, 255]

    box = trimesh.primitives.creation.box(extents=box_extents)

    box.apply_translation(box_translation)
    box.visual.vertex_colors = [255, 255, 0, 50]

    neighbors = np.arange(fused1.vertices.shape[0])[box.contains(fused1.vertices)]
    faired1 = fair_mesh(fused1, neighbors, harmonic_power)

    # Fair everything beyond the end
    neighbors = np.arange(fused2.vertices.shape[0])[box.contains(fused2.vertices)]
    faired2 = fair_mesh(fused2, neighbors, harmonic_power)

    faired1.show(smooth=False)
    faired2.show(smooth=False)
# scene = trimesh.Scene()

# # scene.add_geometry([thin.mesh, mesh1, mesh2, box])
# scene.add_geometry([fused2, box])
# scene.show(smooth=False)
