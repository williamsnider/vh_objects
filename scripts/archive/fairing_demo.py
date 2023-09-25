import trimesh
import numpy as np
import io
from PIL import Image
from objects.utilities import (
    fuse_meshes,
    fair_mesh,
    calc_mesh_boolean_and_edges,
    find_neighbors,
)


def save_scene(mesh_list, fname):

    scene = trimesh.Scene()
    scene.add_geometry(mesh_list)
    data = scene.save_image(resolution=(1920, 1200), smooth=False)
    image = np.array(Image.open(io.BytesIO(data)))
    img = Image.fromarray(image)
    img.save(fname)


# Base sphere and ellipsoid
divisions = 7
sphere = trimesh.primitives.creation.icosphere(divisions)
sphere.apply_translation([0, 0.75, 0])
sphere.visual.vertex_colors = [255, 125, 0, 255]
ellipsoid = trimesh.primitives.creation.icosphere(divisions).apply_scale(
    [3.0, 1.0, 1.0]
)
ellipsoid.visual.vertex_colors = [125, 255, 0, 255]

# Fused sphere and ellipsoid
fused = fuse_meshes(sphere, ellipsoid, 0, "union")
fused.visual.vertex_colors = [125, 125, 255, 255]

# Mesh pre fairing
fairing_distance = 0.3
union_mesh, edge_verts_indices = calc_mesh_boolean_and_edges(sphere, ellipsoid, "union")
neighbors = find_neighbors(union_mesh, edge_verts_indices, distance=fairing_distance)
union_mesh.visual.vertex_colors = [125, 125, 255, 255]
union_mesh.visual.vertex_colors[neighbors] = [255, 255, 0, 255]

#
# Mesh fully faired visual 1
faired = fair_mesh(union_mesh.copy(), neighbors, 2)
faired_1 = faired.copy()
faired_1.visual.vertex_colors = [125, 125, 255, 255]
faired_1.visual.vertex_colors[neighbors] = [255, 255, 0, 255]

# Mesh fully faired visual 2
faired_2 = faired.copy()
faired_2.visual.vertex_colors = [125, 125, 255, 255]

save_scene([sphere, ellipsoid], "img1.png")
save_scene([fused], "img2.png")
save_scene([union_mesh], "img3.png")
save_scene([faired_1], "img4.png")
save_scene([faired_2], "img5.png")
