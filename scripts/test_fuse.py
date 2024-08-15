import trimesh
from vh_objects.utilities import fuse_meshes

meshA = trimesh.primitives.Sphere(2, subdivisions=3)
meshB = meshA.copy()
meshB.apply_translation([0, 1, 0])

scene = trimesh.Scene()
scene.add_geometry(meshA)
scene.add_geometry(meshB)
scene.show()


mesh_C = fuse_meshes(meshA, meshB, 1, "union")
mesh_C.show(smooth=False)