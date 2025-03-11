import trimesh
import numpy as np

# Load or create a mesh
mesh = trimesh.creation.icosphere(subdivisions=3)


def add_contours_to_mesh(mesh, num_lines=10):
    # Define slicing planes (z-coordinates)
    z_values = np.linspace(mesh.bounds[0, 2], mesh.bounds[1, 2], num_lines=10)

    # Create a scene
    scene = trimesh.Scene()
    scene.add_geometry(mesh)  # Add original mesh

    # Collect contour lines
    for z in z_values:
        # Define a slicing plane
        plane_origin = np.array([0, 0, z])
        plane_normal = np.array([0, 0, 1])  # Slicing parallel to XY-plane

        # Intersect the mesh with the plane
        slice_result = mesh.section(plane_normal, plane_origin)

        if slice_result:
            scene.add_geometry(slice_result)  # Directly add Path3D to scene

    return scene


scene = add_contours_to_mesh(mesh)
scene.show()
