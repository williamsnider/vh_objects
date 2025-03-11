import trimesh
from pathlib import Path
import fast_simplification
import pyvista as pv


def trimesh_to_pyvista(trimesh_obj):
    """
    Converts a trimesh object to a PyVista PolyData mesh.

    Parameters:
        trimesh_obj (trimesh.Trimesh): The input trimesh object.

    Returns:
        pyvista.PolyData: The converted PyVista mesh.
    """
    vertices = trimesh_obj.vertices
    faces = trimesh_obj.faces

    # Add the number of vertices per face (3 for triangles) at the start of each face
    faces_with_count = [[3] + list(face) for face in faces]
    faces_flattened = [item for sublist in faces_with_count for item in sublist]

    # Create and return the PyVista mesh
    return pv.PolyData(vertices, faces_flattened)


s = trimesh.load_mesh(Path("/home/williamsnider/Code/vh_objects/sample_shapes_no_duplicates/stl/medial_axis/G071.stl"))
# s = trimesh.load_mesh(Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/demo/D000.stl"))
# Convert from trimesh to pyvista
# s_pyvista = trimesh_to_pyvista(s)
# s_pyvista.plot()
# s_pyvista.is_manifold

s_out = fast_simplification.simplify(s.vertices, s.faces, target_count=1000, agg=3)
s_out_tri = trimesh.Trimesh(s_out[0], s_out[1])
s_out_tri.show()

from pyvista import examples

mesh = examples.download_nefertiti()
out = fast_simplification.simplify_mesh(mesh, target_reduction=0.9)


# s_orig = s.copy()
# s_simp = s.copy()

# # Simplify mesh
# s_simp = s_simp.simplify_quadratic_decimation(0.5)


# # Show
# scene = trimesh.Scene()
# scene.add_geometry(s_orig)
# scene.add_geometry(s_simp)
# scene.show()
