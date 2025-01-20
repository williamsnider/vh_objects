# # Create cube with many triangles

# import trimesh
# from vh_objects.utilities import fuse_meshes

# box = trimesh.creation.box(extents=[1, 1, 1])
# print(f"Num faces: {len(box.faces)}; is watertight: {box.is_watertight}")


# # Subdivide faces
# f = box.faces[0]
# f_verts = box.vertices[f]
# f_center = f_verts.mean(axis=0)

# # New faces
# new_faces = []
# for i in range(3):
#     new_faces.append([f_verts[i], f_center, f_verts[(i + 1) % 3]])


# ico = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
# ico.apply_translation(f_center)

# # Fuse meshes
# fused = fuse_meshes(box, ico, 0, "union")
# print(f"Num faces: {len(fused.faces)}; is watertight: {fused.is_watertight}")
# fused.show(smooth=False)


import trimesh
import numpy as np

# Create a box mesh
# box = trimesh.creation.box(extents=(1, 1, 1))


# def subdivide_mesh(mesh, subdivisions=1):
#     """
#     Subdivide a mesh by adding centroids to each face.

#     Parameters
#     ----------
#     mesh : trimesh.Trimesh
#         The input mesh to subdivide.
#     n_subdivisions : int
#         The number of subdivisions to perform.

#     Returns
#     -------
#     trimesh.Trimesh
#         The subdivided mesh.
#     """
#     # Initialize lists to hold the new faces and vertices
#     new_faces = []
#     vertices = mesh.vertices

#     # Iterate through each face of the mesh
#     for face in mesh.faces:
#         # Get the vertices of the current face
#         face_vertices = vertices[face]

#         # Compute the centroid of the face
#         centroid = face_vertices.mean(axis=0)

#         # Add the centroid to the vertex list
#         centroid_index = len(vertices)
#         vertices = np.vstack([vertices, centroid])

#         # Subdivide the face into triangles using the centroid
#         for i in range(len(face)):
#             new_faces.append([face[i], face[(i + 1) % len(face)], centroid_index])

#     # Create a new mesh with the updated vertices and faces
#     subdivided_mesh = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=True)

#     while subdivisions > 1:
#         return subdivide_mesh(subdivided_mesh, subdivisions - 1)

#     # Verify the mesh is watertight
#     assert subdivided_mesh.is_watertight, "The mesh is not watertight!"

#     return subdivided_mesh


# def load_box_subdivided(extents=[1, 1, 1], subdivisions=1):
#     box = trimesh.creation.box(extents=extents)
#     return subdivide_mesh(box, subdivisions)


def subdivide_mesh(mesh, subdivisions=1):
    # Initialize lists to hold the new faces and vertices
    new_faces = []
    vertices = mesh.vertices

    # Iterate through each face of the mesh
    for face in mesh.faces:
        # Get the vertices of the current face
        face_vertices = vertices[face]

        # Calculate edge midpoints
        edge_midpoints = [(face_vertices[i] + face_vertices[(i + 1) % len(face)]) / 2 for i in range(len(face))]

        # Find the longest edge and its midpoint
        edge_lengths = [np.linalg.norm(face_vertices[i] - face_vertices[(i + 1) % len(face)]) for i in range(len(face))]
        longest_edge_index = np.argmax(edge_lengths)
        longest_midpoint = edge_midpoints[longest_edge_index]

        # Add the midpoint of the longest edge to the vertex list
        midpoint_index = len(vertices)
        vertices = np.vstack([vertices, longest_midpoint])

        # Subdivide the face into right triangles using the midpoint
        for i in range(len(face)):

            # Skip the longest edge
            if i == longest_edge_index:
                continue

            idx1 = i
            idx2 = (i + 1) % len(face)
            new_faces.append([face[idx1], face[idx2], midpoint_index])

    # Create a new mesh with the updated vertices and faces
    subdivided_mesh = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=True)

    # Verify the mesh is watertight
    assert subdivided_mesh.is_watertight, "The mesh is not watertight!"

    while subdivisions > 1:
        return subdivide_mesh(subdivided_mesh, subdivisions - 1)

    return subdivided_mesh


def load_subdivided_box(extents=[1, 1, 1], subdivisions=4):

    # Create a box mesh
    box = trimesh.creation.box(extents=extents)

    subdivided_box = subdivide_mesh(box, subdivisions)

    return subdivided_box


if __name__ == "__main__":

    # subdivide_box = subdivide_mesh(box, 6)
    # print(f"Num faces: {len(subdivide_box.faces)}; is watertight: {subdivide_box.is_watertight}")

    import trimesh
    import numpy as np

    sub = load_subdivided_box([1, 1, 1], 10)
    print(f"Num faces: {len(sub.faces)}; is watertight: {sub.is_watertight}")


# # Initialize lists to hold the new faces and vertices
# new_faces = []
# vertices = box.vertices

# # Iterate through each face of the box
# for face in box.faces:
#     # Get the vertices of the current face
#     face_vertices = vertices[face]

#     # Compute the centroid of the face
#     centroid = face_vertices.mean(axis=0)

#     # Add the centroid to the vertex list
#     centroid_index = len(vertices)
#     vertices = np.vstack([vertices, centroid])

#     # Subdivide the face into triangles using the centroid
#     for i in range(len(face)):
#         new_faces.append([face[i], face[(i + 1) % len(face)], centroid_index])

# # Create a new mesh with the updated vertices and faces
# subdivided_box = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=True)

# # Verify the mesh is watertight
# assert subdivided_box.is_watertight, "The mesh is not watertight!"

# # Save or process the subdivided box as needed
# subdivided_box.show()
