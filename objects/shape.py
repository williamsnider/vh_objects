import trimesh
import numpy as np
import scipy
import matplotlib.pyplot as plt
from compas_cgal.booleans import boolean_union
import igl
from objects.utilities import plot_mesh_and_specific_indices
from objects.parameters import HARMONIC_POWER, FAIRING_DISTANCE


class Shape:
    def __init__(self, ac_list):

        self.ac_list = ac_list

    def check_inputs(self):

        assert type(self.ac_list) is list, "ac_list must be a list, even if it has just 1 ac."

    def fuse_meshes(self, parent_mesh, child_mesh):
        def calc_mesh_boolean_and_edges(mesh1, mesh2):

            # Use compas/CGAL to calculate boolean union
            mesh_A = [mesh1.vertices.tolist(), mesh1.faces.tolist()]
            mesh_B = [mesh2.vertices.tolist(), mesh2.faces.tolist()]
            mesh_C = boolean_union(mesh_A, mesh_B)

            # Get edges - vertices that were in neither initial mesh
            set_A = set([tuple(l) for l in mesh_A[0]])
            set_B = set([tuple(l) for l in mesh_B[0]])
            set_C = set([tuple(l) for l in mesh_C[0]])
            new_verts = (set_C - set_A) - set_B
            edge_verts_pts = np.zeros((len(new_verts), 3))
            for i, v in enumerate(new_verts):
                edge_verts_pts[i] = list(v)

            # Return as trimesh - easier to work with
            mesh = trimesh.Trimesh(
                vertices=mesh_C[0],
                faces=mesh_C[1],
            )

            # Get indices of edge_verts
            tree = scipy.spatial.KDTree(mesh.vertices)
            _, edge_verts_indices = tree.query(edge_verts_pts, k=1)

            return mesh, edge_verts_indices

        def find_neighbors(mesh, group, distance):

            mesh_pts = mesh.vertices.__array__()
            edge_pts = mesh.vertices[group].__array__()

            tree = scipy.spatial.KDTree(mesh_pts)
            neighbors_list = tree.query_ball_point(edge_pts, r=distance)
            neighbors = set()
            for n in neighbors_list:
                neighbors.update(set(n))

            return list(neighbors)

        def fair_mesh(union_mesh, neighbors):

            v = union_mesh.vertices.__array__()
            f = union_mesh.faces.__array__().astype("int32")
            num_verts = v.shape[0]
            b = np.array(list(set(range(num_verts)) - set(neighbors)))  # Bounday indices - NOT to be faired
            bc = v[b]  # XYZ coordinates of the boundary indices
            z = igl.harmonic_weights(v, f, b, bc, HARMONIC_POWER)  # Smooths indices at creases

            union_mesh.vertices = z
            faired_mesh = union_mesh

            return faired_mesh

        union_mesh, edge_verts_indices = calc_mesh_boolean_and_edges(parent_mesh, child_mesh)
        neighbors = find_neighbors(union_mesh, edge_verts_indices, distance=FAIRING_DISTANCE)
        union_mesh = fair_mesh(union_mesh, neighbors)
        union_mesh.show(smooth=False)

    def plot_meshes(self):

        trimesh.Scene([ac.mesh for ac in self.ac_list]).show()
