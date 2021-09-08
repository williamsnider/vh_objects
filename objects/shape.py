from copy import Error
import trimesh
import numpy as np
from objects.parameters import SAMPLING_DENSITY_V, SAMPLING_DENSITY_U, ORDER
from objects.utilities import (
    open_uniform_knot_vector,
    calc_face_normals,
    plot_mesh_vertices_and_normals,
    plot_child_and_junction_edges,
)
import scipy
import networkx as nx
from splipy import BSplineBasis, Curve, Surface
from objects.utilities import (
    remove_subsurface_from_mesh,
    plot_projected_vertices_and_NNs,
    plot_projected_vertices_and_NNs_3D,
    plot_mesh_derivatives,
    plot_surface_linking_axial_components,
)
import matplotlib.pyplot as plt


class Shape:
    def __init__(self, ac_list):

        self.ac_list = ac_list

    def check_inputs(self):

        assert type(self.ac_list) is list, "ac_list must be a list, even if it has just 1 ac."

    def fuse_meshes(self, parent_ac, child_ac):

        parent_mesh = parent_ac.mesh

        # Define functions to carry out fusion in steps

        def find_join_slice_along_child(parent_mesh, child_ac, num_steps_long_axis, num_steps_round_axis):
            """Find slice along the child that is just outside the parent_ac."""

            # TODO: Do these on a simpler mesh to go faster

            # num_steps_long_axis = 10  # Try 10 different slices (increasing distance from parent_mesh)
            # num_steps_round_axis = (
            #     6  # Test if N points along slice are outside mesh
            # )
            (us, vs) = child_ac.surface.start()
            (ue, ve) = child_ac.surface.end()
            u = np.linspace(us, ue, num_steps_round_axis, endpoint=False)
            v = np.linspace(vs, ve, num_steps_long_axis)
            verts_array = child_ac.surface(u, v)

            for slice_num, _ in enumerate(v):
                print(slice_num)
                points = verts_array[:, slice_num, :]

                points_outside_mesh = ~parent_mesh.contains(points)
                if np.all(points_outside_mesh):  # All points outside
                    break

                if slice_num == len(v) - 1:
                    raise NotImplementedError

            INCREASE_FACTOR = 0.1  # Move the connection even further up to make it smoother
            slice_num += np.round(INCREASE_FACTOR * num_steps_long_axis).astype("int")
            slice_dist_approx = np.array(v[slice_num])  # Need to round this up to next highest value in actual v

            # Grab the full-size slice
            uu = SAMPLING_DENSITY_U
            vv = SAMPLING_DENSITY_V
            (us, vs) = child_ac.surface.start()
            (ue, ve) = child_ac.surface.end()
            v = np.linspace(vs, ve, vv)
            u = np.linspace(us, ue, uu, endpoint=False)
            slice_dist = v[v > slice_dist_approx][0]  # First value of v > slice_dist_approx
            full_slice = child_ac.surface(u, slice_dist)
            return full_slice, slice_dist, u, slice_dist_approx

        def project_child_slice_onto_parent_mesh(parent_mesh, child_ac, slice_dist):
            ### Expand this slice and project it onto the surface of the parent_mesh.

            TNB_current = np.stack(
                [
                    child_ac.T(slice_dist)[0],
                    child_ac.N(slice_dist)[0],
                    child_ac.B(slice_dist)[0],
                ],
                axis=0,
            )

            TNB_goal = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )

            R = np.linalg.inv(TNB_current) @ TNB_goal

            EXPANSION_FACTOR = 1.5
            center = child_ac.r(slice_dist)
            full_slice_rotated = ((full_slice - center) @ R * EXPANSION_FACTOR) + center
            mesh_verts_rotated = (parent_mesh.vertices - center) @ R + center

            # Remove x-axis
            full_slice_yz = np.squeeze(full_slice_rotated[:, :, 1:])
            mesh_verts_yz = mesh_verts_rotated[:, 1:]
            self.full_slice_yz = full_slice_yz
            self.mesh_verts_yz = mesh_verts_yz

            # Identify the 5 nearest neighbors for each point on the slice
            NUM_NN = 50
            tree = scipy.spatial.KDTree(mesh_verts_yz)
            dd, ii = tree.query(full_slice_yz, k=NUM_NN)

            # Choose the NN with the shortest distance in 3D
            mesh_points = mesh_verts_rotated[ii]
            slice_points = np.repeat(full_slice_rotated, NUM_NN, axis=1)
            dist = np.sqrt(np.sum((mesh_points - slice_points) ** 2, axis=2))
            min_idx = np.argmin(dist, axis=1)
            closest_NN = np.zeros((len(ii)), dtype="int")
            for i, _ in enumerate(closest_NN):
                closest_NN[i] = ii[i, min_idx[i]]

            _, unique_idx = np.unique(closest_NN, return_index=True)
            unique_NN = [closest_NN[i] for i in sorted(unique_idx)]
            unique_NN.append(unique_NN[0])  # Wrap starting point

            return unique_NN, closest_NN

        def find_path_between_projected_vertices(parent_mesh, unique_NN, closest_NN):
            # edges without duplication
            edges = parent_mesh.edges_unique

            # the actual length of each unique edge
            length = parent_mesh.edges_unique_length

            # create the graph with edge attributes for length
            g = nx.Graph()
            for edge, L in zip(edges, length):
                g.add_edge(*edge, length=L)

            # alternative method for weighted graph creation
            # you can also create the graph with from_edgelist and
            # a list comprehension, which is like 1.5x faster
            ga = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges, length)])

            # arbitrary indices of mesh.vertices to test with
            full_path = []
            for i in range(len(unique_NN)):
                print(i)
                start = unique_NN[i - 1]
                end = unique_NN[i]

                # run the shortest path query using length for edge weight
                new_path = nx.shortest_path(g, source=start, target=end, weight="length")

                # If this path overlaps previous path, remove the previous path
                repeated_verts = [p for p in new_path[1:] if p in full_path]
                if repeated_verts:

                    # Find the intermediate vertices (between the repeated) and cut them out

                    # Find the index of the last repeat in the new path
                    idx_path_repeat = -1
                    for r in repeated_verts:
                        idx = new_path.index(r)
                        if idx > idx_path_repeat:
                            idx_path_repeat = idx

                    # Find the index of the first repeat in the full path
                    idx_full_repeat = full_path.index(repeated_verts[0])  # Initilize
                    for r in repeated_verts:
                        idx = full_path.index(r)
                        if idx < idx_full_repeat:
                            idx_full_repeat = idx

                    # Get the vertices to be appended (skip the repeating ones)
                    try:
                        verts_to_append = new_path[idx_path_repeat + 1 :]
                    except:
                        verts_to_append = []

                    # Get the vertices for which we need to replace the assigned NN
                    NNs_to_replace = full_path[idx_full_repeat + 1 :] + new_path[:idx_path_repeat]
                    NN_replacement = full_path[idx_full_repeat]  # Just assign all

                    # Replace these NNs for the closest_NN array
                    for NN in NNs_to_replace:

                        closest_NN[closest_NN == NN] = NN_replacement
                else:
                    verts_to_append = new_path[1:]  # Don't duplicate first vert

                # Append to the full_path
                full_path.extend(verts_to_append)

            full_path.append(full_path[0])  # Add first element to end to close loop
            return full_path

        def create_surface_between_child_slice_and_parent_mesh(parent_mesh, child_ac, closest_NN, u):

            # Identify derivatives at points along parent_mesh. mesh
            # TODO: Use something smoother than p_V because this causes ripples in the junction mesh
            p_V = parent_mesh.vertices[closest_NN]
            p_T = parent_mesh.vertices[closest_NN] - parent_mesh.vertices[np.roll(closest_NN, -1)]

            # Replace 0 values of tangent with last nonzero value
            for i, row in enumerate(p_T):

                if np.all(row == np.array([0.0, 0.0, 0.0])):

                    p_T[i] = p_T[i - 1]

            p_N = parent_mesh.vertex_normals[closest_NN]
            p_B = np.cross(p_N, p_T)
            p_B = p_B / np.linalg.norm(p_B, axis=1, keepdims=True)  # norm

            # Identify derivatives at points along child's full slice
            uuu = u
            c_V = child_ac.surface(uuu, slice_dist).squeeze()
            c_T = child_ac.surface.derivative(uuu, slice_dist, d=(0, 1)).squeeze()
            c_T = c_T / np.linalg.norm(c_T, axis=1, keepdims=True)

            # Create B-Spline Surface linking child slice and projection on parent

            # Inputs
            degree = ORDER - 1

            # Basis 1 - cross section
            # With >100 controlpoints, the curve essentially passes through the points, so when we go to switch this segment in, if we skip the first and last elements, I think it will work.
            num_cp_per_cross_section = c_V.shape[0]
            num_knots = num_cp_per_cross_section + ORDER + degree
            knot = np.linspace(0, 1, num_knots)
            basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)

            curve = Curve(basis1, c_V, rational=False)

            # Basis 2 - along the major axis of the axial component
            num_rows = 4  # End termini + 2 intermediate points to determine slope
            knot = open_uniform_knot_vector(num_rows, ORDER)
            basis2 = BSplineBasis(order=ORDER, knots=knot, periodic=-1)

            # Controlpoints
            SCALE_FACTOR = 0.1
            cp = np.zeros([c_V.shape[0], 4, 3])
            cp[:, 0, :] = c_V
            cp[:, 1, :] = c_V - c_T * SCALE_FACTOR
            cp[:, 2, :] = p_V - p_B * SCALE_FACTOR
            cp[:, 3, :] = p_V
            # cp = cp.reshape(num_rows * num_cp_per_cross_section, cp.shape[2], order="F")
            cp = cp.reshape(num_rows * num_cp_per_cross_section, cp.shape[2], order="F")
            # XXX: Possibly above change caused problems

            # Surface
            surface = Surface(basis1, basis2, cp, rational=False)
            self.surface = surface

            return surface, c_V, c_T

        def stitch_child_and_junction(child_ac, surface, slice_dist, full_slice):

            # Sample surface of junction
            uu = SAMPLING_DENSITY_U
            vv = SAMPLING_DENSITY_V
            (us, vs) = surface.start()
            (ue, ve) = surface.end()
            # u = np.linspace(us, ue, uu, endpoint=False)
            u = np.linspace(
                ue, us, uu, endpoint=False
            )  # Reverse so that normals point outwards - Not 100% sure why this works.
            v = np.linspace(vs, ve, vv)
            junction_verts_array = surface(u, v)

            v_i = np.where(v == slice_dist)[0] - 1  # SHIFT since first row is deleted for endpoint
            full_slice_idx = np.arange(uu * v_i, uu * (v_i + 1))

            # # # Plot to figure out what's going on
            # fig = plt.figure()
            # ax = plt.axes(projection="3d")
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            # ax.view_init(elev=-90, azim=90)
            # for i, points in enumerate([np.squeeze(full_slice), child_ac.mesh.vertices[full_slice_idx]]):

            #     x, y, z = points.T

            #     if i == 0:
            #         ax.plot(x, y, z, "*", color="green")
            #     if i == 1:
            #         ax.plot(x, y, z, ".", color="red")

            assert np.all(np.isclose(np.squeeze(full_slice), child_ac.mesh.vertices[full_slice_idx]))

            # Get vertices of child (removing all of those before the full_slice)
            endpoint_00 = child_ac.verts.shape[0] - 2  # Index of endpoint at 0.0 position along child
            endpoint_10 = child_ac.verts.shape[0] - 1  # Index of endpoint at 1.0 position along child
            endpoint_of_subgraph_to_delete = endpoint_00
            child_mesh, full_slice_idx_new = remove_subsurface_from_mesh(
                child_ac.mesh, full_slice_idx, endpoint_of_subgraph_to_delete
            )

            # Fuse junction and child
            # Since there are the same number of vertices and they already overlap, we just need to renumber the vertices on the child.

            # Verts
            junction_verts = junction_verts_array.reshape(-1, 3, order="F")

            # Faces - CCW Winding (for consistent normals)
            # faces = np.zeros((uu * (vv - 2) * 2, 3), dtype="int")
            faces = np.zeros((uu * (vv - 1) * 2, 3), dtype="int")
            faces_array = np.zeros((uu * 2, vv - 1, 3), dtype="int")
            base_column = np.zeros((uu * 2, 3), dtype="int")
            base_column[::2, 0] = np.arange(0, uu)
            base_column[1::2, 0] = np.arange(0, uu)
            base_column[::2, 1] = np.arange(uu, uu * 2)
            base_column[1::2, 1] = np.arange(uu + 1, uu * 2 + 1)
            base_column[::2, 2] = np.arange(uu + 1, uu * 2 + 1)
            base_column[1:-1:2, 2] = np.arange(1, uu)
            base_column[-2, 2] = uu  # Fix wrapping
            base_column[-1, 1] = uu  # Fix wrapping
            base_column[:, 1:] = base_column[:, :-3:-1]  # Reverse for CCW winding

            # Grid faces
            for i in range(vv - 1):
                add_to_column = i * uu
                column = base_column + add_to_column
                start = uu * i * 2
                stop = uu * (i + 1) * 2
                faces[start:stop, :] = column
                faces_array[:, i, :] = column
            junction_faces = faces

            junction_face_normals = calc_face_normals(junction_verts, junction_faces)
            num_verts = uu * vv
            junction_vert_normals = trimesh.geometry.mean_vertex_normals(
                num_verts, junction_faces, junction_face_normals
            )
            junction_mesh = trimesh.Trimesh(
                vertices=junction_verts,
                faces=junction_faces,
                face_normals=junction_face_normals,
                vertex_normals=junction_vert_normals,
                process=False,
            )

            # Stitch junction and child meshes
            child_edge_idx = full_slice_idx_new
            junc_edge_child_idx = np.arange(uu * (0), uu * (1))  # TODO: Adjust for other end
            junc_edge_child_idx = junc_edge_child_idx[::-1]  # Flip order to align with child_mesh ordering
            junc_edge_child_idx = np.roll(junc_edge_child_idx, 1)  # TODO: Figure out why this shift is necessary
            junc_edge_parent_idx = np.arange(uu * (vv - 1), uu * (vv))  # Indices of vertices along parent

            # Plot edges
            # plot_child_and_junction_edges(child_mesh, child_edge_idx, junction_mesh, junc_edge_child_idx)

            # Shift junction edge vertices to align with child edge
            junction_mesh.vertices[junc_edge_child_idx] = child_mesh.vertices[child_edge_idx]

            # Increase idx of each junction vertex to account for child vertices
            num_child_verts = child_mesh.vertices.shape[0]
            num_edge_vertices = len(junc_edge_child_idx)
            junction_mesh.faces = junction_mesh.faces + num_child_verts - num_edge_vertices
            junc_edge_child_idx_shifted = junc_edge_child_idx + num_child_verts - num_edge_vertices
            junc_edge_parent_idx_shifted = junc_edge_parent_idx + num_child_verts - num_edge_vertices

            # Renumber the junction_edge vertices since they are already in the child vertices
            for i, junc_idx in enumerate(junc_edge_child_idx_shifted):

                old_idx = junc_idx
                new_idx = child_edge_idx[i]

                mask = junction_mesh.faces == old_idx
                junction_mesh.faces[mask] = new_idx

            # Remove the junction_edge vertices
            verts_to_keep = [i for i in np.arange(junction_mesh.vertices.shape[0]) if i not in junc_edge_child_idx]
            junction_mesh.vertices = junction_mesh.vertices[verts_to_keep]

            # Junction faces containing edge vertices should be renumbered to match child edge vertices
            combined_verts = np.concatenate([child_mesh.vertices, junction_mesh.vertices], axis=0)
            combined_faces = np.concatenate([child_mesh.faces, junction_mesh.faces])
            combined_face_norms = calc_face_normals(combined_verts, combined_faces)
            combined_vert_norms = trimesh.geometry.mean_vertex_normals(
                len(combined_verts), combined_faces, combined_face_norms
            )

            combined_mesh = trimesh.Trimesh(
                vertices=combined_verts,
                faces=combined_faces,
                face_normals=combined_face_norms,
                vertex_normals=combined_vert_norms,
                process=False,
            )

            combined_mesh.show()

            # Plot junction and child meshes
            # trimesh.Scene([child_mesh, junction_mesh]).show()

            return combined_mesh, junc_edge_parent_idx_shifted

        def remove_interior_vertices_from_parent(parent_mesh, full_path):

            # Strategy 1 - networkx
            parent_verts = parent_mesh.vertices

            # Find middle point (assume this is within region we want to cut out)
            centerpoint = parent_verts[full_path].mean(axis=0)

            # Find nearest neighbor of actual vertex to this centerpoint
            tree = scipy.spatial.KDTree(parent_verts)
            _, center_vert_idx = tree.query(centerpoint, k=1)

            mesh, full_path_new = remove_subsurface_from_mesh(parent_mesh, full_path, center_vert_idx)

            return mesh, full_path_new

        def stitch_parent_and_child(parent_mesh, parent_edge_vertices, combined_mesh, combined_mesh_edge_vertices):

            # Plot the two edges to make sure things look okay
            plot_child_and_junction_edges(
                parent_mesh, parent_edge_vertices, combined_mesh, combined_mesh_edge_vertices, plot_linkages=False
            )

            return None

        # Call the above functions to fuse the child and parent
        full_slice, slice_dist, u, slice_dist_approx = find_join_slice_along_child(
            parent_mesh,
            child_ac,
            num_steps_long_axis=10,
            num_steps_round_axis=6,
        )

        unique_NN, closest_NN = project_child_slice_onto_parent_mesh(parent_mesh, child_ac, slice_dist)

        full_path = find_path_between_projected_vertices(parent_mesh, unique_NN, closest_NN)
        plot_projected_vertices_and_NNs_3D(full_slice, closest_NN, parent_mesh.vertices, full_path)
        # plot_projected_vertices_and_NNs(self.full_slice_yz, closest_NN, self.mesh_verts_yz, full_path)

        surface, c_V, c_T = create_surface_between_child_slice_and_parent_mesh(parent_mesh, child_ac, closest_NN, u)
        combined_mesh, combined_mesh_edge_vertices = stitch_child_and_junction(
            child_ac, surface, slice_dist, full_slice
        )
        # plot_surface_linking_axial_components(parent_mesh, child_ac, surface)

        # Delete the hole in the parent mesh
        parent_mesh_new, full_path_new = remove_interior_vertices_from_parent(parent_mesh, full_path)

        new_mesh = stitch_parent_and_child(parent_mesh_new, full_path_new, combined_mesh, combined_mesh_edge_vertices)

        # Stitch together the two
        # For the smaller sequence, find the nearest neighbor of each vertex to points in the other sequence
        # Between these pairings, link points in the larger sequence to the first elmeent of each gap in the shorter sequence
        # Result is a list of edges which we will convert to faces

    def plot_meshes(self):

        trimesh.Scene([ac.mesh for ac in self.ac_list]).show()

    def merge_meshes(self):

        merged_meshes = trimesh.boolean.union([ac.mesh for ac in self.ac_list], engine="scad")
        bf = trimesh.repair.broken_faces(merged_meshes)
        self.merged_meshes = merged_meshes
