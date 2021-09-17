from copy import Error
from networkx.algorithms.core import k_core
from numpy.core.fromnumeric import _searchsorted_dispatcher
import trimesh
import numpy as np
from trimesh import parent
from objects.parameters import SAMPLING_DENSITY_V, SAMPLING_DENSITY_U, ORDER, NUM_NN, WINDOW_SIZE
from objects.utilities import (
    open_uniform_knot_vector,
    calc_face_normals,
    plot_mesh_vertices_and_normals,
    plot_child_and_junction_edges,
    plot_parent_and_child_edges,
    plot_parent_and_child_faces,
    angle_between,
    find_visible_vertices,
    plot_filtered_closest_NN,
    plot_smoothed_NN,
    distribute_indices,
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
    plot_mesh_normals,
)
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import pymesh


class Shape:
    def __init__(self, ac_list):

        self.ac_list = ac_list

    def check_inputs(self):

        assert type(self.ac_list) is list, "ac_list must be a list, even if it has just 1 ac."

    def fuse_meshes(self, parent_mesh, child_mesh):
        def find_NNs_between_meshes(parent_mesh, child_mesh):

            p_verts = parent_mesh.vertices
            c_verts = child_mesh.vertices
            tree = scipy.spatial.KDTree(p_verts)
            dd, closest_NN = tree.query(c_verts, k=1)

            threshold_proportion = 0.005
            cutoff = np.round(threshold_proportion * len(c_verts)).astype("int")
            sorted_by_distance = np.argsort(dd)
            c_NNs = sorted_by_distance[:cutoff]
            p_NNs = closest_NN[c_NNs]

            # Greedily group NNs to find the larger group of NNs
            c_pts = c_verts[c_NNs]
            pairwise_dists = scipy.spatial.distance_matrix(c_pts, c_pts)
            pairwise_indices = np.argsort(pairwise_dists, axis=1)
            remaining_indices = c_NNs.tolist()

            # For each item in c_NNs
            # Find the nearest neighbor that has not already been grouped
            # Group it
            # Find NN of each index

            def flatten(groups):
                flattened = []
                for sublist in groups:
                    for i in sublist:
                        flattened.append(i)
                return flattened

            def remaining_indices(c_NNs, groups):
                remaining = set(c_NNs) - set(flatten(groups))
                return list(remaining)

            num_NN = 25
            groups = []
            while remaining_indices(c_NNs, groups) != []:
                NN_queue = remaining_indices(c_NNs, groups)[:1]

                c = 0
                for q in NN_queue:
                    print(c)
                    i = np.argwhere(c_NNs == q)[0][0]

                    # Get the indices of the 5 nearest neighbors
                    NNs = pairwise_indices[i, 1 : 1 + num_NN]
                    NNs = c_NNs[NNs]

                    # Identify which group to assign these neighbors
                    group_num = None
                    for NN in NNs:
                        for num, g in enumerate(groups):
                            if NN in g:
                                group_num = num
                                break
                        if group_num != None:
                            break
                    else:
                        group_num = "NEW GROUP"

                    new_items = NNs.tolist() + [q]  # Neighbors + the starter

                    # Append this to the group
                    if group_num == "NEW GROUP":
                        groups.append(new_items)
                    else:
                        items_to_add = groups[group_num] + new_items
                        items_to_add = list(set(items_to_add))  # Remove redundant
                        groups[group_num] = items_to_add

                    # Add new items to queue
                    new_queue = list(set(new_items) - set(NN_queue))
                    NN_queue.extend(new_queue)
                    c += 1

            # Test pymesh boolean
            c_pym = pymesh.form_mesh(child_mesh.vertices, child_mesh.faces)
            p_pym = pymesh.form_mesh(parent_mesh.vertices, parent_mesh.faces)
            o_pym = pymesh.boolean(c_pym, p_pym, operation="union")

            verts = o_pym.vertices
            faces = o_pym.faces
            face_norms = calc_face_normals(verts, faces)
            vert_norms = trimesh.geometry.mean_vertex_normals(verts.shape[0], faces, face_norms)

            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                face_normals=face_norms,
                vertex_normals=vert_norms,
            )

            v_list = []
            for v in verts:
                if v not in c_pym.vertices and v not in p_pym.vertices:
                    v_list.append(v)
            v_arr = np.zeros((len(v_list), 3))
            for i, v in enumerate(v_list):
                v_arr[i] = v

            # Find the two edges
            # Plot to verify
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=-90, azim=90)

            # joint
            x, y, z = verts.T
            ax.plot(x, y, z, ".", color="green")

            # new
            x, y, z = v_arr.T
            ax.plot(x, y, z, ".", color="red")
            plt.show()

        find_NNs_between_meshes(parent_mesh, child_mesh)
        # # Given two meshes
        # # Find all nearest neighbors that are within some threshold
        # # Create paths on both meshes
        # # align the vertices along these paths
        # # Gaussian smooth points along the paths based on the distance to neighboring points
        # # TODO: Clean up the variables passed into/out of each function
        # # TODO: Change the surface to align to every vertex on the parent_mesh
        # # TODO: Link the surface to the child

        # # Define functions to carry out fusion in steps
        # def find_join_slice_along_child(parent_mesh, child_ac, num_steps_long_axis, num_steps_round_axis):
        #     """Find slice along the child that is just outside the parent_ac."""

        #     # TODO: Do these on a simpler mesh to go faster

        #     # num_steps_long_axis = 10  # Try 10 different slices (increasing distance from parent_mesh)
        #     # num_steps_round_axis = (
        #     #     6  # Test if N points along slice are outside mesh
        #     # )
        #     (us, vs) = child_ac.surface.start()
        #     (ue, ve) = child_ac.surface.end()
        #     u = np.linspace(us, ue, num_steps_round_axis, endpoint=False)
        #     v = np.linspace(vs, ve, num_steps_long_axis)
        #     verts_array = child_ac.surface(u, v)

        #     for slice_num, _ in enumerate(v):
        #         print(slice_num)
        #         points = verts_array[:, slice_num, :]

        #         points_outside_mesh = ~parent_mesh.contains(points)
        #         if np.all(points_outside_mesh):  # All points outside
        #             break

        #         if slice_num == len(v) - 1:
        #             raise NotImplementedError

        #     INCREASE_FACTOR = 0.1  # Move the connection even further up to make it smoother
        #     slice_num += np.round(INCREASE_FACTOR * num_steps_long_axis).astype("int")
        #     slice_dist_approx = np.array(v[slice_num])  # Need to round this up to next highest value in actual v

        #     # Grab the full-size slice
        #     uu = SAMPLING_DENSITY_U
        #     vv = SAMPLING_DENSITY_V
        #     (us, vs) = child_ac.surface.start()
        #     (ue, ve) = child_ac.surface.end()
        #     v = np.linspace(vs, ve, vv)
        #     u = np.linspace(us, ue, uu, endpoint=False)
        #     slice_dist = v[v > slice_dist_approx][0]  # First value of v > slice_dist_approx
        #     full_slice = child_ac.surface(u, slice_dist)
        #     return full_slice, slice_dist, u, slice_dist_approx

        # def project_child_slice_onto_parent_mesh(parent_mesh, child_ac, slice_dist):

        #     # plot_mesh_normals(parent_mesh, visible_vertices, child_ac, slice_dist)

        #     # Expand the child's full_slice and project it onto the surface of the parent_mesh.
        #     TNB_current = np.stack(
        #         [
        #             child_ac.T(slice_dist)[0],
        #             child_ac.N(slice_dist)[0],
        #             child_ac.B(slice_dist)[0],
        #         ],
        #         axis=0,
        #     )

        #     TNB_goal = np.array(
        #         [
        #             [1, 0, 0],
        #             [0, 1, 0],
        #             [0, 0, 1],
        #         ]
        #     )

        #     R = np.linalg.inv(TNB_current) @ TNB_goal

        #     EXPANSION_FACTOR = 1.5
        #     center = child_ac.r(slice_dist)
        #     full_slice_rotated = ((full_slice - center) @ R * EXPANSION_FACTOR) + center
        #     mesh_verts_rotated = (parent_mesh.vertices - center) @ R + center

        #     # Remove x-axis
        #     full_slice_yz = np.squeeze(full_slice_rotated[:, :, 1:])
        #     mesh_verts_yz_all = mesh_verts_rotated[:, 1:]

        #     # Cull vertices on parent mesh that are not visible (i.e. intersect another face)
        #     visible_vertices = find_visible_vertices(
        #         mesh=parent_mesh,
        #         position=child_ac.r(slice_dist),
        #     )
        #     mesh_verts_yz_visible = mesh_verts_yz_all[visible_vertices]
        #     old_indices = np.argwhere(visible_vertices)
        #     new_indices = np.arange(visible_vertices.sum())
        #     idx_dict = {n: o[0] for (o, n) in zip(old_indices, new_indices)}

        #     # Assign all and visible
        #     self.full_slice_yz = full_slice_yz
        #     self.mesh_verts_yz_all = mesh_verts_yz_all
        #     self.mesh_verts_yz_visible = mesh_verts_yz_visible

        #     # Identify the nearest neighbor for each point on the slice
        #     tree = scipy.spatial.KDTree(mesh_verts_yz_visible)
        #     dd, closest_NN = tree.query(full_slice_yz, k=NUM_NN)

        #     closest_NN = np.array([idx_dict[i] for i in closest_NN])  # Renumber to old numbering

        #     # Remove redundant NNs
        #     closest_NN_redundant = closest_NN.copy()
        #     _, idx = np.unique(closest_NN, return_index=True)
        #     closest_NN = closest_NN[np.sort(idx)]

        #     # Add first element to end
        #     closest_NN_wrapped = np.concatenate([closest_NN, [closest_NN[0]]])

        #     # Plot NNs
        #     fig, ax = plt.subplots()

        #     # Plot full_slice
        #     y, z = full_slice_yz.T
        #     ax.plot(y, z, "b*")

        #     # Plot mesh_verts (NNs)
        #     y, z = mesh_verts_yz_all[closest_NN_wrapped].T
        #     ax.plot(y, z, "k.")

        #     # Plot linkages
        #     for i, NN in enumerate(closest_NN_wrapped):

        #         p1 = mesh_verts_yz_all[NN]
        #         idx = closest_NN_redundant.tolist().index(NN)
        #         p2 = full_slice_yz[idx]

        #         x, y = zip(p1, p2)
        #         ax.plot(x, y, "g-")
        #     plt.show()

        #     return closest_NN_wrapped, visible_vertices, closest_NN_redundant

        # def filter_closest_NN(parent_mesh, closest_NN_wrapped):
        #     """Identify abrupt changes in closest_NN and filter them to be smoother."""

        #     all_valid = False
        #     MAX_ANGLE = np.pi * 3 / 4

        #     while all_valid is False:

        #         pts = parent_mesh.vertices[closest_NN_wrapped[:-1]]  # Skip last vertex since it's a duplicate
        #         arr1 = pts
        #         arr2 = np.roll(arr1, shift=1, axis=0)  #  Shift up 1
        #         vec1 = arr2 - arr1
        #         vec2 = np.roll(vec1, shift=1, axis=0)  # Shift up 1

        #         # Find angles
        #         angles = angle_between(vec2, vec1)  # This order gives angle at vertex without shifting
        #         angles = np.concatenate([angles, angles[:1]])
        #         too_large = angles > MAX_ANGLE

        #         # Plot to verify
        #         # plot_filtered_closest_NN(parent_mesh, closest_NN_wrapped, too_large)

        #         # Update by removing NNs with angles that are too large
        #         closest_NN_wrapped = closest_NN_wrapped[~too_large]  # Remove angles that are too large

        #         # Exit while loop
        #         if np.all(~too_large):
        #             all_valid = True

        #     return closest_NN_wrapped

        # def smooth_closest_NN(closest_NN_wrapped, visible_vertices, window_size):

        #     closest_NN = closest_NN_wrapped[:-1]

        #     # Smooth the closest NN by averaging with a sliding window
        #     roll_list = np.arange(window_size) - window_size // 2
        #     points = np.zeros([len(closest_NN), 3, window_size])
        #     for i, roll in enumerate(roll_list):

        #         rolled_indices = np.roll(closest_NN, roll)
        #         points[:, :, i] = parent_mesh.vertices[rolled_indices]

        #     # Find the nearest vertices to these smoothed points
        #     smoothed_points = np.mean(points, axis=2)
        #     tree = scipy.spatial.KDTree(parent_mesh.vertices[visible_vertices])
        #     _, smoothed_NN = tree.query(smoothed_points, k=1)

        #     # Renumber smoothed_NN to old indices
        #     old_indices = np.argwhere(visible_vertices)
        #     new_indices = np.arange(len(parent_mesh.vertices[visible_vertices]))
        #     idx_dict = {n: o[0] for (o, n) in zip(old_indices, new_indices)}
        #     smoothed_NN = np.array([idx_dict[i] for i in smoothed_NN])

        #     # Remove redundant NNs
        #     _, idx = np.unique(smoothed_NN, return_index=True)
        #     smoothed_NN = smoothed_NN[np.sort(idx)]

        #     # Wrap first element
        #     smoothed_NN = np.concatenate([smoothed_NN, [smoothed_NN[0]]])

        #     # Plot to verify
        #     plot_smoothed_NN(parent_mesh, smoothed_NN, closest_NN_wrapped)
        #     return smoothed_NN

        # def find_path_between_projected_vertices(parent_mesh, closest_NN_wrapped):

        #     # Create the graph with edge attributes for length
        #     edges = parent_mesh.edges_unique
        #     length = parent_mesh.edges_unique_length
        #     g = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges, length)])

        #     full_path = []
        #     for i in range(len(closest_NN_wrapped)):
        #         start = closest_NN_wrapped[i - 1]
        #         end = closest_NN_wrapped[i]

        #         # run the shortest path query using length for edge weight
        #         new_path = nx.shortest_path(g, source=start, target=end, weight="length")

        #         # If this path overlaps previous path, remove the previous path
        #         repeated_verts = [p for p in new_path[1:] if p in full_path]
        #         if repeated_verts:

        #             # Find the intermediate vertices (between the repeated) and cut them out

        #             # Find the index of the last repeat in the new path
        #             idx_path_repeat = -1
        #             for r in repeated_verts:
        #                 idx = new_path.index(r)
        #                 if idx > idx_path_repeat:
        #                     idx_path_repeat = idx

        #             # Find the index of the first repeat in the full path
        #             idx_full_repeat = full_path.index(repeated_verts[0])  # Initialize
        #             for r in repeated_verts:
        #                 idx = full_path.index(r)
        #                 if idx < idx_full_repeat:
        #                     idx_full_repeat = idx

        #             # Get the vertices to be appended (skip the repeating ones)
        #             try:
        #                 verts_to_append = new_path[idx_path_repeat + 1 :]
        #             except:
        #                 verts_to_append = []

        #             # Get the vertices for which we need to replace the assigned NN
        #             NNs_to_replace = full_path[idx_full_repeat + 1 :] + new_path[:idx_path_repeat]
        #             NN_replacement = full_path[idx_full_repeat]  # Just assign all

        #             # Replace these NNs for the closest_NN array
        #             # for NN in NNs_to_replace:

        #             #     closest_NN_wrapped[closest_NN_wrapped == NN] = NN_replacement
        #         else:
        #             verts_to_append = new_path[1:]  # Don't duplicate first vert

        #         # Append to the full_path
        #         full_path.extend(verts_to_append)

        #     assert len(np.unique(full_path)) == len(full_path), "Redudant elements in full_path."
        #     full_path.append(full_path[0])  # Add first element to end to close loop

        #     # Plot to see what's going wrong
        #     fig = plt.figure()
        #     ax = plt.axes(projection="3d")
        #     ax.set_xlabel("x")
        #     ax.set_ylabel("y")
        #     ax.set_zlabel("z")
        #     ax.view_init(elev=-90, azim=90)
        #     x, y, z = parent_mesh.vertices[full_path].T
        #     ax.plot(x, y, z, "-", color="green")

        #     x, y, z = parent_mesh.vertices[closest_NN_wrapped].T
        #     ax.plot(x, y, z, ".k")

        #     plt.show()
        #     return np.array(full_path)

        # def distribute_path_to_child_vertices(parent_mesh, child_ac, full_path, u, slice_dist):
        #     """Full path likely has a different number of vertices than the child_ac (which == SAMPLING_DENSITY_U), so we need to distribute them to create pairings."""

        #     parent_points = parent_mesh.vertices[full_path]
        #     child_points = child_ac.surface(u, slice_dist).squeeze()

        #     p_idx, c_idx = distribute_indices(parent_points, child_points).T
        #     p_idx = full_path[p_idx]  # Renumber p_idx
        #     c_idx = u[c_idx]  # Renumber c_idx

        #     return p_idx, c_idx

        # def create_surface_between_child_slice_and_parent_mesh(parent_mesh, child_ac, p_idx, c_idx, slice_dist):

        #     # Identify derivatives at points along parent_mesh. mesh
        #     # TODO: Use something smoother than p_V because this causes ripples in the junction mesh
        #     p_V = parent_mesh.vertices[p_idx]
        #     p_T = parent_mesh.vertices[p_idx] - parent_mesh.vertices[np.roll(p_idx, -1)]

        #     # Replace 0 values of tangent with last nonzero value
        #     for i, row in enumerate(p_T):

        #         if np.all(row == np.array([0.0, 0.0, 0.0])):

        #             p_T[i] = p_T[i - 1]

        #     p_N = parent_mesh.vertex_normals[p_idx]
        #     p_B = np.cross(p_N, p_T)
        #     p_B = p_B / np.linalg.norm(p_B, axis=1, keepdims=True)  # norm

        #     # Identify derivatives at points along child's full slice
        #     c_V = child_ac.surface(c_idx, slice_dist).squeeze()
        #     c_T = child_ac.surface.derivative(c_idx, slice_dist, d=(0, 1)).squeeze()
        #     c_T = c_T / np.linalg.norm(c_T, axis=1, keepdims=True)

        #     # Create B-Spline Surface linking child slice and projection on parent

        #     # Inputs
        #     degree = ORDER - 1

        #     # Basis 1 - cross section
        #     # With >100 controlpoints, the curve essentially passes through the points, so when we go to switch this segment in, if we skip the first and last elements, I think it will work.
        #     num_cp_per_cross_section = c_V.shape[0]
        #     num_knots = num_cp_per_cross_section + ORDER + degree
        #     knot = np.linspace(0, 1, num_knots)
        #     basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)

        #     curve = Curve(basis1, c_V, rational=False)

        #     # Basis 2 - along the major axis of the axial component
        #     num_rows = 4  # End termini + 2 intermediate points to determine slope
        #     knot = open_uniform_knot_vector(num_rows, ORDER)
        #     basis2 = BSplineBasis(order=ORDER, knots=knot, periodic=-1)

        #     # Controlpoints
        #     SCALE_FACTOR = 0.1
        #     cp = np.zeros([c_V.shape[0], 4, 3])
        #     cp[:, 0, :] = c_V
        #     cp[:, 1, :] = c_V - c_T * SCALE_FACTOR
        #     cp[:, 2, :] = p_V - p_B * SCALE_FACTOR
        #     cp[:, 3, :] = p_V
        #     cp = cp.reshape(num_rows * num_cp_per_cross_section, cp.shape[2], order="F")

        #     # Surface
        #     surface = Surface(basis1, basis2, cp, rational=False)
        #     self.surface = surface

        #     return surface, c_V, c_T

        # def stitch_child_and_junction(child_ac, surface, slice_dist, full_slice):

        #     # Sample surface of junction
        #     uu = SAMPLING_DENSITY_U
        #     vv = SAMPLING_DENSITY_V
        #     (us, vs) = surface.start()
        #     (ue, ve) = surface.end()
        #     # u = np.linspace(us, ue, uu, endpoint=False)
        #     u = np.linspace(
        #         ue, us, uu, endpoint=False
        #     )  # Reverse so that normals point outwards - Not 100% sure why this works.
        #     v = np.linspace(vs, ve, vv)
        #     junction_verts_array = surface(u, v)

        #     v_i = np.where(v == slice_dist)[0] - 1  # SHIFT since first row is deleted for endpoint
        #     full_slice_idx = np.arange(uu * v_i, uu * (v_i + 1))

        #     # # # Plot to figure out what's going on
        #     fig = plt.figure()
        #     ax = plt.axes(projection="3d")
        #     ax.set_xlabel("x")
        #     ax.set_ylabel("y")
        #     ax.set_zlabel("z")
        #     ax.view_init(elev=-90, azim=90)
        #     for v in range(vv):
        #         points = junction_verts_array[v, :, :]
        #         x, y, z = points.T
        #         ax.plot(x, y, z, "-", color="green")
        #     plt.show()

        #     assert np.all(np.isclose(np.squeeze(full_slice), child_ac.mesh.vertices[full_slice_idx]))

        #     # Get vertices of child (removing all of those before the full_slice)
        #     endpoint_00 = child_ac.verts.shape[0] - 2  # Index of endpoint at 0.0 position along child
        #     endpoint_10 = child_ac.verts.shape[0] - 1  # Index of endpoint at 1.0 position along child
        #     endpoint_of_subgraph_to_delete = endpoint_00
        #     child_mesh, full_slice_idx_new = remove_subsurface_from_mesh(
        #         child_ac.mesh, full_slice_idx, endpoint_of_subgraph_to_delete
        #     )

        #     # Fuse junction and child
        #     # Since there are the same number of vertices and they already overlap, we just need to renumber the vertices on the child.

        #     # Verts
        #     junction_verts = junction_verts_array.reshape(-1, 3, order="F")

        #     # Faces - CCW Winding (for consistent normals)
        #     # faces = np.zeros((uu * (vv - 2) * 2, 3), dtype="int")
        #     faces = np.zeros((uu * (vv - 1) * 2, 3), dtype="int")
        #     faces_array = np.zeros((uu * 2, vv - 1, 3), dtype="int")
        #     base_column = np.zeros((uu * 2, 3), dtype="int")
        #     base_column[::2, 0] = np.arange(0, uu)
        #     base_column[1::2, 0] = np.arange(0, uu)
        #     base_column[::2, 1] = np.arange(uu, uu * 2)
        #     base_column[1::2, 1] = np.arange(uu + 1, uu * 2 + 1)
        #     base_column[::2, 2] = np.arange(uu + 1, uu * 2 + 1)
        #     base_column[1:-1:2, 2] = np.arange(1, uu)
        #     base_column[-2, 2] = uu  # Fix wrapping
        #     base_column[-1, 1] = uu  # Fix wrapping
        #     base_column[:, 1:] = base_column[:, :-3:-1]  # Reverse for CCW winding

        #     # Grid faces
        #     for i in range(vv - 1):
        #         add_to_column = i * uu
        #         column = base_column + add_to_column
        #         start = uu * i * 2
        #         stop = uu * (i + 1) * 2
        #         faces[start:stop, :] = column
        #         faces_array[:, i, :] = column
        #     junction_faces = faces

        #     junction_face_normals = calc_face_normals(junction_verts, junction_faces)
        #     num_verts = uu * vv
        #     junction_vert_normals = trimesh.geometry.mean_vertex_normals(
        #         num_verts, junction_faces, junction_face_normals
        #     )
        #     junction_mesh = trimesh.Trimesh(
        #         vertices=junction_verts,
        #         faces=junction_faces,
        #         face_normals=junction_face_normals,
        #         vertex_normals=junction_vert_normals,
        #         process=False,
        #     )

        #     # Stitch junction and child meshes
        #     child_edge_idx = full_slice_idx_new
        #     junc_edge_child_idx = np.arange(uu * (0), uu * (1))  # TODO: Adjust for other end
        #     junc_edge_child_idx = junc_edge_child_idx[::-1]  # Flip order to align with child_mesh ordering
        #     junc_edge_child_idx = np.roll(junc_edge_child_idx, 1)  # TODO: Figure out why this shift is necessary
        #     junc_edge_parent_idx = np.arange(uu * (vv - 2), uu * (vv - 1))  # Indices of vertices along parent

        #     # Plot edges
        #     # plot_child_and_junction_edges(child_mesh, child_edge_idx, junction_mesh, junc_edge_child_idx)

        #     # Shift junction edge vertices to align with child edge
        #     junction_mesh.vertices[junc_edge_child_idx] = child_mesh.vertices[child_edge_idx]

        #     # Increase idx of each junction vertex to account for child vertices
        #     num_child_verts = child_mesh.vertices.shape[0]
        #     num_edge_vertices = len(junc_edge_child_idx)
        #     junction_mesh.faces = junction_mesh.faces + num_child_verts - num_edge_vertices
        #     junc_edge_child_idx_shifted = junc_edge_child_idx + num_child_verts - num_edge_vertices
        #     junc_edge_parent_idx_shifted = junc_edge_parent_idx + num_child_verts - num_edge_vertices

        #     # Renumber the junction_edge vertices since they are already in the child vertices
        #     for i, junc_idx in enumerate(junc_edge_child_idx_shifted):

        #         old_idx = junc_idx
        #         new_idx = child_edge_idx[i]

        #         mask = junction_mesh.faces == old_idx
        #         junction_mesh.faces[mask] = new_idx

        #     # Remove the junction_edge vertices
        #     verts_to_keep = [i for i in np.arange(junction_mesh.vertices.shape[0]) if i not in junc_edge_child_idx]
        #     junction_mesh.vertices = junction_mesh.vertices[verts_to_keep]

        #     # Junction faces containing edge vertices should be renumbered to match child edge vertices
        #     combined_verts = np.concatenate([child_mesh.vertices, junction_mesh.vertices], axis=0)
        #     combined_faces = np.concatenate([child_mesh.faces, junction_mesh.faces])
        #     combined_face_norms = calc_face_normals(combined_verts, combined_faces)
        #     combined_vert_norms = trimesh.geometry.mean_vertex_normals(
        #         len(combined_verts), combined_faces, combined_face_norms
        #     )

        #     combined_mesh = trimesh.Trimesh(
        #         vertices=combined_verts,
        #         faces=combined_faces,
        #         face_normals=combined_face_norms,
        #         vertex_normals=combined_vert_norms,
        #         process=False,
        #     )

        #     # combined_mesh.show()

        #     # Plot junction and child meshes
        #     # trimesh.Scene([child_mesh, junction_mesh]).show()

        #     return combined_mesh, junc_edge_parent_idx_shifted

        # def remove_interior_vertices_from_parent(parent_mesh, full_path):

        #     # Strategy 1 - networkx
        #     parent_verts = parent_mesh.vertices

        #     # Find middle point (assume this is within region we want to cut out)
        #     centerpoint = parent_verts[full_path].mean(axis=0)

        #     # Find nearest neighbor of actual vertex to this centerpoint
        #     tree = scipy.spatial.KDTree(parent_verts)
        #     _, center_vert_idx = tree.query(centerpoint, k=1)

        #     mesh, full_path_new = remove_subsurface_from_mesh(parent_mesh, full_path, center_vert_idx)

        #     return mesh, full_path_new

        # def stitch_parent_and_child(parent_mesh, parent_edge_vertices, combined_mesh, combined_edge_vertices):

        #     # Plot the two edges to make sure things look okay
        #     plot_child_and_junction_edges(
        #         parent_mesh, parent_edge_vertices, combined_mesh, combined_edge_vertices, plot_linkages=False
        #     )

        #     # Label the two meshes as short (s) or long (l)
        #     len_parent = len(parent_edge_vertices)
        #     len_combined = len(combined_edge_vertices)
        #     if len_parent < len_combined:
        #         s_mesh = parent_mesh
        #         s_edge = parent_edge_vertices
        #         l_mesh = combined_mesh
        #         l_edge = combined_edge_vertices
        #     elif len_combined < len_parent:
        #         s_mesh = combined_mesh
        #         s_edge = combined_edge_vertices
        #         l_mesh = parent_mesh
        #         l_edge = parent_edge_vertices
        #     else:  # Two lengths are equal
        #         s_mesh = parent_mesh
        #         s_edge = parent_edge_vertices
        #         l_mesh = combined_mesh
        #         l_edge = combined_edge_vertices

        #     ### Evenly distribute short_edge to long_edge

        #     # Find the nearest neighbor of the 0th element on the short list
        #     l_edge = np.flip(l_edge)  # Align order
        #     pairings = np.zeros([l_edge.shape[0], 2], dtype="int")
        #     tree = scipy.spatial.KDTree(l_mesh.vertices[l_edge])
        #     dd, ii = tree.query(s_mesh.vertices[s_edge[0]], k=1)
        #     l_edge_rolled = np.roll(l_edge, -ii)  # Roll to align l_edge and s_edge

        #     s_l_ratio = len(s_edge) / len(l_edge)

        #     for i, l in enumerate(l_edge_rolled):

        #         s_i = np.round(i * s_l_ratio).astype("int")
        #         s = s_edge[s_i]

        #         pairings[i] = [s, l]

        #     assert np.all(pairings[:, 1] == l_edge_rolled), "Missing l_edge vertices."
        #     assert set(s_edge) == set(pairings[:, 0]), "Missing s_edge vertices."

        #     # Construct faces
        #     faces = []
        #     pairings_wrapped = np.zeros([pairings.shape[0] + 1, pairings.shape[1]], dtype="int")
        #     pairings_wrapped[:-1] = pairings
        #     pairings_wrapped[-1] = pairings[0]

        #     for idx, l_i in enumerate(pairings_wrapped[:-1, 1]):

        #         v1 = pairings_wrapped[idx, 0]
        #         v2 = pairings_wrapped[idx, 1]
        #         v3 = pairings_wrapped[idx + 1, 0]
        #         v4 = pairings_wrapped[idx + 1, 1]

        #         if v1 != v3:  # If short edge vertices are different
        #             faces.append([v1, v3, v2])
        #             faces.append([v3, v4, v2])
        #         else:
        #             faces.append([v1, v4, v2])
        #     # Plot edges
        #     # TODO: FIgure out why edge of parent and child are overlapping - should not be the case
        #     plot_parent_and_child_edges(s_mesh, l_mesh, pairings, plot_linkages=True)
        #     # plot_parent_and_child_faces(s_mesh, l_mesh, faces, s_edge, l_edge)
        #     pass
        #     # # Find the nearest neighbor of the 0th element on the short list
        #     # pairings = {}
        #     # tree = scipy.spatial.KDTree(l_mesh.vertices[l_edge])
        #     # dd, ii = tree.query(s_mesh.vertices[s_edge[0]], k=1)
        #     # pairings[s_edge[0]] = l_edge[ii]

        #     # # Loop through remaining points on short list to identify nearest neighbors
        #     # l_edge_rolled = np.roll(l_edge, -ii)
        #     # l_start_idx = 0
        #     # roll_steps = 0

        #     # for s_i in s_edge[1:]:

        #     #     l_edge_rolled = np.roll(l_edge_rolled, -roll_steps)
        #     #     prev_dist = np.inf
        #     #     prev_idx = np.inf

        #     #     s_p = s_mesh.vertices[s_i]

        #     #     for l_i in l_edge_rolled:

        #     #         l_p = l_mesh.vertices[l_i]
        #     #         curr_dist = np.linalg.norm(s_p - l_p)

        #     #         if curr_dist > prev_dist:
        #     #             break

        #     #         prev_dist = curr_dist  # Update previous distance for comparison in next loop
        #     #         prev_idx = l_i
        #     #         roll_steps += 1

        #     #     pairings[s_i] = prev_idx  # Add PREVIOUS point to pairings

        #     # Link the intermediate points on the other list to that NND
        #     return None

        # # Call the above functions to fuse the child and parent
        # full_slice, slice_dist, u, slice_dist_approx = find_join_slice_along_child(
        #     parent_mesh,
        #     child_ac,
        #     num_steps_long_axis=10,
        #     num_steps_round_axis=6,
        # )
        # closest_NN_wrapped, visible_vertices, closest_NN_redundant = project_child_slice_onto_parent_mesh(
        #     parent_mesh, child_ac, slice_dist
        # )
        # closest_NN_wrapped = filter_closest_NN(parent_mesh, closest_NN_wrapped)
        # closest_NN_wrapped = smooth_closest_NN(closest_NN_wrapped, visible_vertices, window_size=WINDOW_SIZE)
        # full_path = find_path_between_projected_vertices(parent_mesh, closest_NN_wrapped)
        # plot_projected_vertices_and_NNs_3D(full_slice, closest_NN_redundant, parent_mesh.vertices, full_path)
        # plot_projected_vertices_and_NNs(
        #     self.full_slice_yz,
        #     closest_NN_redundant,
        #     self.mesh_verts_yz_all,
        #     full_path,
        # )
        # p_idx, c_idx = distribute_path_to_child_vertices(parent_mesh, child_ac, full_path, u, slice_dist)
        # surface, c_V, c_T = create_surface_between_child_slice_and_parent_mesh(
        #     parent_mesh, child_ac, p_idx, c_idx, slice_dist
        # )
        # plot_surface_linking_axial_components(parent_mesh, child_ac, surface)
        # # TODO: Fix this so that the meshes get joined appropriately :)
        # combined_mesh, combined_edge_vertices = stitch_child_and_junction(child_ac, surface, slice_dist, full_slice)

        # # Delete the hole in the parent mesh
        # parent_mesh_new, full_path_new = remove_interior_vertices_from_parent(parent_mesh, full_path)

        # new_mesh = stitch_parent_and_child(parent_mesh_new, full_path_new, combined_mesh, combined_edge_vertices)

        # # Stitch together the two
        # # For the smaller sequence, find the nearest neighbor of each vertex to points in the other sequence
        # # Between these pairings, link points in the larger sequence to the first elmeent of each gap in the shorter sequence
        # # Result is a list of edges which we will convert to faces

    def plot_meshes(self):

        trimesh.Scene([ac.mesh for ac in self.ac_list]).show()

    def merge_meshes(self):

        merged_meshes = trimesh.boolean.union([ac.mesh for ac in self.ac_list], engine="scad")
        bf = trimesh.repair.broken_faces(merged_meshes)
        self.merged_meshes = merged_meshes
