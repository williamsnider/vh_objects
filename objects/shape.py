from copy import Error
from networkx.algorithms.core import k_core
from networkx.algorithms.operators.binary import union
from networkx.classes.function import neighbors
from numpy.core.fromnumeric import _searchsorted_dispatcher
import trimesh
import numpy as np
from trimesh import parent
from objects.parameters import SAMPLING_DENSITY_V, SAMPLING_DENSITY_U, ORDER, WINDOW_SIZE
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
    flatten,
)
import scipy
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import networkx as nx
from splipy import BSplineBasis, Curve, Surface
from objects.utilities import (
    remove_subsurface_from_mesh,
    plot_projected_vertices_and_NNs,
    plot_projected_vertices_and_NNs_3D,
    plot_mesh_derivatives,
    plot_surface_linking_axial_components,
    plot_mesh_normals,
    fix_mesh,
    sliding_window_mean,
)
import matplotlib.pyplot as plt
import pymesh
from sklearn.cluster import KMeans
from copy import deepcopy
import time


class Shape:
    def __init__(self, ac_list):

        self.ac_list = ac_list

    def check_inputs(self):

        assert type(self.ac_list) is list, "ac_list must be a list, even if it has just 1 ac."

    def fuse_meshes(self, parent_mesh, child_mesh):
        def calc_mesh_boolean_and_edges(mesh1, mesh2):

            # Use PyMesh to get boolean
            pym_1 = pymesh.form_mesh(mesh1.vertices, mesh1.faces)
            pym_2 = pymesh.form_mesh(mesh2.vertices, mesh2.faces)
            pym_o = pymesh.boolean(pym_1, pym_2, operation="union")  # Output

            # Get edges - vertices that were in neither initial mesh
            set_1 = set([tuple(l) for l in pym_1.vertices.tolist()])
            set_2 = set([tuple(l) for l in pym_2.vertices.tolist()])
            set_o = set([tuple(l) for l in pym_o.vertices.tolist()])
            new_verts = (set_o - set_1) - set_2
            edge_verts_pts = np.zeros((len(new_verts), 3))
            for i, v in enumerate(new_verts):
                edge_verts_pts[i] = list(v)

            # Return as trimesh - easier to work with
            verts = pym_o.vertices
            faces = pym_o.faces
            face_norms = calc_face_normals(verts, faces)
            vert_norms = trimesh.geometry.mean_vertex_normals(verts.shape[0], faces, face_norms)

            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                face_normals=face_norms,
                vertex_normals=vert_norms,
            )

            # Get indices of edge_verts
            tree = scipy.spatial.KDTree(mesh.vertices)
            _, edge_verts_indices = tree.query(edge_verts_pts, k=1)

            return mesh, edge_verts_indices

        def segment_edge_verts(union_mesh, edge_verts_indices):

            # Group by nearest neighbor distance
            NUM_NN = np.round(np.sqrt(SAMPLING_DENSITY_U * SAMPLING_DENSITY_V) * 0.05).astype("int")
            edge_verts_pts = union_mesh.vertices[edge_verts_indices]
            tree = scipy.spatial.KDTree(edge_verts_pts)
            dd, closest_NN = tree.query(edge_verts_pts, k=NUM_NN + 1)

            groups = []
            vertices_remaining = [i for i, v in enumerate(edge_verts_pts)]
            while vertices_remaining != []:

                curr_group = [vertices_remaining[0]]  # Start with some edge_vert's index
                prev_group = []

                for idx in curr_group:

                    # Remove this idx from the vertices remaining
                    vertices_remaining.remove(idx)

                    # Gather the nearest neighbors
                    new_verts = closest_NN[idx, 1:]

                    # Include only those that are still in the remaining list
                    new_verts = list(set.intersection(set(new_verts), set(vertices_remaining)))

                    # Exclude those that are already in the current group
                    new_verts = list(set(new_verts) - set(curr_group))

                    # Append to curr_group
                    curr_group.extend(new_verts)

                groups.append(curr_group)

            # Renumber indices to match the union_mesh
            groups_renumbered = []
            for group in groups:

                group_renumbered = []
                for vert in group:
                    group_renumbered.append(edge_verts_indices[vert])

                groups_renumbered.append(np.array(group_renumbered))
            return np.array(groups_renumbered)

        def order_group(union_mesh, unordered_group):

            # Get nearest neighbor table
            tree = scipy.spatial.KDTree(union_mesh.vertices[unordered_group])
            dd, closest_NN = tree.query(union_mesh.vertices[unordered_group], k=len(unordered_group))

            remaining_vertices = [i for i in unordered_group]
            ordered_group = [unordered_group[0]]  # Start with first edge vertex. Note indexing is

            for vert in ordered_group:

                # Exit for loop once all group members have been sorted
                if len(ordered_group) == len(unordered_group):
                    break

                remaining_vertices.remove(vert)
                vert_idx = np.argwhere(unordered_group == vert)[0, 0]
                nearest_neighbors = [unordered_group[i] for i in closest_NN[vert_idx][1:]]  #!!! Skip 0th index (itself)

                if len(ordered_group) == 1:  # Just assign first value so that we can start testing vector directions
                    NN = nearest_neighbors[0]
                else:

                    # Form vector between previous two ordered edge vertices
                    prev_vec = union_mesh.vertices[ordered_group[-2]] - union_mesh.vertices[ordered_group[-1]]

                    # Form vector between current edge vertices and all possible nearest neighbors
                    curr_vecs = union_mesh.vertices[nearest_neighbors] - union_mesh.vertices[ordered_group[-1]]

                    # Calculate angles - two vectors (as determinede above) should have opposite signs
                    angles = angle_between(prev_vec, curr_vecs)
                    CUTOFF = np.pi / 4
                    valid_angles = angles > CUTOFF

                    # Iterate through neighbors until we find the first valid one
                    # TODO: Why is this not breaking out of the loop?
                    for i, NN in enumerate(nearest_neighbors):
                        if (valid_angles[i] == True) and (NN in remaining_vertices):
                            break
                    else:
                        raise ValueError

                ordered_group.append(NN)
            return np.array(ordered_group)

        def fit_spline(union_mesh, group):

            points = union_mesh.vertices[group]
            points = np.concatenate([points, points[0].reshape(1, -1)], axis=0)  # Duplicate first point since periodic

            # Find distances between each successive point
            points_roll = np.roll(points, 1, axis=0)
            distances = np.linalg.norm(points_roll - points, axis=1)
            distances_cum = np.cumsum(distances)
            t = distances_cum / distances.sum()  # Scale 0 to 1

            # Get spline
            spline = CubicSpline(t, points, bc_type="periodic")
            return spline

        def plot_splines(mesh, groups, splines, spacing=1):

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=-90, azim=90)

            # Sample from entire mesh
            x, y, z = mesh.vertices[::spacing].T
            ax.plot(x, y, z, ".", color="green")

            # Plot groups
            for i, group in enumerate(groups):
                c = np.array([0.25, i, 0.75])
                x, y, z = mesh.vertices[group].T
                ax.plot(x, y, z, ".", color=c)

            # Plot group vertex normals
            for i, group in enumerate(groups):
                c = np.array([0, 0, 0])
                for g in group:

                    p1 = mesh.vertices[g]
                    p2 = p1 + mesh.vertex_normals[g] * 0.1
                    x, y, z = zip(p1, p2)
                    ax.plot(x, y, z, "-", color=c)

            # Plot splines
            t = np.linspace(0, 1, 100)
            for i, spline in enumerate(splines):
                c = np.array([i, 0.25, 0.75])
                x, y, z = spline(t).T
                ax.plot(x, y, z, "-", color=c)

            plt.show()
            pass

        def find_valid_neighbors_by_distance(union_mesh, group, distance, distance_cutoff):

            mesh_pts = union_mesh.vertices
            tree = scipy.spatial.KDTree(mesh_pts)

            # Neighbors must within distance range (distance*distance_cutoff, distance)
            group_pts = union_mesh.vertices[group]
            neighbors_within_distance = tree.query_ball_point(group_pts, distance)
            neighbors_too_close = tree.query_ball_point(group_pts, distance * distance_cutoff)
            neighbors_within_cutoff = []  # Neighbors within the correct distance range (not too far, not too close)
            nearby_points = set()  # Gather list of all points within DISTANCE of any edge_vertex
            for i in range(len(group)):
                within_distance = set(neighbors_within_distance[i])
                too_close = set(neighbors_too_close[i])
                within_cutoff = list(within_distance - too_close)
                neighbors_within_cutoff.append(within_cutoff)
                nearby_points.update(within_distance)

            # # XXX: Test whether the distances exceed DISTANCE
            # nearby_pts = union_mesh.vertices[list(nearby_points)]
            # edge_verts_pts = union_mesh.vertices[group]
            # nearby_distances = cdist(edge_verts_pts, nearby_pts)

            # Neighbors must be within angle range relative to tangent at edge_vert
            ROLL_AMOUNT = 1  # shift to next vert to find the tangent vector
            WINDOW_SIZE = 9  # Average the tangent vectors using a sliding window of size 5 (to smooth)
            ANGLE_RANGE = np.array([-1, 1]) * np.pi / 6 + np.pi / 2
            group_tan_vec = group_pts - np.roll(group_pts, ROLL_AMOUNT, axis=0)
            group_tan_vec_smooth = sliding_window_mean(group_tan_vec, window_size=WINDOW_SIZE, axis=0)
            neighbors_within_angle = []
            for i in range(len(group)):
                neighbors = np.array(list(neighbors_within_cutoff[i]))  # Only get neighbors in correct distance range
                neighbors_vecs = mesh_pts[neighbors] - group_pts[i]
                tangent_vector = group_tan_vec_smooth[i]
                angles = angle_between(tangent_vector, neighbors_vecs)
                valid_indices = np.argwhere((ANGLE_RANGE[0] <= angles) & (angles <= ANGLE_RANGE[1]))
                within_angle = neighbors[np.squeeze(valid_indices)]
                neighbors_within_angle.append(within_angle)  # This also includes neighbors at the correct distance

            # Segment neighbors into two divisions on either side of the edge_vertex
            neighbors_by_division = []
            for i, vert in enumerate(group):

                T = group_tan_vec_smooth[i]
                N = union_mesh.vertex_normals[vert]
                B = np.cross(T, N)  # Segment neighbors based on whether they point in same direction as B

                neighbors = np.array(neighbors_within_angle[i])
                try:
                    neighbors_pts = mesh_pts[neighbors]
                except:
                    pass
                neighbors_vecs = neighbors_pts - mesh_pts[vert]

                # THRESHOLD = np.pi / 4
                # angles = angle_between(neighbors_vecs, B)
                # division_A = neighbors[angles <= THRESHOLD].tolist()
                # division_B = neighbors[angles > THRESHOLD].tolist()
                # neighbors_by_division.append([division_A, division_B])

                THRESHOLD = 0
                dot = neighbors_vecs @ B
                division_A = neighbors[dot <= THRESHOLD].tolist()
                division_B = neighbors[dot > THRESHOLD].tolist()
                neighbors_by_division.append([division_A, division_B])
            valid_neighbors = neighbors_by_division

            # # # Plot to verify
            # idx = 38
            # fig = plt.figure()
            # ax = plt.axes(projection="3d")
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            # ax.view_init(elev=-90, azim=90)
            # # Plot all verts
            # c = "black"
            # x, y, z = mesh_pts[::10].T
            # ax.plot(x, y, z, ".", color=c)

            # # Plot vert
            # c = "cyan"
            # x, y, z = mesh_pts[group[idx]].T
            # ax.plot(x, y, z, ".", color=c)

            # # Plot neighbors
            # for j, division in enumerate(valid_neighbors[idx]):
            #     if j == 0:
            #         c = "purple"
            #     else:
            #         c = "yellow"
            #     x, y, z = mesh_pts[division].T
            #     ax.plot(x, y, z, ".", color=c)

            # plt.show()
            # 1D: verts within that group
            # 2D: which division
            # 3D: neighbors

            return valid_neighbors, list(nearby_points)

        def find_valid_neighbors(union_mesh, groups):
            def find_neighbors_by_degree(mesh, vertex_idx, degree=1):

                for d in range(degree + 1):  # We want to include the highest degree

                    if d == 0:
                        neighbors_by_degree = [[vertex_idx]]
                        continue

                    all_neighbors = flatten(neighbors_by_degree)
                    degree_neighbors = []
                    for n in neighbors_by_degree[d - 1]:
                        adjacent = mesh.vertex_neighbors[n]
                        new_neighbors = list(set(adjacent) - set(all_neighbors))
                        degree_neighbors.append(new_neighbors)

                    neighbors_by_degree.append(flatten(degree_neighbors))

                return neighbors_by_degree

            expanded_groups = None

            # Find the vertices on the union_mesh that are not near other edge_vertices
            DEGREE = 7
            degree_shift = 1  # e.g. if we find neighbors 5 degrees away, exclude all that are 3 or closer degrees

            def find_nested_neighbors(union_mesh, groups, degree=DEGREE):
                """
                0th dim = which group
                1st dim = which vertex in that group
                2nd dim = which degree of separation
                3rd dim = which vertex among those vertices at that degree of separation
                """
                nested_neighbors = []
                for group in groups:
                    neighbors = []
                    for vert in group:
                        neighbors.append(find_neighbors_by_degree(union_mesh, vert, degree=degree))
                    nested_neighbors.append(neighbors)
                return nested_neighbors

            # Get list of neighbors that are too close and need to be excluded
            nested_neighbors = find_nested_neighbors(union_mesh, groups, degree=DEGREE)
            neighbors_too_close = []
            for i, group in enumerate(groups):
                group_too_close = []
                for j, vert in enumerate(group):
                    vert_too_close = set()
                    for k in range(DEGREE - degree_shift):
                        vert_too_close.update(nested_neighbors[i][j][k])
                    group_too_close.append(vert_too_close)
                neighbors_too_close.append(group_too_close)

            # Exclude these neighbors based on distance (in degrees of separation)
            groups_neighbors = []  # d0=group; d1=vertex; d2=valid neighbors
            for i, group in enumerate(groups):
                vert_neighbors = []
                for j, vert in enumerate(group):
                    neighbors = list(set(nested_neighbors[i][j][DEGREE]) - neighbors_too_close[i][j])
                    vert_neighbors.append(neighbors)
                groups_neighbors.append(vert_neighbors)

            # Exclude neighbors based on the angle formed between them and the tangent at the edge vertex
            valid_neighbors = deepcopy(groups_neighbors)
            for i, group in enumerate(groups_neighbors):
                for j, neighbors in enumerate(group):

                    # Gather values for the edge vertex, the preceding edge vertex, and the putative neighbors
                    vert_idx = groups[i][j]
                    vert_pts = union_mesh.vertices[vert_idx]
                    next_vert = j - 1
                    next_vert_idx = groups[i][next_vert]
                    next_vert_pts = union_mesh.vertices[next_vert_idx]
                    neighbors_pts = union_mesh.vertices[neighbors]

                    # Form vectors
                    vec1 = next_vert_pts - vert_pts
                    vec_arr = neighbors_pts - vert_pts

                    # Calc angles
                    ANGLE_CUTOFF = [np.pi / 4, 3 * np.pi / 4]
                    angles = angle_between(vec1, vec_arr)
                    valid_angles = (ANGLE_CUTOFF[0] <= angles) & (angles <= ANGLE_CUTOFF[1])

                    for k, valid in enumerate(valid_angles):
                        if valid == False:
                            try:
                                valid_neighbors[i][j].remove(neighbors[k])
                            except:
                                pass

            return valid_neighbors

        def plot_neighbors(mesh, groups, groups_neighbors, spacing=10):

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=-90, azim=90)

            # Sample from entire mesh
            x, y, z = mesh.vertices[::spacing].T
            ax.plot(x, y, z, ".", color="green")

            # Plot groups
            for i, group in enumerate(groups):
                c = np.array([0.25, i, 0.75])
                x, y, z = mesh.vertices[group].T
                ax.plot(x, y, z, ".", color=c)

            # Plot neighbors
            for a, group in enumerate(groups_neighbors[0]):
                for j, neighbors in enumerate(group):
                    c = np.array([0.55, i, 0.75])
                    x, y, z = mesh.vertices[neighbors].T
                    ax.plot(x, y, z, ".", color=c)

                if a == 0:
                    break

            # Plot vertex under question
            x, y, z = mesh.vertices[groups[0][0]].T
            ax.plot(x, y, z, ".", color="red")

            plt.show()
            pass

        def calc_average_of_neighbors(union_mesh, valid_neighbors):

            neighbors_average = np.zeros((len(valid_neighbors), 3))
            for j, vert in enumerate(valid_neighbors):

                division_avg = np.zeros((2, 3))
                for k, division in enumerate(vert):

                    # # Arithmetic mean of points
                    # division_pts = union_mesh.vertices[division]
                    # division_avg[k] = division_pts.mean(axis=0)

                    # # Arithmetic mean of convex hull vertices
                    # division_pts = union_mesh.vertices[division]
                    # hull = ConvexHull(division_pts)
                    # division_avg[k] = division_pts[hull.vertices].mean(axis=0)

                    # Weight the points' contribution based on the size of the faces they touch
                    NUM_FACES = 6  # Most vertices have only this number of neighboring faces
                    d_pts = union_mesh.vertices[division]
                    d_faces = union_mesh.vertex_faces[division][:, :NUM_FACES]

                    # Some vertices may not border NUM_FACES faces, so pad these by duplicating faces that they do border
                    (r, c) = np.where(d_faces == -1)
                    for row, col in zip(r, c):
                        value_to_pad_with = d_faces[row, 0]  # Pad with 0th face in each row
                        d_faces[row, col] = value_to_pad_with
                    assert np.any(d_faces == -1) == False, "Padding this vertex with faces failed."

                    # Weight by area of faces each vertex touches
                    d_face_areas = union_mesh.area_faces[d_faces].sum(axis=1).reshape(-1, 1)
                    d_total_area = d_face_areas.sum()
                    d_weighted_pts = d_pts * d_face_areas / d_total_area
                    division_avg[k] = d_weighted_pts.sum(axis=0)
                neighbors_average[j] = division_avg.mean(axis=0)

            # # Smooth this average
            WINDOW_SIZE = 9
            neighbors_average = sliding_window_mean(neighbors_average, WINDOW_SIZE, axis=0)
            return neighbors_average

        def plot_group_averages(mesh, groups, groups_average, spacing=10):

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=-90, azim=90)

            # Sample from entire mesh
            x, y, z = mesh.vertices[::spacing].T
            ax.plot(x, y, z, ".", color="green")

            # Plot groups
            for i, group in enumerate(groups):
                c = np.array([0.25, i, 0.75])
                x, y, z = mesh.vertices[group].T
                ax.plot(x, y, z, "-", color=c)

            # Plot averages under question
            for j, group_avg in enumerate(groups_average):

                c = "red"
                x, y, z = group_avg.T
                ax.plot(x, y, z, "-", color=c)

            plt.show()
            # TODO: Figure how why average of smaller group is so jagged - need to smooth it out
            pass

        def blend_nearby_points(union_mesh, group, neighbor_averages, nearby_points, distance):

            # Find distance of all mesh points to edge_verts
            edge_verts_pts = union_mesh.vertices[group]
            nearby_pts = union_mesh.vertices[nearby_points]
            nearby_distances = cdist(edge_verts_pts, nearby_pts)
            nearby_ratio = (distance - nearby_distances) / distance  # Convert these distances to ratios
            nearby_ratio[nearby_ratio <= 0] = 0  # Ratio below 0  means neighbor is out of range

            # Find vector of all mesh points to the neighbors_average
            neighbor_averages_3D = np.expand_dims(neighbor_averages, axis=1)  # Add 3D to allow for repetition
            nearby_vec = neighbor_averages_3D - nearby_pts

            # Multiply vector by ratio of distance from point to edge_vert
            nearby_ratio_3D = np.expand_dims(nearby_ratio, axis=2)
            nearby_ratio_3D = np.repeat(nearby_ratio_3D, 3, axis=2)
            nearby_shift = nearby_vec * nearby_ratio_3D

            # # Apply largest shift to each point
            # largest_idx = np.argmax(nearby_ratio, axis=0)
            # num_nearby = nearby_pts.shape[0]
            # largest_shift = nearby_shift[largest_idx, np.arange(num_nearby), :]
            # nearby_pts_shifted = nearby_pts + largest_shift

            # TODO: add in a smoother fall off function
            def falloff(ratio):

                return np.sin(ratio * np.pi / 4)  # Max when ratio = 1, min when ratio = 0.

            # TODO: Weight shifts by their ratios
            # TODO: Figure out why this overloads
            weight_falloff = falloff(nearby_ratio)
            weight_arr = weight_falloff / weight_falloff.sum(axis=0)
            weight_arr = falloff(nearby_ratio) / falloff(nearby_ratio_3D).sum(axis=0)
            shifts = (nearby_shift * weight_arr).sum(axis=0)
            nearby_pts_shifted = nearby_pts + shifts

            # # Plot to verify
            # spacing = 1
            # fig = plt.figure()
            # ax = plt.axes(projection="3d")
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            # ax.view_init(elev=-90, azim=90)

            # # joint
            # x, y, z = union_mesh.vertices[nearby_points].T
            # ax.plot(x, y, z, ".", color="green")

            # # new
            # for i, p1 in enumerate(nearby_pts):
            #     p2 = nearby_pts_shifted[i]
            #     x, y, z = zip(p1, p2)
            #     ax.plot(x, y, z, "-", color="red")
            # plt.show()

            # Update mesh
            union_mesh.vertices[nearby_points] = nearby_pts_shifted
            union_mesh.face_normals = calc_face_normals(union_mesh.vertices, union_mesh.faces)
            union_mesh.vertex_normals = trimesh.geometry.mean_vertex_normals(
                union_mesh.vertices.shape[0],
                union_mesh.faces,
                union_mesh.face_normals,
            )
            union_mesh.show()
            return union_mesh

            # return None

        def calc_tangent_vector(mesh, group, smoothing_window=9):

            group_pts = mesh.vertices[group]
            group_pts_roll = np.roll(group_pts, 1, axis=0)
            group_tangent_vector = group_pts_roll - group_pts
            group_tangent_vector_smooth = sliding_window_mean(group_tangent_vector, smoothing_window, axis=0)
            return group_tangent_vector_smooth

        def plot_mesh_and_group(
            mesh, group, group_tangent_vector=None, spline=None, spacing=1, average_of_neighbors=None
        ):

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=-90, azim=90)

            # Entire mesh
            x, y, z = mesh.vertices[::spacing].T
            ax.plot(x, y, z, ".", color="green")

            # Vertices in the group
            x, y, z = mesh.vertices[group].T
            ax.plot(x, y, z, ".", color="blue")

            # Tangent vector
            if type(group_tangent_vector) != type(None):
                for i, p1 in enumerate(mesh.vertices[group]):
                    p2 = p1 + group_tangent_vector[i]
                    x, y, z = zip(p1, p2)
                    ax.plot(x, y, z, "-", color="red")

            # Spline
            if type(spline) != type(None):
                t = np.linspace(0, 1, 100)
                x, y, z = spline(t).T
                ax.plot(x, y, z, ".", color="black")

            # Average of neighbors
            if type(average_of_neighbors) != type(None):
                x, y, z = average_of_neighbors.T
                ax.plot(x, y, z, "-", color="purple")

            plt.show()

        def calc_average_of_neighbors(mesh, spline, spline_sampling, distance):

            t = np.linspace(0, 1, spline_sampling)
            edge_pts = spline(t)
            mesh_pts = mesh.vertices.__array__()

            ### Create weights for distance, angle from tangent_vector, and the size of the faces that each vertex borders ###

            #########
            # Distance weights
            # Further mesh_pts (that are within distance) should be weighted highly.
            distances = cdist(edge_pts, mesh_pts)
            # w_d = distances / distance
            w_d = distances ** 2 / distance  # Square to give higher weight to further distances
            w_d[w_d > 1] = 0

            #########
            # Angle weights
            # The angle is formed by the tangent vector along the spline and the vector between the spline and the mesh point. Angles near a right angle should point to mesh points that are
            mesh_vec = np.expand_dims(mesh_pts, axis=0) - np.expand_dims(edge_pts, axis=1)
            tan_vec = spline.derivative(1)(t)
            tan_vec = np.expand_dims(tan_vec, axis=1)
            angles = angle_between(mesh_vec, tan_vec)
            # Give weight 0 to nan, which happens when mesh_vec is comparing two identical points (vector with no direction )
            ANGLE_CUTOFF = [np.pi / 3, 2 * np.pi / 3]
            angles[np.isnan(angles)] = 0
            angles[angles < ANGLE_CUTOFF[0]] = 0
            angles[angles > ANGLE_CUTOFF[1]] = 0
            w_a = angles / angles.max()

            ###########
            # Face area weights
            NUM_FACES = 6  # Most vertices have this number of neighboring faces
            mesh_faces = mesh.vertex_faces[:, :NUM_FACES].copy()

            # Some vertices may not border NUM_FACES faces, so pad these by duplicating faces that they do border
            (r, c) = np.where(mesh_faces == -1)
            for row, col in zip(r, c):
                value_to_pad_with = mesh_faces[row, 0]  # Pad with 0th face in each row
                mesh_faces[row, col] = value_to_pad_with
            assert np.any(mesh_faces == -1) == False, "Padding this vertex with faces failed."

            # Weight by area of faces each vertex touches
            mesh_faces_area = mesh.area_faces[mesh_faces].sum(axis=1)
            # w_f = mesh_faces_area / mesh_faces_area.max()
            w_f = np.clip(mesh_faces_area / mesh_faces_area.mean(), 0.0, 1.0)

            # Combine all weights, normalize so that they sum to 1 for each edge_pt
            print("Testing without faces")
            w = w_d * w_a  # * w_f
            w = w / w.sum(axis=1, keepdims=True)
            w = np.expand_dims(w, axis=2)

            # Multiply mesh points by weights, then sum, to find average position
            average_of_neighbors = (mesh_pts * w).sum(axis=1)

            # Plot highly contributing points
            idx = 25
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=-90, azim=90)

            # Entire mesh
            x, y, z = mesh.vertices.T
            ax.plot(x, y, z, ".", color="green")

            # Vertices contributing
            contributors = w[idx] > w[idx].mean() * 100
            x, y, z = mesh.vertices[np.squeeze(contributors)].T
            ax.plot(x, y, z, ".", color="blue")

            # Vertex
            x, y, z = edge_pts[idx]
            ax.plot(x, y, z, ".", color="red")

            # Average
            x, y, z = average_of_neighbors[idx]
            ax.plot(x, y, z, ".", color="purple")
            return average_of_neighbors

        union_mesh, edge_verts_indices = calc_mesh_boolean_and_edges(parent_mesh, child_mesh)
        groups = segment_edge_verts(union_mesh, edge_verts_indices)

        # Do fusion for each of the groups
        for unordered_group in groups:

            print("Testing the largest group only... delete this!")
            if len(unordered_group) != max([len(g) for g in groups]):
                continue

            # Ensure the vertices in each group are in order
            group = order_group(union_mesh, unordered_group)

            # Fit spline to allow for even sampling
            spline = fit_spline(union_mesh, group)

            # Calculate average position of neighbors
            DISTANCE = 0.1
            SPLINE_SAMPLING = 133
            average_of_neighbors = calc_average_of_neighbors(
                union_mesh,
                spline,
                spline_sampling=SPLINE_SAMPLING,
                distance=DISTANCE,
            )

            # Plot mesh
            plot_mesh_and_group(
                mesh=union_mesh,
                group=group,
                spline=spline,
                spacing=1,
                average_of_neighbors=average_of_neighbors,
            )

            # DISTANCE = 0.1
            # valid_neighbors, nearby_points = find_valid_neighbors_by_distance(
            #     union_mesh, group, distance=DISTANCE, distance_cutoff=0.75
            # )
            # # plot_neighbors(union_mesh, groups, groups_neighbors_valid, spacing=10)
            # neighbor_averages = calc_average_of_neighbors(union_mesh, valid_neighbors)
            # # plot_group_averages(union_mesh, groups, groups_avg, spacing=10)
            # # groups_neighbors_distance = calc_distance_to_neighbors(union_mesh, groups, groups_neighbors_valid)
            # blend_nearby_points(union_mesh, group, neighbor_averages, nearby_points, distance=DISTANCE)

    def plot_meshes(self):

        trimesh.Scene([ac.mesh for ac in self.ac_list]).show()

    def merge_meshes(self):

        merged_meshes = trimesh.boolean.union([ac.mesh for ac in self.ac_list], engine="scad")
        bf = trimesh.repair.broken_faces(merged_meshes)
        self.merged_meshes = merged_meshes
