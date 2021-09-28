# General functions used by different classes for objects project

import numpy as np
import matplotlib.pyplot as plt
import pyembree
import trimesh
import networkx as nx
import scipy
import argparse
import numpy as np
from numpy.linalg import norm
import pymesh

##########
# B-Spline Functions


def open_uniform_knot_vector(num_cps, order):

    num_knots = num_cps + order
    knots = np.zeros(num_knots)
    knots[order:-order] = range(1, num_knots - 2 * order + 1)
    knots[-order:] = knots[-order - 1] + 1
    knots = knots / knots.max()  # Scale from 0 to 1
    return knots


##########
# Vector functions
def unit_vector(vector):
    """Returns the unit vector of the vector."""
    a = vector.ndim - 1
    return vector / np.linalg.norm(vector, axis=a, keepdims=True)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    Args:
        v1 (array): Vector 1.
        v2 (array): Vector 2
    Returns:
        angle (float): Radians
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    a = np.max([v1_u.ndim, v2_u.ndim]) - 1
    return np.arccos(np.clip((v1_u * v2_u).sum(axis=a), -1.0, 1.0))
    # if v1_u.ndim == 1 and v2_u.ndim == 1:
    #     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # elif v1_u.ndim == 2 or v2_u.ndim == 2:
    #     return np.arccos(np.clip((v1_u * v2_u).sum(axis=1), -1, 1))
    # else:
    #     raise NotImplementedError


##########
# Mesh Functions


def fix_mesh(mesh, detail="normal"):
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2
    print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        print(count)
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len, preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 10:
            break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh


def calc_face_normals(verts, faces):
    p0 = verts[faces[:, 0]]
    p1 = verts[faces[:, 1]]
    p2 = verts[faces[:, 2]]

    # Subtract to form vectors
    vec0 = p1 - p0
    vec1 = p2 - p0

    # Calculate cross product
    cross = np.cross(vec0, vec1)
    cross /= np.linalg.norm(cross, axis=1, keepdims=True)
    face_norms = cross

    return face_norms


def remove_subsurface_from_mesh(mesh, closed_path, point_in_subsurface_to_remove):

    edges = mesh.edges_unique

    # Create graph
    g = nx.Graph()
    g.add_edges_from(edges)
    g.remove_nodes_from(closed_path)  # Split along closed path
    connected_components = [c.copy() for c in nx.connected_components(g)]

    assert len(connected_components) == 2, "Mesh not split into 2 subsurfaces. closed_path likely not truly closed."

    # Choose subsurface containing the point to remove (we will delete this subsurface)
    for cc in connected_components:
        if point_in_subsurface_to_remove in cc:
            break
    else:
        raise ValueError("No subsurface contains the point_in_subsurface_to_remove")

    # Recreate graph, subtract out vertices of subsurface we are dleeting
    g = nx.Graph()
    g.add_edges_from(edges)
    g.remove_nodes_from(cc)

    # Convert subgraph into a mesh
    verts = mesh.vertices[g.nodes]

    # TODO: probably a faster way to do this w/o for loop
    face_indices = []
    for i, (v1, v2, v3) in enumerate(mesh.faces):
        if v1 not in g.nodes:
            continue
        elif v2 not in g.nodes:
            continue
        elif v3 not in g.nodes:
            continue
        else:
            face_indices.append(i)

    # Renumber vertices in faces
    old_indices = list(g.nodes)
    new_indices = np.arange(0, len(g.nodes))
    renumber_dict = {o: n for o, n in zip(old_indices, new_indices)}
    faces = mesh.faces[face_indices]
    faces = faces.ravel()
    faces = [renumber_dict[v] for v in faces]
    faces = np.array(faces)
    faces = faces.reshape((-1, 3))

    # Renumber vertices in closed_path
    closed_path_renumbered = [renumber_dict[v] for v in closed_path]
    face_norms = calc_face_normals(verts, faces)
    vert_norms = trimesh.geometry.mean_vertex_normals(verts.shape[0], faces, face_norms)

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        face_normals=face_norms,
        vertex_normals=vert_norms,
        process=False,
    )
    return mesh, closed_path_renumbered


def find_visible_vertices(mesh, position, indices="ALL"):
    """Uses simple ray-mesh queries to identify the vertices on a mesh that are visible (not occluded) from a certain position."""

    # TODO: accelerate using pyembree
    if indices == "ALL":
        origins = mesh.vertices
    else:
        origins = mesh.vertices[indices]
    directions = position - origins
    intersects = mesh.ray.intersects_any(
        ray_origins=origins,
        ray_directions=directions,
    )
    return ~intersects  # Negate to get visible vertices, not occluded


def distribute_indices(points1, points2):
    """Given two lists of vertices with unequal lengths, distribute them evenly."""
    idx1 = np.arange(points1.shape[0])
    idx2 = np.arange(points2.shape[0])

    # Identify which is longer and label as short or long
    num1 = len(idx1)
    num2 = len(idx2)

    if num1 < num2:
        l_idx = idx2
        s_idx = idx1
        l_pts = points2
        s_pts = points1
    if num2 < num1:
        l_idx = idx1
        s_idx = idx2
        l_pts = points1
        s_pts = points2
    else:
        l_idx = idx2
        s_idx = idx1
        l_pts = points2
        s_pts = points1

    ### Distribute short to long
    l_num = len(l_idx)
    s_num = len(s_idx)

    # Find the closest l_point to s_idx[0]
    pairings = np.zeros([l_num, 2], dtype="int")
    tree = scipy.spatial.KDTree(l_pts)
    dd, ii = tree.query(s_pts[0], k=1)
    l_idx_roll = np.roll(l_idx, -ii)

    # Use ratio of lengths of l and s to distribute
    s_l_ratio = s_num / l_num

    for i, l in enumerate(l_idx_roll):

        s_i = np.round(i * s_l_ratio).astype("int")
        s = s_idx[s_i]
        pairings[i] = [s, l]

    # Check results
    assert np.all(pairings[:, 1] == l_idx_roll), "Missing l_edge vertices."
    assert set(s_idx) == set(pairings[:, 0]), "Missing s_edge vertices."

    # Raise warning if rolling in other direction produces tighter pairings.
    distA = np.linalg.norm(s_pts[pairings[:, 0]] - l_pts[pairings[:, 1]], axis=1).sum()
    roll_pairings = np.roll(pairings[:, 1], -1)  # Roll back 1 so that 0th element is now at position -1
    flip_pairings = np.flip(roll_pairings)
    distB = np.linalg.norm(s_pts[pairings[:, 0]] - l_pts[flip_pairings], axis=1).sum()
    if distB < distA:
        raise Warning("Flipping the order of the long indices would have resulted in a shorter total distance.")

    # Ensure that pairings[:,0] corresponds to idx1/points1; flip columns if the short is actually points2
    if num2 < num1:
        pairings = np.roll(pairings, 1, axis=1)

    return pairings


##########
# Misc Functions
def flatten(groups):
    flattened = []
    for sublist in groups:
        for i in sublist:
            flattened.append(i)
    return flattened


def sliding_window_mean(arr, window_size, axis):

    assert window_size % 2 == 1, "window_size must be odd."

    big_arr = np.zeros(arr.shape + (window_size,))  # Add extra dimension along which we will average
    for idx in range(window_size):

        shift = (window_size - 1) // 2 - idx
        shifted = np.roll(arr, shift=shift, axis=axis)
        big_arr[..., idx] = shifted  # Ellipsis in python --> get last column. COOL!
        # big_arr[[slice(None)] * (big_arr.ndim - 1) + [idx]] = shifted  # More general solution

    return big_arr.mean(axis=-1)  # Average along the axis we added (the last one)


##########
# Plotting helper functions


def plot_projected_vertices_and_NNs_3D(full_slice, closest_NN, mesh_verts, full_path):

    full_slice = full_slice.squeeze()  # Remove singleton dimension

    fig, ax = plt.subplots()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)
    ax.set_xlim([full_slice.min(), full_slice.max()])
    ax.set_ylim([full_slice.min(), full_slice.max()])
    ax.set_zlim([full_slice.min(), full_slice.max()])

    x = full_slice[:, 0]
    y = full_slice[:, 1]
    z = full_slice[:, 2]
    ax.scatter(x, y, z, "b*")

    x = mesh_verts[::25, 0]
    y = mesh_verts[::25, 1]
    z = mesh_verts[::25, 2]
    ax.scatter(x, y, z, "k.")

    # Plot nearest neighbors
    for slice_i, NN_IDX in enumerate(closest_NN):

        # for mesh_i in NN_IDX:
        #     p1 = full_slice_yz[slice_i, :]
        #     p2 = mesh_verts_yz[mesh_i, :]

        #     x, y = zip(p1, p2)
        #     ax.plot(x, y, "g-")

        p1 = full_slice[slice_i, :]
        p2 = mesh_verts[NN_IDX, :]

        x, y, z = zip(p1, p2)
        ax.plot(x, y, z, "g-")

    # # Plot nearest neighbors
    # x, y = mesh_verts_yz[closest_NN].T
    # ax.plot(x.ravel(), y.ravel(), "r-")

    # # Plot slice points
    # x, y = full_slice_yz.T
    # ax.plot(x.ravel(), y.ravel(), "r-")

    # Plot shortest path
    x, y, z = mesh_verts[full_path].T
    ax.plot(x, y, z, "r-")
    plt.show()


def plot_projected_vertices_and_NNs(full_slice_yz, closest_NN, mesh_verts_yz_all, full_path):

    fig, ax = plt.subplots()
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    minx = np.min([full_slice_yz.min(), mesh_verts_yz_all[closest_NN].min()])
    maxx = np.max([full_slice_yz.max(), mesh_verts_yz_all[closest_NN].max()])
    ax.set_xlim([minx, maxx])
    ax.set_ylim([minx, maxx])

    y, z = full_slice_yz.T
    ax.plot(y, z, "b*")

    # Plot nearest neighbors
    for slice_i, NN_IDX in enumerate(closest_NN):

        p1 = mesh_verts_yz_all[NN_IDX]

        p1 = full_slice_yz[slice_i, :]
        p2 = mesh_verts_yz_all[NN_IDX, :]

        x, y = zip(p1, p2)
        ax.plot(x, y, "g-")

    # # Plot nearest neighbors
    x, y = mesh_verts_yz_all[closest_NN].T
    ax.plot(x.ravel(), y.ravel(), "r-")

    # # Plot slice points
    # x, y = full_slice_yz.T
    # ax.plot(x.ravel(), y.ravel(), "r-")

    # Plot shortest path
    x, y = mesh_verts_yz_all[full_path].T
    ax.plot(x, y, "r-")
    plt.show()


def plot_mesh_normals(mesh, indices, child_ac, slice_dist):

    SPACING = 100
    pts = mesh.vertices[indices][::SPACING]
    norms = mesh.vertex_normals[indices][::SPACING]

    # Plot mesh derivatives
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)

    x, y, z = pts.T
    ax.plot3D(x, y, z, "k.")

    for i, _ in enumerate(norms):

        # Normal
        p1 = pts[i]
        p2 = p1 + norms[i] * 0.1

        x, y, z = zip(p1, p2)
        ax.plot3D(x, y, z, "b-")

        # Line to centerpoint
        p1 = child_ac.r(slice_dist).T
        p2 = pts[i]
        x, y, z = zip(p1, p2)
        ax.plot3D(x, y, z, "r-")

    plt.show()


def plot_filtered_closest_NN(parent_mesh, closest_NN_wrapped, too_large):

    # Plot to verify
    SPACING = 30
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)

    # All
    x, y, z = parent_mesh.vertices[closest_NN_wrapped].T
    ax.plot(x, y, z, "-", color="k")

    # Valid
    x, y, z = parent_mesh.vertices[closest_NN_wrapped[~too_large]].T
    ax.plot(x, y, z, ".", color="green")

    # Invalid
    x, y, z = parent_mesh.vertices[closest_NN_wrapped[too_large]].T
    ax.plot(x, y, z, ".", color="red")

    invalid_idx = np.argwhere(too_large)
    for idx in invalid_idx:

        i = idx[0]
        p1 = parent_mesh.vertices[closest_NN_wrapped[i - 1]]
        p2 = parent_mesh.vertices[closest_NN_wrapped[i]]
        p3 = parent_mesh.vertices[closest_NN_wrapped[i + 1]]

        x, y, z = np.stack([p1, p2, p3], axis=0).T
        ax.plot(x, y, z, "-", color="red")

    x, y, z = parent_mesh.vertices[::SPACING].T
    ax.plot(x, y, z, ".k")

    plt.show()


def plot_smoothed_NN(parent_mesh, smoothed_NN, closest_NN_wrapped):

    # Plot to verify
    SPACING = 30
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)

    # Original
    x, y, z = parent_mesh.vertices[closest_NN_wrapped].T
    ax.plot(x, y, z, "-", color="k")

    # Smoothed
    x, y, z = parent_mesh.vertices[smoothed_NN].T
    ax.plot(x, y, z, "-", color="green")

    x, y, z = parent_mesh.vertices[::SPACING].T
    ax.plot(x, y, z, ".k")

    plt.show()


def plot_mesh_derivatives(parent_mesh, child_ac, full_path, c_V, c_T):

    # Plot mesh derivatives
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)

    x, y, z = child_ac.verts[::100].T
    ax.plot3D(x, y, z, "k.")

    for i in range(c_V.shape[0]):

        p1 = c_V[i]
        p2 = p1 + c_T[i] * 0.1

        x, y, z = zip(p1, p2)
        ax.plot3D(x, y, z, "b-")

    # Plot parent
    x, y, z = parent_mesh.vertices[::100].T
    ax.plot3D(x, y, z, "r.")
    x, y, z = parent_mesh.vertices[full_path].T
    ax.plot3D(x, y, z, "g-")
    plt.show()


def plot_surface_linking_axial_components(parent_mesh, child_ac, surface):

    # Surface
    (us, vs) = surface.start()
    (ue, ve) = surface.end()
    uu = np.linspace(us, ue, 20)
    vv = np.linspace(vs, ve, 20)
    grid = surface(uu, vv)
    sx, sy, sz = grid.reshape(-1, surface.dimension).T
    cx, cy, cz = surface.controlpoints.T

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.view_init(elev=-90, azim=90)
    # ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "k.")
    ax.plot3D(cx.ravel(), cy.ravel(), cz.ravel(), "g-")

    # Child
    x, y, z = child_ac.verts.T
    ax.plot3D(x, y, z, "k.")

    # Parent
    x, y, z = parent_mesh.vertices.T
    ax.plot3D(x, y, z, "r.")

    # Surface
    ax.plot3D(sx, sy, sz, "b.")

    plt.show()


def plot_mesh_vertices_and_normals(mesh):

    # Plot mesh vertices and normals
    verts = mesh.vertices
    norms = mesh.vertex_normals

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)

    # Vertices
    x, y, z = verts.T
    ax.plot3D(x, y, z, "k.")

    # Normals
    for i in range(norms.shape[0]):

        p1 = verts[i]
        p2 = p1 + norms[i] * 0.1

        x, y, z = zip(p1, p2)
        ax.plot3D(x, y, z, "b-")

    # Plot parent
    plt.show()


def plot_child_and_junction_edges(child_mesh, child_edge_idx, junction_mesh, junction_edge_idx, plot_linkages=True):

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)

    # Child
    x, y, z = child_mesh.vertices.T
    ax.plot3D(x, y, z, "g.")
    x, y, z = child_mesh.vertices[child_edge_idx].T
    ax.plot3D(x, y, z, "k.")

    # Junction
    x, y, z = junction_mesh.vertices.T
    ax.plot3D(x, y, z, "b.")
    x, y, z = junction_mesh.vertices[junction_edge_idx].T
    ax.plot3D(x, y, z, "r.")

    # Plot linkages of these points
    if plot_linkages is True:
        for i in range(len(child_edge_idx)):

            p1 = child_mesh.vertices[child_edge_idx[i]]
            p2 = junction_mesh.vertices[junction_edge_idx[i]]

            x, y, z = zip(p1, p2)
            ax.plot3D(x, y, z, "-y")

    plt.show()


def plot_parent_and_child_edges(parent_mesh, child_mesh, pairings, plot_linkages=True):

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)

    # # Parent
    # x, y, z = parent_mesh.vertices.T
    # ax.plot3D(x, y, z, "g.")

    # # Junction
    # x, y, z = child_mesh.vertices.T
    # ax.plot3D(x, y, z, "b.")

    # Plot linkages of these points
    if plot_linkages is True:
        for [p_i, c_i] in pairings:

            p1 = parent_mesh.vertices[p_i]
            p2 = child_mesh.vertices[c_i]

            x, y, z = zip(p1, p2)
            ax.plot3D(x, y, z, "-k")

    # Plot linkages between edges
    for i in [0, 1]:

        items = pairings[:, i]
        _, idx = np.unique(items, return_index=True)
        items = items[np.sort(idx)]

        items_wrapped = np.zeros([items.shape[0] + 1], dtype="int")
        items_wrapped[:-1] = items
        items_wrapped[-1] = items[0]

        if i == 0:
            points = parent_mesh.vertices[items_wrapped]
            x, y, z = points.T
            ax.plot3D(x, y, z, "-g")
        else:
            points = child_mesh.vertices[items]
            x, y, z = points.T
            ax.plot3D(x, y, z, "-b")

    plt.show()


def plot_parent_and_child_faces(s_mesh, l_mesh, faces, s_edge, l_edge):

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=-90, azim=90)

    # Parent
    x, y, z = s_mesh.vertices.T
    ax.plot3D(x, y, z, "g.")

    # Junction
    x, y, z = l_mesh.vertices.T
    ax.plot3D(x, y, z, "b.")

    for [v1, v2, v3] in faces:

        # Check that indices are not overlapping - sloppy but just to plot quickly
        for v in [v1, v2, v3]:

            if v in s_edge & v in l_edge:
                raise NotImplementedError

        if v1 in s_edge:
            p1 = s_mesh.vertices[v1]
        else:
            p1 = l_mesh.vertices[v1]

        if v2 in s_edge:
            p2 = s_mesh.vertices[v2]
        else:
            p2 = l_mesh.vertices[v2]

        if v3 in s_edge:
            p3 = s_mesh.vertices[v3]
        else:
            p3 = l_mesh.vertices[v3]

        x, y, z = np.stack([p1, p2, p3, p1], axis=0).T
        ax.plot3D(x, y, z, "k-")
    plt.show()
