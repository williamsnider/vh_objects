# General functions used by different classes for objects project

import numpy as np
import matplotlib.pyplot as plt
import trimesh
import networkx as nx

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
# Mesh Functions


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


def plot_projected_vertices_and_NNs(full_slice_yz, closest_NN, mesh_verts_yz, full_path):

    fig, ax = plt.subplots()
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_xlim([full_slice_yz.min(), full_slice_yz.max()])
    ax.set_ylim([full_slice_yz.min(), full_slice_yz.max()])

    y = full_slice_yz[:, 0]
    z = full_slice_yz[:, 1]
    ax.plot(y, z, "b*")

    # y = mesh_verts_yz[:, 0]
    # z = mesh_verts_yz[:, 1]
    # ax.plot(y, z, "k.")

    # Plot nearest neighbors
    for slice_i, NN_IDX in enumerate(closest_NN):

        # for mesh_i in NN_IDX:
        #     p1 = full_slice_yz[slice_i, :]
        #     p2 = mesh_verts_yz[mesh_i, :]

        #     x, y = zip(p1, p2)
        #     ax.plot(x, y, "g-")

        p1 = full_slice_yz[slice_i, :]
        p2 = mesh_verts_yz[NN_IDX, :]

        x, y = zip(p1, p2)
        ax.plot(x, y, "g-")

    # # Plot nearest neighbors
    # x, y = mesh_verts_yz[closest_NN].T
    # ax.plot(x.ravel(), y.ravel(), "r-")

    # # Plot slice points
    # x, y = full_slice_yz.T
    # ax.plot(x.ravel(), y.ravel(), "r-")

    # Plot shortest path
    x, y = mesh_verts_yz[full_path].T
    ax.plot(x, y, "r-")
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
