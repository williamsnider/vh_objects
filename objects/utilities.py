# General functions used by different classes for objects project

import numpy as np
import matplotlib.pyplot as plt

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
# Plotting helper functions


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

    # TODO: Why is 611731 not in full_path

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
