from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [0.1, 0.1],
    ]
)

# Plot tangent vectors
def plot_tangent_vectors():
    cs = CrossSection(base_cp, 0.0)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs])
    t = np.linspace(0, 1, 3)
    v = np.linspace(0, 1, 51)
    r = ac.r(v)
    T = ac.T(t)
    N = ac.N(t)
    B = ac.B(t)

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
    ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "k.")

    # Plot lines
    for i, v in enumerate([T, N, B]):
        color = ["red", "green", "blue"][i]
        for j, t_val in enumerate(t):

            p0 = ac.r(t_val)
            p1 = p0 + v[j, :] * 0.1
            line = np.stack([p0, p1], axis=0)
            x, y, z = line.T
            ax.plot3D(x[0], y[0], z[0], "-", color=color)
    plt.show()


def plot_controlpoints():
    cs0 = CrossSection(base_cp * 0.5, position=0.3, tilt=np.pi / 4)
    cs1 = CrossSection(base_cp * 0.5, position=0.7, rotation=np.pi)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs0, cs1])

    # Controlpoints
    ac.get_controlpoints()
    cp = ac.controlpoints
    x = cp[:, :, 0].ravel()
    y = cp[:, :, 1].ravel()
    z = cp[:, :, 2].ravel()

    # Backbone
    v = np.linspace(0, 1, 51)
    r = ac.r(v)

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
    ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "k.")
    ax.plot3D(x, y, z, "g-")
    plt.show()


def plot_surface():
    cs0 = CrossSection(base_cp * 0.5, position=0.3, tilt=np.pi / 4)
    cs1 = CrossSection(base_cp * 0.5, position=0.7, rotation=np.pi)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs0, cs1])
    ac.get_controlpoints()
    ac.make_surface()

    # Backbone
    v = np.linspace(0, 1, 51)
    r = ac.r(v)

    # Surface
    uu = np.linspace(0, 1, 20)
    grid = ac.surface(uu, uu)
    sx, sy, sz = grid.reshape(-1, ac.surface.dimension).T
    cx, cy, cz = ac.surface.controlpoints.T

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
    # ax.plot3D(cx.ravel(), cy.ravel(), cz.ravel(), "g.")
    ax.plot3D(sx, sy, sz, "r-")

    plt.show()

    ac.mesh.show()


def plot_face_normals():
    cs0 = CrossSection(base_cp * 0.5, 0.3)
    cs1 = CrossSection(base_cp * 0.5, 0.7)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs0, cs1])
    ac.get_controlpoints()
    ac.make_surface()
    ac.make_mesh()

    faces = ac.faces
    face_norms = ac.face_norms

    # Calculate midpoint of each face
    verts = ac.verts[faces]
    midpoints = np.mean(verts, axis=1)

    sx, sy, sz = ac.verts.T

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
    ax.plot3D(sx, sy, sz, "k.")

    for i, _ in enumerate(faces):

        if i % 10 != 0:
            continue

        fx, fy, fz = midpoints[i]
        nx, ny, nz = face_norms[i] * 0.1
        x = [fx, fx + nx]
        y = [fy, fy + ny]
        z = [fz, fz + nz]

        ax.plot3D(x, y, z, "b-")
        # ax.plot3D(fx, fy, fz, "k.")

    plt.show()


def plot_vertex_normals():
    cs0 = CrossSection(base_cp * 0.5, 0.3)
    cs1 = CrossSection(base_cp * 0.5, 0.7)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs0, cs1])
    ac.get_controlpoints()
    ac.make_surface()
    ac.make_mesh()

    verts = ac.verts
    vert_norms = ac.vert_norms

    x, y, z = verts.T
    nx, ny, nz = vert_norms.T

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
    ax.plot3D(x, y, z, "k.")

    for i, [nx, ny, nz] in enumerate(vert_norms):

        if i % 10 != 0:
            continue

        vx, vy, vz = verts[i]

        x = [vx, vx + nx * 0.1]
        y = [vy, vy + ny * 0.1]
        z = [vz, vz + nz * 0.1]

        ax.plot3D(x, y, z, "b-")

    plt.show()


def plot_align_axial_components():
    cs0 = CrossSection(base_cp * 0.5, 0.3)
    cs1 = CrossSection(base_cp * 0.5, 0.7)
    ac1 = AxialComponent(2 * np.pi * 1 * 0.25, curvature=0, cross_sections=[cs0, cs1])
    ac2 = AxialComponent(
        2 * np.pi * 1 * 0.25,
        curvature=1 / 1,
        cross_sections=[cs0, cs1],
        parent_axial_component=ac1,
        position_along_parent=0.75,
        position_along_self=0.25,
    )
    ac3 = AxialComponent(
        2 * np.pi * 1 * 1,
        curvature=1 / 4,
        cross_sections=[cs0, cs1],
        parent_axial_component=ac2,
        position_along_parent=0.75,
        position_along_self=0.25,
    )

    # Plot backbones
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.view_init(elev=-90, azim=90)

    # Plot lines
    for ac in [ac1, ac2, ac3]:

        t = np.linspace(0, 1, 3)
        v = np.linspace(0, 1, 51)
        r = ac.r(v)
        T = ac.T(t)
        N = ac.N(t)
        B = ac.B(t)

        if ac == ac1:
            ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "k.")

            # Plot controlpoints
            cp = ac.controlpoints
            x = cp[:, :, 0].ravel()
            y = cp[:, :, 1].ravel()
            z = cp[:, :, 2].ravel()
            ax.plot3D(x, y, z, "k-")

        if ac == ac2:
            ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "y.")

            # Plot controlpoints
            cp = ac.controlpoints
            x = cp[:, :, 0].ravel()
            y = cp[:, :, 1].ravel()
            z = cp[:, :, 2].ravel()
            ax.plot3D(x, y, z, "y-")

        if ac == ac3:
            ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "c.")

            # Plot controlpoints
            cp = ac.controlpoints
            x = cp[:, :, 0].ravel()
            y = cp[:, :, 1].ravel()
            z = cp[:, :, 2].ravel()
            ax.plot3D(x, y, z, "c-")

        for i, v in enumerate([T, N, B]):
            color = ["red", "green", "blue"][i]
            for j, t_val in enumerate(t):

                p0 = ac.r(t_val)
                p1 = p0 + v[j, :] * 0.1
                line = np.stack([p0, p1], axis=0)
                x, y, z = line.T
                ax.plot3D(x[0], y[0], z[0], "-", color=color)

    plt.show()


def plot_euler_angles():

    cs0 = CrossSection(base_cp * 0.5, 0.3)
    cs1 = CrossSection(base_cp * 0.5, 0.7)
    ac1 = AxialComponent(2 * np.pi * 1 * 0.25, curvature=0, cross_sections=[cs0, cs1])
    ac2 = AxialComponent(
        2 * np.pi * 1 * 0.25,
        curvature=1 / 1,
        cross_sections=[cs0, cs1],
        parent_axial_component=ac1,
        position_along_parent=0.75,
        position_along_self=0.25,
        euler_angles=np.array([0, 0, 0]),
    )
    ac3 = AxialComponent(
        2 * np.pi * 1 * 1,
        curvature=1 / 2,
        cross_sections=[cs0, cs1],
        parent_axial_component=ac2,
        position_along_parent=0.75,
        position_along_self=0.25,
        euler_angles=np.array([0, 0, np.pi]),
    )

    # Plot backbones
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.view_init(elev=-90, azim=90)

    # Plot lines
    for ac in [ac1, ac2, ac3]:

        t = np.linspace(0, 1, 3)
        v = np.linspace(0, 1, 51)
        r = ac.r(v)
        T = ac.T(t)
        N = ac.N(t)
        B = ac.B(t)

        if ac == ac1:
            ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "k.")

            # Plot controlpoints
            cp = ac.controlpoints
            x = cp[:, :, 0].ravel()
            y = cp[:, :, 1].ravel()
            z = cp[:, :, 2].ravel()
            ax.plot3D(x, y, z, "k-")

        if ac == ac2:
            ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "y.")

            # Plot controlpoints
            cp = ac.controlpoints
            x = cp[:, :, 0].ravel()
            y = cp[:, :, 1].ravel()
            z = cp[:, :, 2].ravel()
            ax.plot3D(x, y, z, "y-")

        if ac == ac3:
            ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "c.")

            # Plot controlpoints
            cp = ac.controlpoints
            x = cp[:, :, 0].ravel()
            y = cp[:, :, 1].ravel()
            z = cp[:, :, 2].ravel()
            ax.plot3D(x, y, z, "c-")

        for i, v in enumerate([T, N, B]):
            color = ["red", "green", "blue"][i]
            for j, t_val in enumerate(t):

                p0 = ac.r(t_val)
                p1 = p0 + v[j, :] * 0.1
                line = np.stack([p0, p1], axis=0)
                x, y, z = line.T
                ax.plot3D(x[0], y[0], z[0], "-", color=color)

    plt.show()


def plot_meshes_as_shape():
    cs0 = CrossSection(base_cp * 0.5, 0.3)
    cs1 = CrossSection(base_cp * 0.5, 0.7)
    ac1 = AxialComponent(2 * np.pi * 1 * 0.25, curvature=0, cross_sections=[cs0, cs1])
    ac2 = AxialComponent(
        2 * np.pi * 1 * 0.25,
        curvature=1 / 1,
        cross_sections=[cs0, cs1],
        parent_axial_component=ac1,
        position_along_parent=0.75,
        position_along_self=0.25,
        euler_angles=np.array([0, np.pi / 3, 0]),
    )
    ac3 = AxialComponent(
        2 * np.pi * 1 * 1,
        curvature=1 / 2,
        cross_sections=[cs0, cs1],
        parent_axial_component=ac2,
        position_along_parent=0.75,
        position_along_self=0.25,
        euler_angles=np.array([0, 0, np.pi]),
    )
    s = Shape([ac1, ac2, ac3])
    s.merge_meshes()
    s.merged_meshes.show()


if __name__ == "__main__":
    plot_tangent_vectors()
    plot_controlpoints()
    plot_surface()
    plot_face_normals()
    plot_vertex_normals()
    plot_align_axial_components()
    plot_euler_angles()
    # plot_meshes_as_shape()
