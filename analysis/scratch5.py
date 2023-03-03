import numpy as np
import scipy.linalg
import scipy.spatial
import matplotlib.pyplot as plt
import trimesh


def calc_R_to_align_rows_of_vectors(a, b):
    """Calculates the rotation matrix R to rotate row[i] of matrix b (vector b) onto row[i] of matrix a (vector a).

    R formula from https://math.stackexchange.com/a/2672702"""

    # a and b must have shape num_rows x 1 x 3
    assert a.shape[1:] == b.shape[1:] == (1, 3)
    assert a.shape[0] == b.shape[0]

    # Normalize rows
    a_norm = np.linalg.norm(a, axis=2, keepdims=True)
    a = a / a_norm
    b_norm = np.linalg.norm(b, axis=2, keepdims=True)
    b = b / b_norm

    num_rows = a.shape[0]
    k = a + b
    kt = k.reshape(num_rows, 3, 1)  # transposed
    I = np.repeat(np.eye(3).reshape(1, 3, 3), num_rows, axis=0)

    R_stack = 2 * kt @ k / (k @ kt) - I

    return R_stack


# Vertices
mesh = trimesh.creation.icosphere(4)
mesh = mesh.apply_scale([1, 3, 5])

num_verts = mesh.vertices.shape[0]
k1_arr = np.zeros(num_verts)
k2_arr = np.zeros(num_verts)
k1_vec_arr = np.zeros((num_verts, 3))
k2_vec_arr = np.zeros((num_verts, 3))
tree = scipy.spatial.cKDTree(mesh.vertices)

for v_idx in range(num_verts):
    neighbors = tree.query_ball_point(mesh.vertices[v_idx], 0.5)
    nonneighbors = np.ones(mesh.vertices.shape[0], dtype="bool")
    nonneighbors[neighbors] = 0

    # Transform to origin, align normal vector to (0,0,-1)
    verts = mesh.vertices
    verts = verts - verts[v_idx]  # Translate to origin
    R = calc_R_to_align_rows_of_vectors(
        mesh.vertex_normals[v_idx].reshape(-1, 1, 3), np.array([[[0, 0, -1]]])
    )
    verts = verts @ R[0]  # Align normal vector to (0,0,-1)

    data = verts[neighbors]

    # best-fit quadratic surface (z = L/2*x^2, M*xy, N/2*y^2)
    A = np.c_[
        np.ones(data.shape[0]),
        data[:, :2],
        np.prod(data[:, :2], axis=1),
        data[:, :2] ** 2,
    ]
    A = np.c_[
        1 / 2 * data[:, 0] ** 2,
        data[:, 0] * data[:, 1],
        1 / 2 * data[:, 1] ** 2,
    ]
    b = data[:, 2]
    x, _, _, _ = scipy.linalg.lstsq(A, b)

    # Not sure if this is accurate
    L = x[0]
    M = x[1]
    N = x[2]

    II = np.array([[L, M], [M, N]])
    w, v = np.linalg.eig(II)
    principal_curvatures = w
    principal_directions = v.T  # np returns eigenvectors as columns

    # Assign largest principal component as K1
    k1_mask = principal_curvatures.argmax()
    k1 = principal_curvatures[k1_mask]
    k1_vec = principal_directions[k1_mask]
    k2 = principal_curvatures[~k1_mask]
    k2_vec = principal_directions[~k1_mask]

    # Store values
    k1_arr[v_idx] = k1
    k2_arr[v_idx] = k2

    # Rotate k1_vec and k2_vec to R3
    k1_vec_arr[v_idx] = R[0] @ np.array([k1_vec[0], k1_vec[1], 0])
    k2_vec_arr[v_idx] = R[0] @ np.array([k2_vec[0], k2_vec[1], 0])

    # # Plot verts with principal directions

    if v_idx == 100:

        print(L, M, N)
        # evaluate it on a grid
        X, Y = np.meshgrid(
            np.linspace(data[:, 0].min(), data[:, 0].max(), 10),
            np.linspace(data[:, 1].min(), data[:, 1].max(), 10),
        )
        XX = X.flatten()
        YY = Y.flatten()
        Z = np.dot(
            np.c_[
                1 / 2 * XX**2,
                XX * YY,
                1 / 2 * YY**2,
            ],
            x,
        ).reshape(X.shape)
        Z = np.dot(
            np.c_[
                1 / 2 * XX**2,
                XX * YY,
                1 / 2 * YY**2,
            ],
            np.array([L, M, N]),
        ).reshape(X.shape)

        scale = 0.1
        N_pts = np.array(
            [
                [verts[v_idx]],
                [verts[v_idx] + scale * mesh.vertex_normals[v_idx]],
            ]
        )
        K1_pts = np.array(
            [
                verts[v_idx] - scale * np.array([k1_vec[0], k1_vec[1], 0]),
                verts[v_idx] + scale * np.array([k1_vec[0], k1_vec[1], 0]),
            ]
        )
        K2_pts = np.array(
            [
                verts[v_idx] - scale * np.array([k2_vec[0], k2_vec[1], 0]),
                verts[v_idx] + scale * np.array([k2_vec[0], k2_vec[1], 0]),
            ]
        )
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

        ax.plot(K1_pts[:, 0], K1_pts[:, 1], K1_pts[:, 2], "-g")
        ax.plot(K2_pts[:, 0], K2_pts[:, 1], K2_pts[:, 2], "-b")

        # ax.plot(
        #     verts[nonneighbors, 0],
        #     verts[nonneighbors, 1],
        #     verts[nonneighbors, 2],
        #     "k.",
        # )
        ax.plot(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            "b.",
        )
        ax.plot(
            verts[v_idx, 0],
            verts[v_idx, 1],
            verts[v_idx, 2],
            "r.",
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        ax.set_zlabel("Z")
        ax.axis("equal")
        ax.axis("tight")
        plt.show()
        break


# Plot mesh with principal directions
scale = 0.1
v = mesh.vertices.reshape(-1, 1, 3)
vK1 = k1_vec_arr.reshape(-1, 1, 3)

K1_segs = np.hstack((v - scale * vK1, v + scale * vK1))
K1_path = trimesh.load_path(K1_segs)
K1_path.colors = np.tile(np.array([0, 255, 0, 255]), (len(K1_path.entities), 1))

# K2_segs = np.hstack((v - scale * vK2, v + scale * vK2))
# K2_path = trimesh.load_path(K2_segs)
# K2_path.colors = np.tile(np.array([[0, 0, 255, 255]]), (len(K2_path.entities), 1))

# u_segs = np.hstack((v - scale * up_r, v + scale * up_r))
# u_path = trimesh.load_path(u_segs)
# u_path.colors = np.tile(np.array([[0, 255, 0, 255]]), (mesh.vertices.shape[0], 1))

# v_segs = np.hstack((v - scale * vp_r, v + scale * vp_r))
# v_path = trimesh.load_path(v_segs)
# v_path.colors = np.tile(np.array([[0, 0, 255, 255]]), (mesh.vertices.shape[0], 1))

# scene = trimesh.Scene()
# scene.add_geometry([mesh, K1_path])
# scene.show()

# Plot k1/k2


# # evaluate it on a grid
# X, Y = np.meshgrid(
#     np.linspace(data[:, 0].min(), data[:, 0].max(), 10),
#     np.linspace(data[:, 1].min(), data[:, 1].max(), 10),
# )
# XX = X.flatten()
# YY = Y.flatten()
# Z = np.dot(
#     np.c_[
#         1 / 2 * XX**2,
#         XX * YY,
#         1 / 2 * YY**2,
#     ],
#     x,
# ).reshape(X.shape)

# # plot points and fitted surface
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
# ax.plot(
#     data[:, 0],
#     data[:, 1],
#     data[:, 2],
#     "b.",
# )
# ax.plot(
#     verts[v_idx, 0],
#     verts[v_idx, 1],
#     verts[v_idx, 2],
#     "r.",
# )
# plt.xlabel("X")
# plt.ylabel("Y")
# ax.set_zlabel("Z")
# ax.axis("equal")
# ax.axis("tight")
# plt.show()
