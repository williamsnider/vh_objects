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


def calc_curvature_at_vertex(tree, mesh, v_idx, distance):
    neighbors = tree.query_ball_point(mesh.vertices[v_idx], distance)
    nonneighbors = np.ones(mesh.vertices.shape[0], dtype="bool")
    nonneighbors[neighbors] = 0

    # Transform to origin, align normal vector to (0,0,-1)
    verts = mesh.vertices
    verts = verts - verts[v_idx]  # Translate to origin
    R = calc_R_to_align_rows_of_vectors(mesh.vertex_normals[v_idx].reshape(-1, 1, 3), np.array([[[0, 0, -1]]]))
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
    # k1_arr[v_idx] = k1
    # k2_arr[v_idx] = k2

    # # Rotate k1_vec and k2_vec to R3
    k1_vec_rot = R[0] @ np.array([k1_vec[0], k1_vec[1], 0])
    k2_vec_rot = R[0] @ np.array([k2_vec[0], k2_vec[1], 0])
    return k1, k2, k1_vec_rot, k2_vec_rot


def calc_mesh_curvature(mesh, indices, distance):

    num_verts = len(indices)
    k1_arr = np.zeros(num_verts)
    k2_arr = np.zeros(num_verts)
    k1_vec_arr = np.zeros((num_verts, 3))
    k2_vec_arr = np.zeros((num_verts, 3))
    tree = scipy.spatial.cKDTree(mesh.vertices)

    for i, v_idx in enumerate(indices):

        # # Plot verts with principal directions
        k1, k2, k1_vec, k2_vec = calc_curvature_at_vertex(tree, mesh, v_idx, distance)
        k1_arr[i] = k1
        k2_arr[i] = k2
        k1_vec_arr[i] = k1_vec
        k2_vec_arr[i] = k2_vec
    return k1_arr, k2_arr, k1_vec_arr, k2_vec_arr


def plot_curvature(mesh, indices, k1_vec_arr):

    # Plotting
    scale = 0.1
    v = mesh.vertices[indices].reshape(-1, 1, 3)
    vK1 = k1_vec_arr.reshape(-1, 1, 3)

    K1_segs = np.hstack((v - scale * vK1, v + scale * vK1))
    K1_path = trimesh.load_path(K1_segs)
    K1_path.colors = np.tile(np.array([0, 255, 0, 255]), (len(K1_path.entities), 1))

    scene = trimesh.Scene()
    scene.add_geometry([mesh, K1_path])
    scene.show()

    # K2_segs = np.hstack((v - scale * vK2, v + scale * vK2))
    # K2_path = trimesh.load_path(K2_segs)
    # K2_path.colors = np.tile(np.array([[0, 0, 255, 255]]), (len(K2_path.entities), 1))

    # u_segs = np.hstack((v - scale * up_r, v + scale * up_r))
    # u_path = trimesh.load_path(u_segs)
    # u_path.colors = np.tile(np.array([[0, 255, 0, 255]]), (mesh.vertices.shape[0], 1))

    # v_segs = np.hstack((v - scale * vp_r, v + scale * vp_r))
    # v_path = trimesh.load_path(v_segs)
    # v_path.colors = np.tile(np.array([[0, 0, 255, 255]]), (mesh.vertices.shape[0], 1))


if __name__ == "__main__":

    # Plot mesh with principal directions
    # Vertices
    mesh = trimesh.creation.icosphere(4)
    mesh = mesh.apply_scale([1, 3, 5])
    distance = 0.5
    indices = np.arange(mesh.vertices.shape[0])[::11]
    k1_arr, k2_arr, k1_vec_arr, k2_vec_arr = calc_mesh_curvature(mesh, indices, distance)

    plot_curvature(mesh, indices, k1_vec_arr)
