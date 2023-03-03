import numpy as np
import trimesh
from scipy.sparse import coo_matrix
import scipy.linalg
import scipy.sparse


def calc_face_normals(mesh):
    """Calculates normals of faces."""
    return mesh.face_normals


def calc_vertex_normals(mesh):
    """Calculates vertex normals.

    For each vertex, the contribution of each neighboring face's normal is weighted by the face's area divided by the squares of the two sides touching the vertex."""

    def summed_sparse(vertex_count, faces, face_weights, face_normals):
        # use a sparse matrix of which face contains each vertex to
        # figure out the summed normal at each vertex
        # allow cached sparse matrix to be passed
        # fill the matrix with vertex-corner angles as weights
        corner_angles = face_weights[
            np.repeat(np.arange(len(faces)), 3), np.argsort(faces, axis=1).ravel()
        ]
        # create a sparse matrix
        matrix = trimesh.geometry.index_sparse(vertex_count, faces).astype(np.float64)
        # assign the corner angles to the sparse matrix data
        matrix.data = corner_angles

        return matrix.dot(face_normals)

    # Edge vectors
    e0 = mesh.vertices[mesh.faces[:, 2], :] - mesh.vertices[mesh.faces[:, 1], :]
    e1 = mesh.vertices[mesh.faces[:, 0], :] - mesh.vertices[mesh.faces[:, 2], :]
    e2 = mesh.vertices[mesh.faces[:, 1], :] - mesh.vertices[mesh.faces[:, 0], :]

    de0 = np.linalg.norm(e0, axis=1)
    de1 = np.linalg.norm(e1, axis=1)
    de2 = np.linalg.norm(e2, axis=1)

    face_areas = np.linalg.norm(np.cross(e0, e1), axis=1) / 2
    wmesh1 = face_areas / ((de1**2) * (de2**2))
    wmesh2 = face_areas / ((de0**2) * (de2**2))
    wmesh3 = face_areas / ((de1**2) * (de0**2))

    face_normals = mesh.face_normals
    vertex_count = len(mesh.vertices)
    faces = mesh.faces
    face_weights = np.array([wmesh1, wmesh2, wmesh3]).T

    # normals should be unit vectors
    face_ok = (face_normals**2).sum(axis=1) > 0.5
    # don't consider faces with invalid normals
    faces = faces[face_ok]
    face_normals = face_normals[face_ok]
    face_weights = face_weights[face_ok]

    result = trimesh.util.unitize(
        summed_sparse(vertex_count, faces, face_weights, face_normals)
    )

    vertex_normals = result / np.linalg.norm(result, axis=1, keepdims=True)
    return vertex_normals


def calc_triangle_area(P, Q, R):
    """Calculates area of a triangle from 3 vertices."""
    area = np.linalg.norm(np.cross(Q - P, R - P), axis=1) / 2
    return area


def calc_edge_vectors(mesh):
    """Calculates the edges of each face."""

    # Edge vectors
    e0 = mesh.vertices[mesh.faces[:, 2], :] - mesh.vertices[mesh.faces[:, 1], :]
    e1 = mesh.vertices[mesh.faces[:, 0], :] - mesh.vertices[mesh.faces[:, 2], :]
    e2 = mesh.vertices[mesh.faces[:, 1], :] - mesh.vertices[mesh.faces[:, 0], :]

    # Normalized edge vectors
    e0_norm = e0 / np.linalg.norm(e0, axis=1, keepdims=True)
    e1_norm = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2_norm = e2 / np.linalg.norm(e2, axis=1, keepdims=True)

    # Edge lengths
    de0 = np.linalg.norm(e0, axis=1)
    de1 = np.linalg.norm(e1, axis=1)
    de2 = np.linalg.norm(e2, axis=1)

    return e0, e1, e2, e0_norm, e1_norm, e2_norm, de0, de1, de2


def calc_voronoi_area(P, Q, R, ang_Q, ang_R):
    """Calculates the area of a triangle closest to point P.

    This is accurate only for non-obtuse triangles."""
    PR = np.linalg.norm(R - P, axis=1)
    PQ = np.linalg.norm(Q - P, axis=1)
    area = 1 / 8 * (PR**2 / np.tan(ang_Q) + PQ**2 / np.tan(ang_R))
    return area


def calc_face_voronoi_areas(mesh):
    """Calculates the area of a face closest to each of the three vertices.

    See Meyer 2002."""

    verts_by_faces = mesh.vertices[mesh.faces]
    P = verts_by_faces[:, 0, :]
    Q = verts_by_faces[:, 1, :]
    R = verts_by_faces[:, 2, :]
    ang_P, ang_Q, ang_R = mesh.face_angles.T

    triangle_areas = calc_triangle_area(P, Q, R)
    face_voronoi_areas = np.vstack(
        [
            calc_voronoi_area(P, Q, R, ang_Q, ang_R),
            calc_voronoi_area(Q, P, R, ang_P, ang_R),
            calc_voronoi_area(R, Q, P, ang_Q, ang_P),
        ]
    ).T
    assert np.all(np.isclose(triangle_areas, face_voronoi_areas.sum(axis=1)))
    return face_voronoi_areas


def adjust_obtuse_face_voronoi_areas(mesh, face_areas, face_voronoi_areas):
    """Adjusts the voronoi areas for obtuse faces.

    See Meyer 2002."""
    obtuse_verts = mesh.face_angles >= np.pi / 2
    obtuse_faces = obtuse_verts.max(axis=1)
    nonobtuse_verts = obtuse_verts[obtuse_faces].argsort(axis=1)[
        :, :2
    ]  # indices (two from 0,1,2) of the non-obtuse vertices of an obtuse triangle

    adjusted_voronoi_areas = face_voronoi_areas.copy()
    adjusted_voronoi_areas[obtuse_verts] = face_areas[obtuse_faces] / 2
    adjusted_voronoi_areas[obtuse_faces, nonobtuse_verts.T] = (
        face_areas[obtuse_faces] / 4
    )
    return adjusted_voronoi_areas


def initialize_vertex_coordinate_system(mesh, vertex_normals):
    """Initializes the vertex coordinate system (u, v) in the tanget plane of the vertex."""

    edge1 = np.zeros(mesh.vertices.shape)

    # We need to calculate a vector from each vertex to one of its neighbors. Using mesh.vertex_neighbors was slow for large meshes. Instead, calculate an edge between v1/v2, v2/v3, and v3/v1 for each face. Then use the vertex index to populate edge1. This approach is "wasteful" in that some vertices will have multiple edges calculated (since vertices appear in multiple faces), and these edges will be overwritten in the for loop. Regardless this is much faster than using mesh.vertex_neighbors.
    face_verts = mesh.vertices[mesh.faces]
    face_verts_roll = np.roll(face_verts, 1, axis=1)
    edges = face_verts_roll - face_verts
    for i in range(3):
        edge1[mesh.faces[:, i]] = edges[:, i, :]

    up = edge1 / np.linalg.norm(edge1, axis=1, keepdims=True)

    up = np.cross(up, vertex_normals)
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    vp = np.cross(up, vertex_normals)

    return up, vp


def construct_vertex_x_face_sparse_matrix(mesh):
    """Constructs a sparse matrix with shape NUM_VERTICES x NUM_FACES.

    inputs:
        mesh
    outputs:
        sparseMatrix - empty sparse matrix with shape NUM_VERTICES x NUM_FACES
        f_indices - face index used to reference data with the mesh.faces structure
        col - the column in which the vertex of interest is located in the mesh.faces array (shape: NUM_FACES x (v1,v2,v3)). In short, this is used to obtain the data for the sparseMatrix by referencing data_array[f_indices, col], given that data_array has the mesh.faces structure.

    A sparse matrix is more efficient because most vertices are part of only a few faces. This is useful when e.g. summing the contribution of each face onto each vertex."""

    f_indices = mesh.vertex_faces.ravel()  # Faces touching each vertex (padded with -1)
    v_increasing = np.arange(len(mesh.vertices)).reshape(-1, 1)
    v_indices = np.repeat(v_increasing, mesh.vertex_faces.shape[1], axis=1).ravel()

    # Remove padded values and convert to 1D
    padded_values = f_indices == -1
    f_indices = f_indices[~padded_values]
    v_indices = v_indices[~padded_values]

    # In the mesh.faces array (shape: NUM_FACES x (v1,v2,v3)), find which column the vertex appears in
    _, col = np.where(
        np.repeat(v_indices.reshape(-1, 1), 3, axis=1) == mesh.faces[f_indices]
    )

    sparseMatrix = coo_matrix((np.zeros(len(v_indices)), (v_indices, f_indices)))

    return sparseMatrix, f_indices, col


def calc_face_weights(mesh, adjusted_voronoi_areas):

    sparseMatrix, f_indices, col = construct_vertex_x_face_sparse_matrix(mesh)
    data = adjusted_voronoi_areas[f_indices, col]
    sparseMatrix.data = data

    # Sum along face axis
    sum_of_weights_per_vertex = sparseMatrix.sum(axis=1).A.ravel()
    wfp = adjusted_voronoi_areas / sum_of_weights_per_vertex[mesh.faces]

    return wfp


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


def project_curvature_tensor(mesh, II_f, uf, vf, new_u_arr, new_v_arr):

    u1_arr = np.zeros((mesh.faces.shape[0], 3, 1))
    v1_arr = np.zeros((mesh.faces.shape[0], 3, 1))
    u2_arr = np.zeros((mesh.faces.shape[0], 3, 1))
    v2_arr = np.zeros((mesh.faces.shape[0], 3, 1))

    Lv_arr = np.zeros((mesh.faces.shape[0], 3))
    Mv_arr = np.zeros((mesh.faces.shape[0], 3))
    Nv_arr = np.zeros((mesh.faces.shape[0], 3))

    for i in range(3):
        u1_arr[:, i, :] = (
            new_u_arr[:, i, :].reshape(-1, 1, 3) @ uf.reshape(-1, 3, 1)
        ).reshape(-1, 1)
        v1_arr[:, i, :] = (
            new_u_arr[:, i, :].reshape(-1, 1, 3) @ vf.reshape(-1, 3, 1)
        ).reshape(-1, 1)
        u2_arr[:, i, :] = (
            new_v_arr[:, i, :].reshape(-1, 1, 3) @ uf.reshape(-1, 3, 1)
        ).reshape(-1, 1)
        v2_arr[:, i, :] = (
            new_v_arr[:, i, :].reshape(-1, 1, 3) @ vf.reshape(-1, 3, 1)
        ).reshape(-1, 1)

        u1_b = u1_arr[:, :1, :]
        v1_b = v1_arr[:, :1, :]
        u2_b = u2_arr[:, :1, :]
        v2_b = v2_arr[:, :1, :]

        Lv_b = np.dstack([u1_b, v1_b]) @ II_f @ np.hstack([u1_b, v1_b])
        Mv_b = np.dstack([u1_b, v1_b]) @ II_f @ np.hstack([u2_b, v2_b])
        Nv_b = np.dstack([u2_b, v2_b]) @ II_f @ np.hstack([u2_b, v2_b])

        Lv_arr[:, i] = Lv_b.ravel()
        Mv_arr[:, i] = Mv_b.ravel()
        Nv_arr[:, i] = Nv_b.ravel()

    return Lv_arr, Mv_arr, Nv_arr


def align_coordinate_systems(mesh, up, vp, nf):
    """Aligns the vertex coordinate system (up,vp) with the plane defined by the face's normal vector nf."""
    npp = np.cross(up, vp) / np.linalg.norm(np.cross(up, vp), axis=1, keepdims=True)
    nf_reshaped = nf.reshape(-1, 1, 3)

    verts_by_faces = npp[mesh.faces]

    new_u_arr = np.zeros((mesh.faces.shape[0], 3, 3))
    new_v_arr = np.zeros((mesh.faces.shape[0], 3, 3))

    for i in range(3):
        col_slice = verts_by_faces[:, i, :].reshape(-1, 1, 3)
        R_stack = calc_R_to_align_rows_of_vectors(nf_reshaped, col_slice)

        u_slice = up[mesh.faces][:, i, :].reshape(-1, 3, 1)
        v_slice = vp[mesh.faces][:, i, :].reshape(-1, 3, 1)

        new_u = R_stack @ u_slice
        new_v = R_stack @ v_slice

        new_u_arr[:, i, :] = new_u.reshape(-1, 3)
        new_v_arr[:, i, :] = new_v.reshape(-1, 3)

    return new_u_arr, new_v_arr


def setup_lstsq_A_b(mesh, vert_norms, face_norms, e0, e1, e2, e0_norm, B_norm):

    e0_dot_e0_norm = e0.reshape(-1, 1, 3) @ e0_norm.reshape(-1, 3, 1)
    e0_dot_B_norm = e0.reshape(-1, 1, 3) @ B_norm.reshape(-1, 3, 1)

    e1_dot_e0_norm = e1.reshape(-1, 1, 3) @ e0_norm.reshape(-1, 3, 1)
    e1_dot_B_norm = e1.reshape(-1, 1, 3) @ B_norm.reshape(-1, 3, 1)

    e2_dot_e0_norm = e2.reshape(-1, 1, 3) @ e0_norm.reshape(-1, 3, 1)
    e2_dot_B_norm = e2.reshape(-1, 1, 3) @ B_norm.reshape(-1, 3, 1)

    zeros_col = np.zeros(e0_dot_e0_norm.shape)

    A_stack = np.hstack(
        [
            np.dstack([e0_dot_e0_norm, e0_dot_B_norm, zeros_col]),
            np.dstack([zeros_col, e0_dot_e0_norm, e0_dot_B_norm]),
            np.dstack([e1_dot_e0_norm, e1_dot_B_norm, zeros_col]),
            np.dstack([zeros_col, e1_dot_e0_norm, e1_dot_B_norm]),
            np.dstack([e2_dot_e0_norm, e2_dot_B_norm, zeros_col]),
            np.dstack([zeros_col, e2_dot_e0_norm, e2_dot_B_norm]),
        ]
    )

    n0 = vert_norms[mesh.faces][:, 0, :]
    n1 = vert_norms[mesh.faces][:, 1, :]
    n2 = vert_norms[mesh.faces][:, 2, :]

    b_stack = np.hstack(
        [
            (n2 - n1).reshape(-1, 1, 3) @ e0_norm.reshape(-1, 3, 1),
            (n2 - n1).reshape(-1, 1, 3) @ B_norm.reshape(-1, 3, 1),
            (n0 - n2).reshape(-1, 1, 3) @ e0_norm.reshape(-1, 3, 1),
            (n0 - n2).reshape(-1, 1, 3) @ B_norm.reshape(-1, 3, 1),
            (n1 - n0).reshape(-1, 1, 3) @ e0_norm.reshape(-1, 3, 1),
            (n1 - n0).reshape(-1, 1, 3) @ B_norm.reshape(-1, 3, 1),
        ]
    )

    return A_stack, b_stack


def calc_face_II(mesh, A_stack, b_stack):
    """Use sparse block diagonal matrix to simultaneously solve the least squares problem for all vertices."""
    FaceSFM = np.zeros((len(mesh.faces), 2, 2))
    Kn = np.zeros((1, mesh.faces.shape[0]))

    row_indices = np.repeat(
        np.arange(6 * mesh.faces.shape[0]).reshape(-1, 1), 3, axis=1
    ).ravel()
    col_indices = np.tile(
        np.tile(np.arange(3), 6), mesh.faces.shape[0]
    ) + 3 * np.repeat(np.arange(mesh.faces.shape[0]), 18)
    data = A_stack.ravel()
    A = scipy.sparse.coo_matrix((data, (row_indices, col_indices)))

    x = scipy.sparse.linalg.lsqr(A, b_stack.reshape(-1, 1))[0]
    x = x.reshape(A_stack.shape[0], 3)
    FaceSFM[:, 0, 0] = x[:, 0]
    FaceSFM[:, 0, 1] = x[:, 1]
    FaceSFM[:, 1, 0] = x[:, 1]
    FaceSFM[:, 1, 1] = x[:, 2]

    # Kn[0][i] = np.dot(
    #     np.array([1, 0]), np.dot(FaceSFM[i], np.array([[1.0], [0.0]]))
    # )

    return FaceSFM


def calc_vertex_II(mesh, face_norms, vert_norms, new_u_arr, new_v_arr, wfp):
    """Calculates II (second fundamental tensor) for each vertex by weighting and projecting the II from surrounding faces

    CalcFaceCurvature recives a list of vertices and faces in mesh structure
    and the normal at each vertex and calculates the second fundemental
    matrix and the curvature using least squares

    INPUT :
    mesh - face-vertex data structure containing a list of vertices and a list of faces
    VertexNoRMALS - n*3 matrix ( n = number of vertices ) containing the normal at each vertex
    FaceNormals - m*3 matrix ( m = number of faces ) containing the normal of each face

    OUTPOUT
    FaceSFM - an m*1 cell matrix second fundemental
    VertexSFM - an n*w cell matrix second fundementel
    wfp - corner voronoi weights"""
    "Matrix of each face at each cell"

    VertexSFM = np.zeros((len(mesh.vertices), 2, 2))

    e0, e1, e2, e0_norm, e1_norm, e2_norm, de0, de1, de2 = calc_edge_vectors(mesh)
    B_norm = np.cross(face_norms, e0_norm, axis=1)

    A_stack, b_stack = setup_lstsq_A_b(
        mesh, vert_norms, face_norms, e0, e1, e2, e0_norm, B_norm
    )

    FaceSFM = calc_face_II(mesh, A_stack, b_stack)

    Lv_arr, Mv_arr, Nv_arr = project_curvature_tensor(
        mesh,
        FaceSFM,
        e0_norm,
        np.cross(e0_norm, face_norms, axis=1),
        new_u_arr,
        new_v_arr,
    )

    sparseMatrix_Lv, f_indices, col = construct_vertex_x_face_sparse_matrix(mesh)
    sparseMatrix_Mv = sparseMatrix_Lv.copy()
    sparseMatrix_Nv = sparseMatrix_Lv.copy()
    sparseMatrix_wfp = sparseMatrix_Lv.copy()

    # Assign data
    sparseMatrix_Lv.data = Lv_arr[f_indices, col].ravel()
    sparseMatrix_Mv.data = Mv_arr[f_indices, col].ravel()
    sparseMatrix_Nv.data = Nv_arr[f_indices, col].ravel()
    sparseMatrix_wfp.data = wfp[f_indices, col].ravel()

    Lv = sparseMatrix_Lv.multiply(sparseMatrix_wfp).sum(axis=1).A.ravel()
    Mv = sparseMatrix_Mv.multiply(sparseMatrix_wfp).sum(axis=1).A.ravel()
    Nv = sparseMatrix_Nv.multiply(sparseMatrix_wfp).sum(axis=1).A.ravel()

    VertexSFM[:, 0, 0] = Lv
    VertexSFM[:, 0, 1] = Mv
    VertexSFM[:, 1, 0] = Mv
    VertexSFM[:, 1, 1] = Nv

    return VertexSFM


def calc_k1_k2(II):
    """Calculates the principal curvatures (eigenvalues) and principal directions (eigenvectors) of II."""
    w, v = np.linalg.eig(II)
    k1, k2 = w.T
    k1_vec = v[:, :, 0]
    k2_vec = v[:, :, 1]
    return k1, k2, k1_vec, k2_vec


def calc_principal_curvatures(mesh):

    # Faces
    face_norms = calc_face_normals(mesh)
    face_areas = calc_triangle_area(
        mesh.vertices[mesh.faces[:, 0]],
        mesh.vertices[mesh.faces[:, 1]],
        mesh.vertices[mesh.faces[:, 2]],
    )
    face_voronoi_areas = calc_face_voronoi_areas(mesh)
    adjusted_voronoi_areas = adjust_obtuse_face_voronoi_areas(
        mesh, face_areas, face_voronoi_areas
    )
    wfp = calc_face_weights(mesh, adjusted_voronoi_areas)

    # Vertices
    vert_norms = calc_vertex_normals(mesh)
    up, vp = initialize_vertex_coordinate_system(mesh, vert_norms)

    new_u_arr, new_v_arr = align_coordinate_systems(mesh, up, vp, face_norms)

    # Second fundamental tensor (II)
    II = calc_vertex_II(mesh, face_norms, vert_norms, new_u_arr, new_v_arr, wfp)

    # Principal curvatures and directions
    k1, k2, k1_vec, k2_vec = calc_k1_k2(II)

    # Rotate principal direction vector to be in R3
    curr = np.hstack(
        [up.reshape(-1, 1, 3), vp.reshape(-1, 1, 3), vert_norms.reshape(-1, 1, 3)]
    )
    goal = np.tile(np.eye(3), (mesh.vertices.shape[0], 1, 1))
    R = goal @ np.linalg.inv(curr)
    i = 0
    k1_vec_e = np.hstack((k1_vec, np.zeros((mesh.vertices.shape[0], 1)))).reshape(
        -1, 3, 1
    )
    k2_vec_e = np.hstack((k2_vec, np.zeros((mesh.vertices.shape[0], 1)))).reshape(
        -1, 3, 1
    )

    k1_R3 = R[i] @ k1_vec_e[i]

    v = mesh.vertices.reshape(-1, 1, 3)
    vN = vert_norms.reshape(-1, 1, 3)
    vK1 = (R @ k1_vec_e).reshape(-1, 1, 3)
    vK2 = (R @ k2_vec_e).reshape(-1, 1, 3)
    up_r = up.reshape(-1, 1, 3)
    vp_r = vp.reshape(-1, 1, 3)

    scale = 0.1

    N_segs = np.hstack((v, (v + scale * vN).reshape(-1, 1, 3)))
    N_path = trimesh.load_path(N_segs)
    N_path.colors = np.tile(np.array([[255, 0, 0, 255]]), (mesh.vertices.shape[0], 1))

    # K1_segs = np.hstack((v - scale * vK1, v + scale * vK1))
    # K1_path = trimesh.load_path(K1_segs)
    # K1_path.colors = np.tile(np.array([[0, 255, 0, 255]]), (mesh.vertices.shape[0], 1))

    # K2_segs = np.hstack((v - scale * vK2, v + scale * vK2))
    # K2_path = trimesh.load_path(K2_segs)
    # K2_path.colors = np.tile(np.array([[0, 0, 255, 255]]), (mesh.vertices.shape[0], 1))

    u_segs = np.hstack((v - scale * up_r, v + scale * up_r))
    u_path = trimesh.load_path(u_segs)
    u_path.colors = np.tile(np.array([[0, 255, 0, 255]]), (mesh.vertices.shape[0], 1))

    v_segs = np.hstack((v - scale * vp_r, v + scale * vp_r))
    v_path = trimesh.load_path(v_segs)
    v_path.colors = np.tile(np.array([[0, 0, 255, 255]]), (mesh.vertices.shape[0], 1))

    scene = trimesh.Scene()
    # scene.add_geometry([mesh, K1_path, K2_path])
    scene.add_geometry([mesh, N_path, u_path, v_path])

    # scene.add_geometry([mesh])

    scene.show(smooth=False)

    # In [1]: import numpy as np

    # In [2]: import trimesh

    # In [3]: segments = np.random.random((100,2,3))

    # In [4]: p = trimesh.load_path(segments)

    # In [5]: p
    # Out[5]: <trimesh.path.path.Path3D at 0x7fbaa49eb240>

    # In [6]: p.show()

    return k1, k2, k1_vec, k2_vec


if __name__ == "__main__":

    mesh = trimesh.creation.icosphere(2)
    mesh = mesh.apply_scale([1, 1, 3])
    # mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    k1, k2, k1_vec, k2_vec = calc_principal_curvatures(mesh)
    print(k1)
    print(k1_vec)
