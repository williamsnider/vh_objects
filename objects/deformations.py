from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
import numpy as np
import copy
from scipy.spatial import cKDTree


def deform_ac(ac, base_ac, row, column, deformation_filter):
    """Deform the specified axial component by moving the controlpoint at row, column by the given filter."""

    if type(deformation_filter) == type(None):
        return ac

    # Organize inputs
    cp_r = row
    cp_c = column
    neighborhood = np.array(deformation_filter.shape)  # 3x3 neighborhood
    assert all(neighborhood % 2 == [1, 1]), "Neighborhood must be odd."

    # Gather the indices of the neighboring controlpoints
    filter_num_rows = neighborhood[0]
    filter_num_cols = neighborhood[1]
    shift_matrix = np.empty([filter_num_rows, filter_num_cols, 2], dtype="int")
    for i in range(filter_num_rows):
        for j in range(filter_num_cols):
            shift_matrix[i, j, :] = [i - (filter_num_rows - 1) // 2, j - (filter_num_cols - 1) // 2]

    # Handle column values that need to be wrapped around
    filter_indices = shift_matrix + np.array([cp_r, cp_c])
    filter_indices[:, :, 1] = filter_indices[:, :, 1] % base_ac.controlpoints.shape[1]

    pts = ac.controlpoints[filter_indices[:, :, 0], filter_indices[:, :, 1], :]
    pts_flat = pts.reshape(-1, 3)

    # Find plane of points
    def fit_plane(points):

        # Barycenter of the points
        G = points.sum(axis=0) / points.shape[0]

        # SVD
        u, s, vh = np.linalg.svd(points - G)

        # Unitary normal vector
        u_norm = vh[2, :]

        return u_norm, G

    u_norm, G = fit_plane(pts_flat)

    # Check if u_norm points away from the backbone and flip if not
    u_norm_correct_direction = False
    NUM_EXTRA_CROSS_SECTIONS = 6  # Ends of axial component have 3 CSs to help with slope at endpoint
    backbone_point = base_ac.r(row / (base_ac.controlpoints.shape[0] + NUM_EXTRA_CROSS_SECTIONS))
    while u_norm_correct_direction == False:
        a, b, c = u_norm
        d = -np.dot(u_norm, G)

        # Find a point on plane by assuming two of x,y,z = 0. Handle cases where a, b, or c==0.
        if ~np.isclose(c, 0):
            plane_point = np.array([0, 0, -d / c])
        elif ~np.isclose(b, 0):
            plane_point = np.array([0, -d / b, 0])
        elif ~np.isclose(a, 0):
            plane_point = np.array([-d / a, 0, 0])

        vec_backbone_to_plane = plane_point - backbone_point[0]

        if np.dot(u_norm, vec_backbone_to_plane) > 0:
            u_norm_correct_direction = True
        elif np.dot(u_norm, vec_backbone_to_plane) < 0:
            u_norm = -u_norm
        else:
            raise ValueError("u_norm and vec_backbone_to_plane are orthogonal.")

    plane_normal = u_norm

    # Align controlpoints to this plane
    def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
        ndotu = planeNormal.dot(rayDirection)
        if abs(ndotu) < epsilon:
            raise RuntimeError("no intersection or line is within plane")

        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi

    pts_new = np.zeros(pts_flat.shape)

    # Find closest point on surface of mesh
    tree = cKDTree(base_ac.verts)
    dd, ii = tree.query(ac.controlpoints[cp_r, cp_c])
    surface_point = base_ac.verts[ii]

    for i, (row, col) in enumerate(filter_indices.reshape(-1, 2)):

        # Get pos of cross section
        CS_idx = (
            row - 3 - 1
        )  # 3 "helper" cross sections at beginning of axial component; row here (not col) because contrrolpoints are organized as (num_cross_sections+6, num_cp_per_cross_section, 3)
        pos = ac.cross_sections[CS_idx].position
        cp = ac.controlpoints[row, col]
        orthogonal_vec = u_norm
        # vec_from_backbone_to_cp = cp - ac.r(pos)[0]

        pts_new[i] = LinePlaneCollision(plane_normal, surface_point, orthogonal_vec, cp)

    pts_on_plane = pts_new

    # TODO: Shift points so that they form a circle on this plane (diagonal filters will work better)
    # Scale changes based on avg distance from neighbors to center
    num_neighbors = neighborhood.prod()
    middle_cp = (num_neighbors + 1) // 2
    neighbor_indices = [i for i in range(num_neighbors) if i != middle_cp]
    avg_dist = np.linalg.norm(pts_flat[neighbor_indices] - pts_flat[middle_cp], axis=1).mean(axis=0)

    FILTER_SCALE = 0.5

    shifted_pts = pts_on_plane + deformation_filter.reshape(-1, 1, order="F") * plane_normal * avg_dist * FILTER_SCALE
    ac.controlpoints[filter_indices[:, :, 0], filter_indices[:, :, 1], :] = shifted_pts.reshape(
        filter_num_rows, filter_num_cols, 3
    )

    return ac


# Plane
plane = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
)

# Concave ellipsoid
concave_ellipsoid = np.array(
    [
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0],
    ]
)

# Concave cylinder
concave_cylinder_vert = np.array(
    [
        [0, -1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ]
)

concave_cylinder_diag_down = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]
)
concave_cylinder_diag_up = np.array(
    [
        [0, 0, -1],
        [0, -1, 0],
        [-1, 0, 0],
    ]
)

concave_cylinder_hori = np.array(
    [
        [0, 0, 0],
        [-1, -1, -1],
        [0, 0, 0],
    ]
)

# Hyperboloid


hyperboloid_surface_vert = np.array(
    [
        [0, -1, 0],
        [1, 0, 1],
        [0, -1, 0],
    ]
)

hyperboloid_surface_diag_down = np.array(
    [
        [-1, 0, 1],
        [0, 0, 0],
        [1, 0, -1],
    ]
)

hyperboloid_surface_diag_up = np.array(
    [
        [1, 0, -1],
        [0, 0, 0],
        [-1, 0, 1],
    ]
)

hyperboloid_surface_hori = np.array(
    [
        [0, 1, 0],
        [-1, 0, -1],
        [0, 1, 0],
    ]
)

# Convex cylinder
convex_cylinder_vert = np.array(
    [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
)

convex_cylinder_diag_down = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)

convex_cylinder_diag_up = np.array(
    [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]
)

convex_cylinder_hori = np.array(
    [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ]
)


# Convex ellipsoid
convex_ellipsoid = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
)
