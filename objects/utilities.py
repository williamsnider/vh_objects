# General functions used by different classes for objects project

from re import A
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from sympy import Q
from objects.parameters import ORDER, HARMONIC_POWER
from splipy import BSplineBasis, Curve
import igl
import trimesh
import scipy
from scipy.spatial.transform import Rotation
from compas_cgal.booleans import boolean_union, boolean_difference

##########
# B-Spline Functions


def open_uniform_knot_vector(num_cps, order):

    num_knots = num_cps + order
    knots = np.zeros(num_knots)
    knots[order:-order] = range(1, num_knots - 2 * order + 1)
    knots[-order:] = knots[-order - 1] + 1
    knots = knots / knots.max()  # Scale from 0 to 1
    return knots


def approximate_arc(MAX_ANGLE, arc_length):
    """Construct a B-Spline curve that approximates a circular arc."""

    # Handle MAX_ANGLE = 0  (straight line)
    if MAX_ANGLE == 0:
        NUM_CP = 5
        cp_array = np.array(
            [
                np.linspace(0, arc_length, NUM_CP),
                np.zeros(NUM_CP),
                np.zeros(NUM_CP),
            ]
        ).T
        return cp_array

    radius = arc_length / (2 * np.pi) * (2 * np.pi / MAX_ANGLE)

    # if NUM_CP_PER_SEGMENT != 5:
    #     raise NotImplementedError("NUM_CP_PER_SEGMENT must be 5 for this funtion to work.")

    def make_arc_array(a, b, c):

        # We can think of the second to last arc controlpoint as lying along a vector from the last controlpoint. The vector's slope can be determined from the tangent line of the circle (which is negated). We then can use a single parameter (d) as a measure of how far along this vector we are travelling. This reduces the number of parameters we need, and also ensures that the tangent of the resulting arc at the end will match that of the circle

        def tan_vec(MAX_ANGLE):
            tangent_vec = np.array(
                [
                    -radius * np.sin(MAX_ANGLE),
                    radius * np.cos(MAX_ANGLE),
                ]
            )
            return tangent_vec

        start_tan_vec = tan_vec(0)
        end_tan_vec = -tan_vec(MAX_ANGLE)  # Negate so this points toward start

        arc_array = np.array(
            [
                [radius, 0, 0],
                [
                    radius * np.cos(0) + a * start_tan_vec[0],
                    radius * np.sin(0) + a * start_tan_vec[1],
                    0,
                ],  # Tangent line from start
                [b, c, 0],
                [
                    radius * np.cos(MAX_ANGLE) + a * end_tan_vec[0],
                    radius * np.sin(MAX_ANGLE) + a * end_tan_vec[1],
                    0,
                ],  # Tangent line from end
                [radius * np.cos(MAX_ANGLE), radius * np.sin(MAX_ANGLE), 0],
            ]
        )
        return arc_array

    def radius_error(vars):

        # Make the arc array
        [a, b, c] = vars
        arc_array = make_arc_array(a, b, c)

        # Make curve
        knot = open_uniform_knot_vector(arc_array.shape[0], ORDER)
        basis = BSplineBasis(order=ORDER, knots=knot, periodic=-1)
        curve = Curve(basis=basis, controlpoints=arc_array, rational=False)

        # Sample points along the backbone
        t = np.linspace(0, 1, 10)
        r = curve(t)

        # distance from origin should be close to radius if points are well_aligned
        dist = np.linalg.norm(r, axis=1)
        return ((dist - radius) ** 2).sum()

    fun = radius_error
    x0 = [0.1, radius, radius]
    bounds = [
        [0.0, 100 * radius],
        [radius * np.cos(MAX_ANGLE / 2), 100 * radius],  # Convex hull property of B-Splines
        [radius * np.sin(MAX_ANGLE / 2), 100 * radius],  # Convex hull property of B-Splines
    ]
    result = minimize(fun=fun, x0=x0, bounds=bounds)
    [a, b, c] = result.x

    arc_array = make_arc_array(a, b, c)

    # Shift so that the curve begins at the origin
    arc_array[:, 0] -= radius
    arc_array[:, [0, 1]] = arc_array[:, [1, 0]]  # Flip x and y-axis so long portion points in +X direction
    arc_array[:, 1] = -arc_array[:, 1]  # Negate y axis so curves upward (towards +Y)

    return arc_array


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


def calc_R_euler_angles(euler_angles):
    s = np.sin
    c = np.cos

    a1, a2, a3 = euler_angles
    R_x = np.array(
        [
            [1, 0, 0],
            [0, c(a1), -s(a1)],
            [0, s(a1), c(a1)],
        ]
    )

    R_y = np.array(
        [
            [c(a2), 0, s(a2)],
            [0, 1, 0],
            [-s(a2), 0, c(a2)],
        ]
    )

    R_z = np.array(
        [
            [c(a3), -s(a3), 0],
            [s(a3), c(a3), 0],
            [0, 0, 1],
        ]
    )

    R_euler_angles = R_z @ (R_y @ R_x)
    return R_euler_angles


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


def check_and_move_identical_verts(mesh1, mesh2):
    """Mesh boolean fails often when two vertices on two meshes are identical, so we need to shift one vertex very slightly."""

    # Calculate distance between two sets of vertices
    tree = scipy.spatial.KDTree(mesh1.vertices)
    dd, ii = tree.query(mesh2.vertices, k=1)
    identical_verts = np.isclose(dd, 0, atol=0.01)

    # Shift these verts
    mesh2.vertices[identical_verts] += 1e-2  #

    # Check that this shift was successful
    dd, ii = tree.query(mesh2.vertices, k=1)
    identical_verts = np.isclose(dd, 0)
    assert np.any(identical_verts) == False, "Shifting the identical vertex did not work."

    return mesh1, mesh2


def check_and_move_verts_on_edges(mesh1, mesh2):
    """Mesh boolean fails often when a vertex on mesh1 lies on an edge of mesh2, so we need to shift such vertices slightly.

    Currently this function only identifies colinear vertices and edges and does not check if the vertex is actually between the two points on the edge. That could be implemented by checking if the distance from the vertex to both edge points is less than the distance between the two edge points."""

    def find_colinear_vertices(mesh1, mesh2):

        # Cross product reveals whether 3 poitns are
        edges = mesh1.vertices[mesh1.edges]
        edges_vec = edges[:, 1, :] - edges[:, 0, :]  # Vector between two points defining edge
        verts_vec = (
            mesh2.vertices.reshape(-1, 1, 3) - edges[:, 0, :]
        )  # Vector between putative point in between edge and on point defining edge
        cross = np.cross(edges_vec, verts_vec)
        norm = np.linalg.norm(cross, axis=2)
        EPSILON = 0.01
        colinear_verts = np.any(norm < EPSILON, axis=1)
        return colinear_verts
        # # Cross product reveals whether 3 points are colinear
        # edges = mesh1.vertices[mesh1.edges]
        # edges_vec = edges[:, 1, :] - edges[:, 0, :]  # Vector between two points defining edge
        # verts_vec = (
        #     mesh2.vertices.reshape(-1, 1, 3) - edges[:, 0, :]
        # )  # Vector between putative point in between edge and on point defining edge
        # epsilon = 0.001
        # angles = angle_between(verts_vec, edges_vec)
        # angles_near_0 = np.any(angles < epsilon, axis=1)
        # angles_near_2pi = np.any(angles > 2 * np.pi - epsilon, axis=1)
        # colinear_verts = np.logical_or(angles_near_0, angles_near_2pi)
        # return colinear_verts

    none_colinear = False
    count = 0
    while count < 10:

        colinear_verts = find_colinear_vertices(mesh1, mesh2)
        mesh2.vertices[colinear_verts] += 1e-2  # Shift these verts

        # Confirm shift worked
        colinear_verts = find_colinear_vertices(mesh1, mesh2)
        none_colinear = np.all(~colinear_verts)

        # Except loop if successful
        if none_colinear == True:
            print("Shifting colinear vertices worked after {count} loops.".format(count=count))
            break
    else:
        print("Shifting colinear vertices did not work after {count} loops.".format(count=count))

    return mesh1, mesh2


def move_verts_on_broken_faces(union_mesh, mesh1, mesh2):
    """Shifts vertices that are part of the broken faces of a mesh.

    Doing this retroactively (as opposed to shifting all vertices that lie on an edge of the other mesh) is a lot faster."""

    # Identify vertices of broken faces (indexed by union_mesh)
    broken = trimesh.repair.broken_faces(union_mesh)
    vert_indices = union_mesh.faces[broken].ravel()
    verts_to_shift = union_mesh.vertices[vert_indices]

    # Find same vertices on mesh1 and mesh2
    tree = scipy.spatial.KDTree(verts_to_shift)
    for mesh in [mesh1, mesh2]:
        dd, ii = tree.query(mesh.vertices, k=1)
        mesh_verts_to_shift = np.isclose(dd, 0, atol=0.01)

        mesh.vertices[mesh_verts_to_shift] += 1e-2

    return mesh1, mesh2


def calc_mesh_boolean_and_edges(mesh1, mesh2, operation):

    # Use compas/CGAL to calculate boolean operation
    mesh_A = [mesh1.vertices.tolist(), mesh1.faces.tolist()]
    mesh_B = [mesh2.vertices.tolist(), mesh2.faces.tolist()]
    if operation == "union":
        mesh_C = boolean_union(mesh_A, mesh_B)
    elif operation == "difference":
        mesh_C = boolean_difference(mesh_A, mesh_B)
    else:
        raise NotImplementedError("Boolean operation must be 'union' or 'difference'.")

    # Get edges - vertices that were in neither initial mesh
    set_A = set([tuple(l) for l in mesh_A[0]])
    set_B = set([tuple(l) for l in mesh_B[0]])
    set_C = set([tuple(l) for l in mesh_C[0]])
    new_verts = (set_C - set_A) - set_B
    edge_verts_pts = np.zeros((len(new_verts), 3))
    for i, v in enumerate(new_verts):
        edge_verts_pts[i] = list(v)

    # Return as trimesh - easier to work with
    mesh = trimesh.Trimesh(
        vertices=mesh_C[0],
        faces=mesh_C[1],
    )

    # Get indices of edge_verts
    tree = scipy.spatial.KDTree(mesh.vertices)
    _, edge_verts_indices = tree.query(edge_verts_pts, k=1)

    # XXX: Debug
    # trimesh.repair.broken_faces(mesh, color=[255, 0, 0, 255])
    # mesh.show(smooth=False)
    # pass

    # if mesh.is_watertight:
    #     print("Mesh is watertight")
    # else:
    #     print("Mesh is NOT watertight")
    return mesh, edge_verts_indices


def find_neighbors(mesh, group, distance):

    mesh_pts = mesh.vertices.__array__()
    edge_pts = mesh.vertices[group].__array__()

    tree = scipy.spatial.KDTree(mesh_pts)
    neighbors_list = tree.query_ball_point(edge_pts, r=distance)
    neighbors = set()
    for n in neighbors_list:
        neighbors.update(set(n))

    return list(neighbors)


def fair_mesh(union_mesh, neighbors, harmonic_power):

    v = union_mesh.vertices.__array__()
    f = union_mesh.faces.__array__().astype("int64")
    num_verts = v.shape[0]
    b = np.array(list(set(range(num_verts)) - set(neighbors))).astype("int64")  # Bounday indices - NOT to be faired
    bc = v[b]  # XYZ coordinates of the boundary indices
    z = igl.harmonic_weights(v, f, b, bc, harmonic_power)  # Smooths indices at creases

    union_mesh.vertices = z
    faired_mesh = union_mesh

    return faired_mesh


def fuse_meshes(meshA, meshB, fairing_distance, operation):
    """Fuses two meshes and smoothly fairs their intersection."""

    # Copy meshes to avoid altering
    mesh1 = meshA.copy()
    mesh2 = meshB.copy()

    # Check that meshes are watertight
    for mesh in [mesh1, mesh2]:
        assert mesh.is_watertight, "Input meshes must be watertight."

    count = 0
    while count < 5:

        # Compute boolean
        union_mesh, edge_verts_indices = calc_mesh_boolean_and_edges(mesh1, mesh2, operation)

        # Check watertightness; shift vertices slightly if not and repeat loop
        if union_mesh.is_watertight == False:
            mesh1, mesh2 = move_verts_on_broken_faces(union_mesh, mesh1, mesh2)
        else:
            break

        count += 1
    else:
        print("Mesh boolean failed to form a watertight mesh after {count} loops.".format(count=count))
    if fairing_distance > 0:
        neighbors = find_neighbors(union_mesh, edge_verts_indices, distance=fairing_distance)
        faired_mesh = fair_mesh(union_mesh, neighbors, HARMONIC_POWER)
        return faired_mesh
    else:
        return union_mesh
    # if union_mesh.is_watertight is False:
    #     print("Mesh will not be faired, as it is not watertight.")
    #     return union_mesh
    # else:
    #     faired_mesh = fair_mesh(union_mesh, neighbors, HARMONIC_POWER)
    #     return faired_mesh


def find_closest_surface_point(backbone_point, N, surface_points):

    # Find surface points lying in direction of vector (i.e. correct side of object)
    backbone_to_vertices = surface_points - backbone_point
    dp = np.squeeze(np.dot(backbone_to_vertices, N.T))
    P = surface_points[dp > 0]

    # Find closest surface point
    A = backbone_point
    B = backbone_point + N  # second point on line
    AB = B - A  # equivalent to N
    PA = A - P
    PB = B - P
    dist = np.linalg.norm(np.cross(PA, PB), axis=1) / np.linalg.norm(AB)
    closest_surface_point = P[dist.argmin()]

    # # XXX: Debug
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # ax.view_init(elev=-90, azim=90)

    # # Plot backbone point
    # x, y, z = backbone_point.T
    # ax.plot3D(x, y, z, ".k", markersize=20)

    # # Plot surface
    # x, y, z = surface_points[::17].T
    # ax.plot3D(x, y, z, "b.")

    # # Plot closest surface point
    # x, y, z = closest_surface_point.T
    # ax.plot3D(np.array([x]), np.array([y]), np.array([z]), "b.", markersize=20)

    # # Plot N
    # p0 = backbone_point
    # p1 = p0 + N * 10
    # x, y, z = np.vstack([p0, p1]).T
    # ax.plot3D(x, y, z, "g-")

    # plt.show()
    return closest_surface_point


def transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment):
    """Transform the surface deformation mesh so that it is centered on the parent axial component's surface and has the desired orientation."""

    sd_mesh = sd_mesh.copy()

    # Find point on backbone, surface, and basis of
    backbone_point = ac.backbone.r(pos)
    surface_points = ac.mesh.vertices

    # Get tangent and normal vectors of backbone basis of parent shape
    T = ac.backbone.T(pos)[0]
    N = ac.backbone.N(pos)[0]

    # Rotate normal vector based on theta_backbone (rotation about backbone)
    R_about_T = Rotation.from_rotvec(theta_backbone * T).as_matrix()
    N_rotated = N @ R_about_T

    # Goal Basis
    goal_T = N_rotated
    goal_N = T
    goal_B = np.cross(T, N_rotated)
    goal_TNB = np.array([goal_T, goal_N, goal_B])

    # Rotate goal_TNB about T  (rotation about vector through backbone and surface point)
    R_about_B = Rotation.from_rotvec(theta_linear_segment * goal_T).as_matrix()
    surface_point = find_closest_surface_point(backbone_point, N_rotated, surface_points)
    T = np.eye(4)
    T[:3, :3] = np.linalg.inv(goal_TNB @ R_about_B)
    T[:3, 3] = surface_point

    # T = ac.backbone.T(pos)[0]
    # N = ac.backbone.N(pos)[0]
    # B = ac.backbone.B(pos)[0]
    # R_upright = Rotation.from_rotvec(
    #     np.pi / 2 * B
    # ).as_matrix()  # 90deg rotation so that sd_mesh is orthogonal to backbone
    # R_backbone = Rotation.from_rotvec(-theta_backbone * T).as_matrix()  # Rotation about backbone's tangent vector
    # R_linear_segment = calc_R_euler_angles(
    #     [theta_linear_segment, 0, 0]
    # )  # Rotation about vector through surface point and backbone
    # N_rotated = N @ R_backbone
    # surface_point = find_closest_surface_point(backbone_point, N_rotated, surface_points)

    # # Align sd with parent
    # surface_point = find_closest_surface_point(backbone_point, N_rotated, surface_points)
    # T = np.eye(4)
    # T[:3, :3] = R_upright @ R_backbone @ np.linalg.inv(parent_basis)  # @ R_upright  # @ R_linear_segment
    # T[:3, 3] = surface_point

    # Apply transformations
    sd_mesh.apply_translation(-origin)  # Move centroid to origin
    sd_mesh.apply_transform(T)
    # # Calculate rotation matrices
    # T = ac.backbone.T(pos)[0]
    # N = ac.backbone.N(pos)[0]
    # R_backbone = Rotation.from_rotvec(-theta_backbone * T).as_matrix()  # Rotation about backbone's tangent vector
    # R_linear_segment = calc_R_euler_angles(
    #     [theta_linear_segment, 0, 0]
    # )  # Rotation about vector through surface point and backbone
    # N_rotated = N @ R_backbone
    # surface_point = find_closest_surface_point(backbone_point, N_rotated, surface_points)
    # T = np.eye(4)
    # T[:3, :3] = parent_basis @ R_backbone @ R_linear_segment
    # T[:3, 3] = surface_point

    # # Apply transformations
    # sd_mesh.apply_translation(-origin)  # Move centroid to origin
    # sd_mesh.apply_transform(T)

    return sd_mesh


def calc_mesh_principal_curvatures(mesh):
    """Calculates the principal curvatures (k1, k2) of a triangular mesh."""

    # TODO: COMPAS implementation probably faster https://compas.dev/compas/latest/api/generated/compas_rhino.geometry.trimesh.trimesh_principal_curvature.html
    RADIUS = 1
    K = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, RADIUS)
    H = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, RADIUS)

    # Handle nan's by replacing with 0 (k1 and k2 then both equal guassian curvature H)
    sqrt = np.sqrt(H**2 - K)
    sqrt[np.isnan(sqrt)] = 0

    k1 = H + sqrt
    k2 = H - sqrt
    return k1, k2


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


def plot_mesh_and_specific_indices(
    mesh,
    specific_indices,
    spacing=1,
):

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("blue = specific indices")
    ax.view_init(elev=-90, azim=90)

    # Entire mesh
    x, y, z = mesh.vertices[::spacing].T
    ax.plot(x, y, z, ".", color="green")

    # Vertices in the group
    x, y, z = mesh.vertices[specific_indices].T
    ax.plot(x, y, z, ".", color="blue")

    plt.show()


def plot_controlpoints(ac):

    # Controlpoints
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
    ax_min = cp.min()
    ax_max = cp.max()
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.set_zlim([ax_min, ax_max])
    ax.view_init(elev=-90, azim=90)
    ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "k.")
    ax.plot3D(x, y, z, "g-")
    plt.show()
