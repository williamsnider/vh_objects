# General functions used by different classes for objects project

from re import A
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize, minimize_scalar
from sympy import Q
from objects.parameters import ORDER, HARMONIC_POWER
from splipy import BSplineBasis, Curve, Surface
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


def approximate_arc(MAX_ANGLE, arc_length, num_cp):
    """Construct a B-Spline curve that approximates a circular arc."""

    assert num_cp % 2 == 1, "num_cp must be odd."
    assert num_cp >= 5, "num_cp must be 5 or greater and odd"

    num_params = (num_cp - 1) // 2

    # Handle MAX_ANGLE = 0  (straight line)
    if MAX_ANGLE == 0:
        cp_array = np.array(
            [
                np.linspace(0, arc_length, num_cp),
                np.zeros(num_cp),
                np.zeros(num_cp),
            ]
        ).T
        return cp_array

    radius = arc_length / (2 * np.pi) * (2 * np.pi / MAX_ANGLE)

    def make_arc_array(params):
        """Constructs an array of controlpoints that approximates a circular arc.

        params are variables that get optimized.

        param[0]: distance from cp[0] to cp[1]
        param[n]: distance from (0,0) to cp[n+1]"""

        # Check that num_cp and num_params are compatible
        assert 2 * len(params) + 1 == num_cp

        # Slope of arc at initial point
        start_tan_vec = np.array([0, 1])

        ### Assemble controlpoint array ###
        # We will do this by calculating the first half, then reflecting this the bisecting vector, which results in a symmetric controlpoint array (and minimizes number of parameters to optimize)

        # First point - spline starts here
        cp = [[radius, 0]]

        for i in range(num_params):

            # Second point - determines angle of spline at first point
            if i == 0:
                cp.append(
                    [
                        radius + params[i] * start_tan_vec[0],
                        0 + params[i] * start_tan_vec[1],
                    ]
                )

            # Remaining points
            else:

                # Constrain point to lie along vector defined by this theta, which distributes the controlpoints more or less evenly across
                th = MAX_ANGLE / 2 * (i + 1) / num_params
                cp.append(
                    [
                        params[i] * np.cos(th),
                        params[i] * np.sin(th),
                    ]
                )
        cp = np.array(cp)  # Convert to numpy

        # Flip about bisect_vec axis
        R_reflect = np.array(
            [
                [np.cos(MAX_ANGLE), np.sin(MAX_ANGLE)],
                [np.sin(MAX_ANGLE), -np.cos(MAX_ANGLE)],
            ]
        )

        flip = cp @ R_reflect
        arc_array = np.vstack([cp, flip[-2::-1]])

        # Add z-axis of zeros
        arc_array = np.hstack([arc_array, np.zeros((arc_array.shape[0], 1))])

        return arc_array

    def radius_error(params):

        # Make the arc array
        arc_array = make_arc_array(params)

        # Make curve
        knot = open_uniform_knot_vector(arc_array.shape[0], ORDER)
        basis = BSplineBasis(order=ORDER, knots=knot, periodic=-1)
        curve = Curve(basis=basis, controlpoints=arc_array, rational=False)

        # Sample points along the backbone
        t = np.linspace(0, 1, 3 * arc_array.shape[0])
        r = curve(t)

        # distance from origin should be close to radius if points are well_aligned
        dist = np.linalg.norm(r, axis=1)
        return ((dist - radius) ** 2).sum()

    fun = radius_error
    x0 = [0.1] + [radius for i in range(num_params - 1)]
    bounds = [[0.0, 2 * radius]] + [[0, 2 * radius] for i in range(num_params - 1)]
    result = minimize(fun=fun, x0=x0, bounds=bounds)

    arc_array = make_arc_array(result.x)

    # Shift so that the curve begins at the origin
    arc_array[:, 0] -= radius
    arc_array[:, [0, 1]] = arc_array[:, [1, 0]]  # Flip x and y-axis so long portion points in +X direction
    # arc_array[:, 1] = -arc_array[:, 1]  # Negate y axis so curves upward (towards +Y)

    return arc_array


# # Calculate arc
# radius = 1
# out1 = approximate_arc(np.pi / 2, radius * np.pi / 2)
# # out1 = make_arc(1)
# vec_frac = out1[:, 0].reshape(-1, 1, 1)


def calc_hemisphere_controlpoints(base_cp, tan_vec, endpoint, poly, x, morph_to_ellipse):
    """Calculates the controlpoints needed to approximate a hemispherical ending to an axial component.

    This works by calculating a 5-controlpoint arc that will connect a quadratic curve (poly), resulting in a hemisphere shape. This arc serves as the scale (first column) and translation (second column) that are applied to base_cp."""

    ### Calculate the controlpoint arc that approximates a hemisphere ###
    num_cp_to_use = 11

    # Solve for a and r (of the fitting circle) given m and quadratic polynomial
    y = np.polyval(poly, x)
    der = np.polyder(poly, 1)
    m = np.polyval(der, x)
    a = x + y * m  # Solve for a
    r = np.sqrt((x - a) ** 2 + y**2)  # Solve for r

    # Calculate the theta and arc length of the arc
    if x <= 0:
        th = np.arctan2(y, (x - a))
        arc_length = np.pi - th
        x_shift = x
    else:
        th = np.arctan2(y, (x - a))
        arc_length = th
        x_shift = x

    # Calculate and transform the controlpoints based on which side of the axial component we are
    cp = approximate_arc(arc_length, r * arc_length, num_cp_to_use)
    cp_rot = cp[:, [1, 0]]  # Rotate 45deg
    if x <= 0:
        cp_rot[:, 0] *= -1
    cp_T = cp_rot - np.array([cp_rot[-1, 0] - x_shift, 0])  # Shift to align with end of quadratic

    scale_ratio = cp_T[:, 1] / y  # Normalize this column by the height of the quadratic

    # # Plot circle
    # import matplotlib.pyplot as plt

    # ax = plt.figure().add_subplot()
    # tt = np.linspace(0, 2 * np.pi)
    # xx = r * np.cos(tt) + a
    # yy = r*np.sin(tt)
    # ax.plot(xx,yy,'-k')
    # ax.plot(x,y,'r*')
    # ax.plot(cp_T[:,0], cp_T[:,1], "-g")
    # ax.set_aspect('equal')
    # plt.show()

    ### Apply these transformations to base_cp ###

    # Rotate base_cp to be in yz plane
    vecA = base_cp[0] - base_cp[1]
    vecB = base_cp[0] - base_cp[2]
    N = np.cross(vecA, vecB) / np.linalg.norm(np.cross(vecA, vecB))
    assert np.all(np.isclose(N, tan_vec)) or np.all(np.isclose(N, -tan_vec))
    T = np.cross(vecA, N) / np.linalg.norm(np.cross(vecA, N))
    B = np.cross(T, N)
    curr = np.vstack([N.reshape(1, -1), T.reshape(1, -1), B.reshape(1, -1)])
    goal = np.eye(3)
    R = (goal @ np.linalg.inv(curr.T)).T
    yz_cp = (base_cp - endpoint) @ R

    # Scale base_cp
    if morph_to_ellipse == True:
        cs = yz_cp[:, 1:]
        assert cs.shape[0] == 8
        a = -np.linalg.norm(cs[0])
        b = np.linalg.norm(cs[2])

        # Points reordered on ellipse
        pt = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        px = a * np.cos(pt)
        py = b * np.sin(pt)

        # Points in transition
        curr = cs
        goal = np.hstack([px.reshape(-1, 1), py.reshape(-1, 1)])
        vec = goal - curr
        vt = np.linspace(1, 0, num_cp_to_use)
        out_cp = np.zeros((num_cp_to_use, *yz_cp.shape))
        for i, t in enumerate(vt):
            pts = curr + vec * t
            out_cp[i, :, 1:] = pts

    else:
        out_cp = np.tile(yz_cp, (num_cp_to_use, 1, 1))

    # for i in range(5):
    #     out_cp[i, :, 0] = i
    # print("test scale")
    # scale_ratio = np.ones(scale_ratio.shape)
    cp_scale = out_cp * scale_ratio.reshape(-1, 1, 1)
    cp_scale[:, :, 0] = yz_cp[:, 0]

    # Translate base_cp
    vec_rotated = tan_vec @ R
    cp_shift = cp_scale + vec_rotated * (cp_T[:, 0].reshape(-1, 1, 1)) - np.array([x, 0, 0])
    result = (cp_shift) @ R.T + endpoint

    from scripts.sheets import plot_arr

    # plot_arr(out_cp)
    assert np.all(np.isclose(result[-1], base_cp)), "base_cp not aligned with result[-1]"

    # # Plot everything
    # fig, ax = plt.subplots()
    # t = np.linspace(0, 1, 100)
    # vals = np.polyval(poly, t)
    # ax.plot(t, vals)
    # # ax.plot(xx, yy, "-k")  # Arc
    # ax.plot(x, y, "*r")  # Intersection point
    # # ax.plot(new_cp[:, 0], new_cp[:, 1], "g-")
    # ax.set_aspect("equal")
    # plt.show()

    return result, a


def find_cp_for_desired_radius(target_radius, num_cp_per_cross_section):
    """Calculates the radius of controlpoints that will result in a B-spline with the target radius.

    B-splines do not pass through the controlpoints, so a larger controlpoint radius is needed to achieve a target B-spline radius."""
    ORDER = 3

    def make_bspline_curve(cp_r):

        # Make controlpoints
        c = np.cos
        s = np.sin
        th = np.linspace(0, 2 * np.pi, num_cp_per_cross_section, endpoint=False).reshape(-1, 1)
        cp = np.hstack((cp_r * c(th), cp_r * s(th)))

        # Make curve
        degree = ORDER - 1
        num_knots = num_cp_per_cross_section + ORDER + degree
        knot = np.linspace(0, 1, num_knots)
        basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
        curve = Curve(basis1, controlpoints=cp, rational=False)

        # Sample curve
        curve.reparam()
        t = np.linspace(0, 1, num_cp_per_cross_section)
        xy = curve(t)

        return xy

    def objective_function(cp_r):

        # Make curve
        xy = make_bspline_curve(cp_r)

        # Calc distances
        dists = np.linalg.norm(xy, axis=1)
        avg_radius = dists.mean()

        return (avg_radius - target_radius) ** 2

    res = minimize_scalar(objective_function, method="bounded", bounds=(0, target_radius**2))
    if res.success == True:
        return res.x
    else:
        return None


def make_surface(controlpoints):
    """Makes a B-spline surface given an array of controlpoints.

    The array is assumed to have shape M x N x 3. For a cylinder-like axial component, the M dimension is each of the cross sections along the backbone, and the N dimension is the X points that define each 2D cross section."""

    # Inputs
    degree = ORDER - 1

    # Basis 1 - cross section
    num_cp_per_cross_section = controlpoints.shape[1]
    num_knots = num_cp_per_cross_section + ORDER + degree
    knot = np.linspace(0, 1, num_knots)
    basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)

    # Basis 2 - along the major axis of the axial component
    num_rows = controlpoints.shape[0]
    knot = open_uniform_knot_vector(num_rows, ORDER)
    basis2 = BSplineBasis(order=ORDER, knots=knot, periodic=-1)

    # Controlpoints
    cp = controlpoints.copy()
    cp = cp.reshape(num_rows * num_cp_per_cross_section, cp.shape[2])

    # Surface
    surface = Surface(basis1, basis2, cp, rational=False)
    return surface


def make_mesh(surface, SAMPLING_DENSITY_U, SAMPLING_DENSITY_V):
    """Converts a B-spline surface into a watertight mesh."""

    NUM_ENDPOINTS = 2
    uu = SAMPLING_DENSITY_U
    vv = SAMPLING_DENSITY_V

    ####################
    # Vertices
    (us, vs) = surface.start()
    (ue, ve) = surface.end()
    u = np.linspace(us, ue, uu, endpoint=False)
    v = np.linspace(vs, ve, vv)
    verts_array = surface(u, v)
    verts = np.zeros(((uu) * (vv - 2) + NUM_ENDPOINTS, 3))
    verts[:-2, :] = verts_array[:, 1:-1, :].reshape(-1, 3, order="F")  # Skip endpoints
    verts[-2, :] = surface(0, 0)  # Add 0.0 endpoint
    verts[-1, :] = surface(1, 1)  # Add 1.0 endpoint
    verts = verts

    ####################
    # Faces - CCW Winding (for consistent normals)
    faces = np.zeros((uu * (vv - 2) * 2, 3), dtype="int")
    # faces_array = np.zeros((SD * 2, SD - 2, 3), dtype="int")
    base_column = np.zeros((uu * 2, 3), dtype="int")
    base_column[::2, 0] = np.arange(0, uu)
    base_column[1::2, 0] = np.arange(0, uu)
    base_column[::2, 1] = np.arange(uu, uu * 2)
    base_column[1::2, 1] = np.arange(uu + 1, uu * 2 + 1)
    base_column[::2, 2] = np.arange(uu + 1, uu * 2 + 1)
    base_column[1:-1:2, 2] = np.arange(1, uu)
    base_column[-2, 2] = uu  # Fix wrapping
    base_column[-1, 1] = uu  # Fix wrapping
    base_column[:, 1:] = base_column[:, :-3:-1]  # Reverse for CCW winding

    # Grid faces
    for i in range(vv - 3):
        add_to_column = i * uu
        column = base_column + add_to_column
        start = uu * i * 2
        stop = uu * (i + 1) * 2
        faces[start:stop, :] = column
        # faces_array[:, i, :] = column

    # Endpoint faces
    num_verts = verts.shape[0]
    endpoint_idx = num_verts - 2  # 0.0 endpoint
    column = np.zeros((uu, 3), dtype="int")
    column[:, 0] = np.arange(0, uu)
    column[:-1, 1] = np.arange(1, uu)
    column[:, 2] = endpoint_idx
    column[:, 1:] = column[:, :-3:-1]  # Reverse for CCW winding
    faces[-uu * 2 : -uu] = column

    # Endpoint faces
    endpoint_idx = num_verts - 1  # 1.0 endpoint
    column = np.zeros((uu, 3), dtype="int")
    column[:, 0] = np.arange(
        uu * (vv - 3),
        uu * (vv - 2),
    )
    column[:, 1] = endpoint_idx
    column[:-1, 2] = np.arange(
        uu * (vv - 3) + 1,
        uu * (vv - 2),
    )
    column[-1, 2] = uu * (vv - 3)
    column[:, 1:] = column[:, :-3:-1]  # Reverse for CCW winding
    faces[-uu:] = column
    assert ~np.any(faces > num_verts), "Faces include vertices that don't exist."
    faces = faces

    ####################
    # Skip calculations for face and vertex normals since that should be done after fusing all axial components

    ####################
    # Construct trimesh
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=False,
    )

    return mesh


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
    broken = trimesh.repair.broken_faces(union_mesh, color=[255, 0, 0, 255])
    # union_mesh.show()
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


def fair_mesh(input_mesh, neighbors, harmonic_power):

    union_mesh = input_mesh.copy()
    v = union_mesh.vertices.__array__()
    f = union_mesh.faces.__array__().astype("int64")
    num_verts = v.shape[0]
    b = np.array(list(set(range(num_verts)) - set(neighbors))).astype("int64")  # Bounday indices - NOT to be faired
    bc = v[b]  # XYZ coordinates of the boundary indices
    z = igl.harmonic_weights(v, f, b, bc, harmonic_power)  # Smooths indices at creases

    union_mesh.vertices = z
    faired_mesh = union_mesh

    return faired_mesh


def fuse_meshes(meshA, meshB, fairing_distance, operation, add_verts=None):
    """Fuses two meshes and smoothly fairs their intersection."""

    # Copy meshes to avoid altering
    mesh1 = meshA.copy()
    mesh2 = meshB.copy()

    MAX_NUM_FUSES = 2
    fair_attempt = 0
    while fair_attempt < MAX_NUM_FUSES:

        # Shift meshB if initial fuse failed
        if fair_attempt > 0:
            print("fair_attempt: {}".format(fair_attempt))
            meshB = meshB.apply_translation(np.array([0, 0, 0.1 * fair_attempt]))

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

        # union_mesh.show()
        if fairing_distance > 0:
            edge_neighbors = find_neighbors(union_mesh, edge_verts_indices, distance=fairing_distance)

            # tree = scipy.spatial.KDTree(union_mesh.vertices)
            # add_verts_neighbors_list = tree.query_ball_point(add_verts, r=0)
            # add_verts_neighbors = set()
            # for n in add_verts_neighbors_list:
            #     add_verts_neighbors.update(set(n))

            # add_verts_neighbors = list(add_verts_neighbors)
            all_neighbors = edge_neighbors  # +add_verts_neighbors

            faired_mesh = fair_mesh(union_mesh, all_neighbors, HARMONIC_POWER)

            if np.any(np.isnan(faired_mesh.vertices)):
                print("Fairing failed")
                fair_attempt += 1
                assert False
                continue
            else:
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


def get_deformation_points_along_plane(mesh, N, point):

    lines, face_index = trimesh.intersections.mesh_plane(mesh, N.ravel(), point.ravel(), return_faces=True)
    pts = lines.mean(axis=1)
    normals = mesh.face_normals[face_index]

    return pts, normals


def get_deformation_vertex(mesh, ac, dist_along_backbone, N_rotation=0):

    r = ac.r(dist_along_backbone)
    T = ac.T(dist_along_backbone)
    N = ac.N(dist_along_backbone)
    v = mesh.vertices

    rot = Rotation.from_rotvec(N_rotation * T)
    N_rot = rot.apply(N)

    # Limit vertices to those that N_rot points to
    vec = v - r
    mask = angle_between(vec, N_rot) < np.pi / 2

    dist = np.linalg.norm(np.cross((v - r), N_rot), axis=1) / np.linalg.norm(N)
    dist[mask] = dist.max()
    # idx = dist.argmin()
    # pts = mesh.vertices[idx].reshape(1, -1)
    # normals = mesh.vertex_normals[idx].reshape(1, -1)
    # Take points within 3mm of N_rot
    idx = (dist < 3) & (dist > 2)
    pts = mesh.vertices[idx].mean(axis=0).reshape(1, -1)
    normals = mesh.vertex_normals[idx].mean(axis=0).reshape(1, -1)

    return pts, normals


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
