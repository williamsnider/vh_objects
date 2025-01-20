# # Load mesh
# # Define texture
# # Apply texture
# # Compare meshes

# import trimesh
# from pathlib import Path

# import numpy as np
# from scipy.spatial import cKDTree


# def maximize_spheres(points, radius, runs=10):
#     best_solution = []
#     best_count = 0
#     radius_check = 2 * radius

#     for _ in range(runs):
#         # Randomly shuffle points
#         np.random.shuffle(points)

#         # Initialize accepted points and KDTree
#         accepted_points = []
#         tree = None  # Start with no tree

#         for point in points:
#             if tree is None:
#                 # Accept the first point unconditionally
#                 accepted_points.append(point)
#                 tree = cKDTree([point])
#             else:
#                 # Query the tree for nearby points within radius_check
#                 nearby = tree.query_ball_point(point, radius_check)
#                 if not nearby:
#                     # No overlap, accept the point
#                     accepted_points.append(point)
#                     # Update the tree
#                     tree = cKDTree(accepted_points)

#         # Check if this solution is better
#         if len(accepted_points) > best_count:
#             best_solution = accepted_points
#             best_count = len(accepted_points)

#     return np.array(best_solution)


# # Load mesh
# mesh1_fname = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet/sheet_017.stl")
# mesh1 = trimesh.load(mesh1_fname)
# mesh1.show()

# # Choose verts
# verts = mesh1.vertices.copy()

# pts, face_idx = trimesh.sample.sample_surface_even(mesh1, 100000)
# pts = pts[pts[:, 2] > 0.1]


# sphere_points = maximize_spheres(pts, 1.0, runs=10)


# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pts, size=0.01))
# scene.add_geometry(mesh1)
# scene.show()


# # Subtract from mesh
# e = 1.0
# texture_unit = trimesh.creation.icosphere(radius=1, subdivisions=1)

# # Aggregate into a single mesh
# agg_faces = []
# agg_verts = []
# for i in range(len(sphere_points)):
#     p = sphere_points[i]
#     t = texture_unit.copy()
#     t.apply_translation(p + 1e-2)

#     agg_faces.extend((t.faces + len(agg_verts)).tolist())
#     agg_verts.extend((t.vertices).tolist())

# agg_mesh = trimesh.Trimesh(vertices=agg_verts, faces=agg_faces, process=True)
# agg_mesh.show()


# output = mesh1.copy()
# for i in range(len(pts)):
#     print(i)

#     p = pts[i]
#     t = texture_unit.copy()
#     t.apply_translation(p + 1e-2)

#     # Subtract
#     output = fuse_meshes(output, t, 0, operation="difference")

# output.show(smooth=False)


import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import trimesh
from vh_objects.utilities import fuse_meshes


def poisson_disk_sampling(points, min_dist):
    """
    Downsample a 3D point cloud using Poisson disk sampling.

    Parameters:
        points (numpy.ndarray): Nx3 array of 3D points.
        radius (float): Minimum distance between sampled points.

    Returns:
        numpy.ndarray: Downsampled point cloud.
    """
    # Shuffle the input points to reduce bias
    np.random.shuffle(points)

    # Initialize the sampled point list
    sampled_points = []

    # Build a spatial acceleration structure (k-d tree)

    for point in points:
        if len(sampled_points) == 0:
            sampled_points.append(point)
            tree = cKDTree(sampled_points)
            continue

        # Check if the current point is at least `radius` away from all sampled points
        distances, indices = tree.query(point, len(sampled_points), distance_upper_bound=min_dist)

        # If no sampled points are within the radius, add the current point
        if np.all(np.isinf(distances)):
            sampled_points.append(point)
            tree = cKDTree(sampled_points)

    return np.array(sampled_points)


def apply_blue_noise_to_mesh(mesh, min_dist, texture_element, operation, z_min=0.0):

    # Sample the mesh
    pts, _ = trimesh.sample.sample_surface_even(mesh, 100000)
    pts = pts[pts[:, 2] > z_min]

    # Downsample the point cloud with Poisson disk sampling
    downsampled_cloud = poisson_disk_sampling(pts, min_dist)

    # Check constraints met
    tree = cKDTree(downsampled_cloud)
    distances, indices = tree.query(downsampled_cloud, k=2)
    assert np.all(distances[:, 1] >= min_dist)
    assert np.all(downsampled_cloud[:, 2] > z_min)

    # Aggregate into a single mesh
    agg_faces = []
    agg_verts = []
    for i in range(len(downsampled_cloud)):
        p = downsampled_cloud[i]
        t = texture_element.copy()
        t.apply_translation(p + 1e-2)

        agg_faces.extend((t.faces + len(agg_verts)).tolist())
        agg_verts.extend((t.vertices).tolist())

    agg_mesh = trimesh.Trimesh(vertices=agg_verts, faces=agg_faces, process=True)

    # Apply the operation
    textured_mesh = trimesh.boolean.boolean_manifold([mesh, agg_mesh], operation=operation)

    return textured_mesh


def distance_matrix(points):
    """Calculate the distance matrix for all points."""
    points = np.array(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def sort_points_circle(points):
    """Sort points in circular order starting from the first point."""
    points = np.array(points)
    sorted_indices = [0]  # Start with the first point
    remaining_indices = set(range(1, len(points)))  # Remaining points to sort

    while remaining_indices:
        last_index = sorted_indices[-1]
        # Compute distances from the last sorted point to remaining points
        distances = np.linalg.norm(points[list(remaining_indices)] - points[last_index], axis=1)
        nearest_index = list(remaining_indices)[np.argmin(distances)]
        sorted_indices.append(nearest_index)
        remaining_indices.remove(nearest_index)

    return points[sorted_indices]


# Example usage:
if __name__ == "__main__":

    # # Load mesh
    # mesh1_fname = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet/sheet_017.stl")
    # mesh1 = trimesh.load(mesh1_fname)

    # # Define texture
    # radius = 0.5
    # texture_element = trimesh.creation.icosphere(radius=0.5, subdivisions=1)

    # # Apply texture
    # textured_mesh = apply_blue_noise_to_mesh(
    #     mesh=mesh1,
    #     min_dist=1.0,
    #     texture_element=texture_element,
    #     operation="difference",
    #     z_min=0.0,
    # )

    # # Compare meshes
    # scene = trimesh.Scene()
    # xshift = 0
    # for m in [mesh1, textured_mesh]:
    #     m_copy = m.copy()
    #     m.apply_translation([xshift, 0, 0])
    #     xshift += m.extents[0]
    #     scene.add_geometry(m)
    # scene.show()

    # Add linear pattern
    mesh1_fname = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet/sheet_017.stl")
    mesh1 = trimesh.load(mesh1_fname)
    mesh1.apply_scale([1, 2, 1])

    # Find outline of mesh sliced by plane
    plane_origin = np.array([0, 0, 20])
    plane_normal = np.array([0, 0, -1])

    section = mesh1.section(plane_origin=plane_origin, plane_normal=plane_normal)

    # Sort vertices by their nearest neighbors (can assume it is)
    sorted_points = sort_points_circle(section.vertices)
    sorted_points = np.vstack([sorted_points, sorted_points[0]])  # Close loop

    from vh_objects.backbone import Backbone

    b = Backbone(controlpoints=sorted_points, reparameterize=True)

    # # Plot in 2D
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # t = np.linspace(0, 1, 100)
    # p = b.r(t)
    # ax.plot(p[:, 0], p[:, 1], "o-")
    # ax.set_aspect("equal")
    # plt.show()

    from vh_objects.cross_section import CrossSection

    # Create cross section
    th = np.linspace(0, 2 * np.pi, 8)
    cp = np.array([np.cos(th), np.sin(th)]).T

    # TODO: Chase down the number of cross sections impacting the correct vv ratio.
    cs_list = [CrossSection(cp, position=pos) for pos in np.linspace(0, 1, 10)]

    # Create axial component
    from vh_objects.axial_component import AxialComponent

    ac = AxialComponent(b, cs_list)

    from scripts.sheets import plot_arr

    # plot_arr(ac.controlpoints[2:-2])

    cp = ac.controlpoints.copy()

    # Adjust to complete tube
    cp[0] = cp[-5]
    cp[1] = cp[-4]
    cp[2] = cp[-3]  # Match the first and last points
    cp[-2] = cp[3]
    cp[-1] = cp[4]

    # plot_arr(cp)

    from vh_objects.utilities import make_surface, make_mesh

    surface = make_surface(cp)
    # mesh = make_mesh(surf, 100,100)
    # mesh.show(smooth=False)

    uu = 100
    vv = 100
    NUM_ENDPOINTS = 2

    (us, vs) = surface.start()
    (ue, ve) = surface.end()
    vs = 0.125  # TODO: decipher why this is necessary
    ve = 1 - vs
    u = np.linspace(us, ue, uu, endpoint=False)
    v = np.linspace(vs, ve, vv)
    verts_array = surface(u, v)

    # # Plot 3D
    # from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # for col in range(verts_array.shape[1]):
    #     ax.plot(verts_array[:, col, 0], verts_array[:, col, 1], verts_array[:, col, 2], "g-")

    # # Plot endpoints
    # ax.plot(verts_array[0, 0, 0], verts_array[0, 0, 1], verts_array[0, 0, 2], "ro")
    # ax.plot(verts_array[1, 0, 0], verts_array[1, 0, 1], verts_array[1, 0, 2], "bo")
    # ax.plot(verts_array[0, -1, 0], verts_array[0, -1, 1], verts_array[0, -1, 2], "kx")
    # ax.plot(verts_array[1, -1, 0], verts_array[1, -1, 1], verts_array[1, -1, 2], "bx")

    # # Set equal aspect ratio
    # max_range = np.array([verts_array.max(axis=0), verts_array.min(axis=0)]).ptp()
    # ax.set_xlim(-max_range, max_range)
    # ax.set_ylim(-max_range, max_range)
    # ax.set_zlim(-max_range, max_range)

    # plt.show()

    # Stitch tube
    assert np.all(np.isclose(verts_array[:, 0, :], verts_array[:, -1, :]))

    verts = verts_array[:, :-1, :].reshape(-1, 3, order="F")

    faces = np.zeros((uu * (vv - 1) * 2, 3), dtype="int")
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
    for i in range(vv - 1):
        add_to_column = i * uu
        column = base_column + add_to_column
        start = uu * i * 2
        stop = uu * (i + 1) * 2
        faces[start:stop, :] = column

    faces = faces % (uu * (vv - 1))

    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Combine
    output_mesh = trimesh.boolean.union([m, mesh1])
    output_mesh.show(smooth=False)
