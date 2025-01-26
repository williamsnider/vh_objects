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

    # # Shuffle the input points to reduce bias
    # np.random.shuffle(points)

    # Sort points by descending z for better packing
    points = points[np.argsort(points[:, 2])[::-1]]

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


def apply_blue_noise_to_mesh(mesh, min_dist, texture_element, operation, z_min=0.0, rexclude=6.0):

    mesh = mesh.copy()

    # Sample the mesh
    pts, _ = trimesh.sample.sample_surface_even(mesh, 1000000)

    # Exclude points at post
    pts = pts[pts[:, 2] > -20.0]
    within_rexclude = np.logical_and(np.linalg.norm(pts[:, :2], axis=1) < rexclude, pts[:, 2] < z_min)
    pts = pts[~within_rexclude]

    # Downsample the point cloud with Poisson disk sampling
    downsampled_cloud = poisson_disk_sampling(pts, min_dist)

    # Check constraints met
    tree = cKDTree(downsampled_cloud)
    distances, indices = tree.query(downsampled_cloud, k=2)
    assert np.all(distances[:, 1] >= min_dist)
    # assert np.all(downsampled_cloud[:, 2] > z_min)

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

    # # Apply the operation
    # if operation == "union":
    #     textured_mesh = trimesh.boolean.union([mesh, agg_mesh])
    # elif operation == "difference":
    #     textured_mesh = trimesh.boolean.difference([mesh, agg_mesh])

    # Apply operation using manifold3d
    mesh_sequence = [mesh, agg_mesh]
    if operation == "union":
        textured_mesh = trimesh.boolean.boolean_manifold(mesh_sequence)
    elif operation == "difference":
        textured_mesh = trimesh.boolean.boolean_manifold(mesh_sequence, operation="difference")

    # Label broken faces
    if not textured_mesh.is_watertight:
        for face in textured_mesh.faces:
            if not textured_mesh.face_adjacency[face].size:
                textured_mesh.visual.face_colors[face] = [255, 0, 0, 255]
        textured_mesh.show()
    assert textured_mesh.is_watertight

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


def create_entity_tube(entity_verts, radius):

    sorted_points = sort_points_circle(entity_verts)
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
    cp *= radius

    # TODO: Chase down the number of cross sections impacting the correct vv ratio.
    cs_list = [CrossSection(cp, position=pos) for pos in np.linspace(0, 1, 100)]

    # Create axial component
    from vh_objects.axial_component import AxialComponent

    ac = AxialComponent(b, cs_list)

    from scripts.sheets_utilities import plot_arr

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

    uu = 20
    vv = 300
    NUM_ENDPOINTS = 2

    (us, vs) = surface.start()
    (ue, ve) = surface.end()
    vs = 3 / (2 * (cp.shape[0] - 2))  # TODO: decipher why this is necessary
    ve = 1 - vs
    u = np.linspace(us, ue, uu, endpoint=False)
    v = np.linspace(vs, ve, vv)
    verts_array = surface(u, v)

    # # Plot 3D
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt

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
    assert np.all(np.abs(verts_array[:, 0, :] - verts_array[:, -1, :] < 0.01))

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

    return m


def apply_linear_texture_to_mesh(mesh, tube_radius, tube_spacing, operation, zmin, rexclude):
    mesh = mesh.copy()

    # Slice mesh to identify sections to apply texture
    plane_normal = np.array([0, 0, -1])
    section_list = []
    for z in np.arange(-20, mesh.bounds[1, 2], tube_spacing):
        plane_origin = np.array([0, 0, z])
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is not None:
            section_list.append(section)

    # Create list of tubes (textures)
    etube_list = []
    for section in section_list:
        for i in range(len(section.entities)):
            sub_verts = section.vertices[section.entities[i].points]

            # Skip small sections
            if len(sub_verts) < 10:
                continue

            # Skip sections that are solely within the exclusion region (i.e. the post)
            if (
                np.all(np.linalg.norm(sub_verts[:, :2], axis=1) < rexclude)
                and (np.all(sub_verts[:, 2] < zmin))
                or np.any(sub_verts[:, 2] < -20)
            ):
                continue

            etube = create_entity_tube(sub_verts, tube_radius)
            etube_list.append(etube)

    # Aggregate into a single mesh
    agg_faces = []
    agg_verts = []
    for i in range(len(etube_list)):
        t = etube_list[i].copy()

        agg_faces.extend((t.faces + len(agg_verts)).tolist())
        agg_verts.extend((t.vertices).tolist())

    agg_mesh = trimesh.Trimesh(vertices=agg_verts, faces=agg_faces, process=True)

    # # Apply the operation
    # if operation == "union":
    #     textured_mesh = trimesh.boolean.union([mesh, agg_mesh])
    # elif operation == "difference":
    #     textured_mesh = trimesh.boolean.difference([mesh, agg_mesh])

    # Apply operation using manifold3d
    mesh_sequence = [mesh, agg_mesh]
    if operation == "union":
        textured_mesh = trimesh.boolean.boolean_manifold(mesh_sequence)
    elif operation == "difference":
        textured_mesh = trimesh.boolean.boolean_manifold(mesh_sequence, operation="difference")

    return textured_mesh


def apply_voxelization_texture_to_mesh(input_mesh, voxel_size, zmin, rexclude):

    pitch = voxel_size
    mesh = input_mesh.copy()

    # Remove cap region
    cap_radius = 10
    slice_box = trimesh.primitives.Box([cap_radius * 2, cap_radius * 2, 0.1])
    slice_box.apply_translation([0, 0, -slice_box.bounds[1, 2]])  # Top of box is at xy plane
    meshes = trimesh.boolean.difference([mesh, slice_box])

    # Isolate top watertight portion
    meshes = meshes.split()
    assert len(meshes) == 2
    zval = -np.inf
    target_mesh = None
    for m in meshes:
        if m.bounds[0, 2] > zval:
            zval = m.bounds[0, 2]
            mesh = m

    # Voxelize and fill
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxel_grid.fill()

    # # Convert to mesh using marching cubes
    # mc = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid.matrix, pitch=pitch / 10)
    # mc.show()

    voxels = voxel_grid.as_boxes()
    verts = voxels.vertices
    faces = voxels.faces
    m_new = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    print(m_new.is_watertight)
    m_new.show(smooth=False)

    voxel_grid.show()


# voxel_grid = trimesh.voxel.VoxelGrid(mesh, pitch=pitch)


# Example usage:
save_list = []

if __name__ == "__main__":

    fname_list = [
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/medial_axis/G013.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/medial_axis/G044.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/medial_axis/G151.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/medial_axis/G251.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/medial_axis/G261.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso/G600.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso/G601.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso/G602.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso/G603.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso/G622.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso/G657.stl"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/torso/G746.stl"),
    ]
    # torso_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/axial_component")
    # fname_list = list(torso_dir.glob("*.stl"))
    # fname_list.sort()
    # fname_list = fname_list[:12]

    # input_mesh = trimesh.load(fname_list[0])
    # voxel_size = 1
    # apply_voxelization_texture_to_mesh(input_mesh, voxel_size, 5, 5)

    for fname in fname_list:
        mesh = trimesh.load(fname)

        # Apply blue noise
        radius = 1.0
        min_dist = 2 * radius + 0.5
        operation = "difference"
        texture_element = trimesh.creation.icosphere(radius=radius, subdivisions=1)
        texture_element.apply_translation(2e-2 * np.ones(3))
        textured_mesh = apply_blue_noise_to_mesh(mesh, min_dist, texture_element, operation, z_min=radius, rexclude=6.0)
        new_name = "A" + fname.stem[-3:]
        new_path = Path(fname.parent.parent, "texture", f"{new_name}.stl")
        save_list.append([new_path, textured_mesh])

    for fname in fname_list:
        mesh = trimesh.load(fname)

        # Apply blue noise
        radius = 2.5
        min_dist = 2 * radius + 0.5
        operation = "difference"
        texture_element = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        texture_element.apply_translation(2e-2 * np.ones(3))
        textured_mesh = apply_blue_noise_to_mesh(mesh, min_dist, texture_element, operation, z_min=radius, rexclude=6.0)
        new_name = "B" + fname.stem[-3:]
        new_path = Path(fname.parent.parent, "texture", f"{new_name}.stl")
        save_list.append([new_path, textured_mesh])
    for fname in fname_list:

        mesh = trimesh.load(fname)

        # Apply linear texture
        tube_radius = 0.5
        tube_spacing = 3 * tube_radius
        textured_mesh = apply_linear_texture_to_mesh(
            mesh=mesh,
            tube_radius=tube_radius,
            tube_spacing=tube_spacing,
            operation="difference",
            zmin=0.0,
            rexclude=6.0,
        )
        new_name = "C" + fname.stem[-3:]
        new_path = Path(fname.parent.parent, "texture", f"{new_name}.stl")
        save_list.append([new_path, textured_mesh])

    for i, (savepath, mesh) in enumerate(save_list):

        savepath.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(savepath)
