# # Test if two meshes are duplicates (i.e. can be rotated around z-axis) to have same volume, using voxelization

import trimesh
from pathlib import Path
import numpy as np


def create_uniform_voxel_grids(mesh1_sliced_voxel, mesh2_sliced_voxel, pitch):

    min_corner = np.min([mesh1_sliced_voxel.translation, mesh2_sliced_voxel.translation], axis=0)
    max_corner = np.max(
        [
            mesh1_sliced_voxel.translation + np.array(mesh1_sliced_voxel.shape) * pitch,
            mesh2_sliced_voxel.translation + np.array(mesh2_sliced_voxel.shape) * pitch,
        ],
        axis=0,
    )
    big_dims = np.round((max_corner - min_corner) / pitch).astype(int)
    # big_dims = np.maximum(mesh1_sliced_voxel.shape, mesh2_sliced_voxel.shape)
    big_T = mesh1_sliced_voxel.transform.copy()
    big_T[:3, 3] = np.min([mesh1_sliced_voxel.transform[:3, 3], mesh2_sliced_voxel.transform[:3, 3]], axis=0)
    big_matrix = np.ones(big_dims, dtype=bool)

    big_voxel1 = trimesh.voxel.VoxelGrid(big_matrix, transform=big_T)
    big_voxel1.matrix[:, :, :] = False

    sparse_indices = mesh1_sliced_voxel.sparse_indices
    xshift = round(-(big_voxel1.transform[0, 3] - mesh1_sliced_voxel.transform[0, 3]) / pitch)
    yshift = round(-(big_voxel1.transform[1, 3] - mesh1_sliced_voxel.transform[1, 3]) / pitch)
    zshift = round(-(big_voxel1.transform[2, 3] - mesh1_sliced_voxel.transform[2, 3]) / pitch)
    big_indices = sparse_indices + np.array([xshift, yshift, zshift])
    big_voxel1.matrix[big_indices[:, 0], big_indices[:, 1], big_indices[:, 2]] = True

    for p in mesh1_sliced_voxel.points:
        if p not in big_voxel1.points:
            raise ValueError("Point not in big_voxel1")
    assert len(mesh1_sliced_voxel.points) == len(big_voxel1.points) > 0

    # Create big for 2
    big_voxel2 = trimesh.voxel.VoxelGrid(big_matrix, transform=big_T)
    big_voxel2.matrix[:, :, :] = False
    sparse_indices = mesh2_sliced_voxel.sparse_indices
    xshift = round(-(big_voxel2.transform[0, 3] - mesh2_sliced_voxel.transform[0, 3]) / pitch)
    yshift = round(-(big_voxel2.transform[1, 3] - mesh2_sliced_voxel.transform[1, 3]) / pitch)
    zshift = round(-(big_voxel2.transform[2, 3] - mesh2_sliced_voxel.transform[2, 3]) / pitch)
    big_indices = sparse_indices + np.array([xshift, yshift, zshift])
    big_voxel2.matrix[big_indices[:, 0], big_indices[:, 1], big_indices[:, 2]] = True

    for p in mesh2_sliced_voxel.points:
        if p not in big_voxel2.points:
            raise ValueError("Point not in big_voxel2")
    assert len(mesh2_sliced_voxel.points) == len(big_voxel2.points) > 0

    return big_voxel1, big_voxel2


def test_meshes_are_duplicates(mesh1, mesh2, pitch, threshold):

    mesh1_sliced, mesh1_sliced_voxel = mesh_slice_voxelize(mesh1, pitch)
    mesh2_sliced, mesh2_sliced_voxel = mesh_slice_voxelize(mesh2, pitch)

    big_voxel1, big_voxel2 = create_uniform_voxel_grids(mesh1_sliced_voxel, mesh2_sliced_voxel, pitch)

    intersection = np.logical_and(big_voxel1.matrix, big_voxel2.matrix)
    union = np.logical_or(big_voxel1.matrix, big_voxel2.matrix)
    overlap_fraction = intersection.sum() / union.sum()

    # Print result
    # print(f"Overlap fraction: {overlap_fraction}")

    # # Plot voxel grids together
    # scene = trimesh.Scene()
    # scene.add_geometry(big_voxel1.as_boxes(colors=(1, 0, 0, 0.3)))
    # scene.add_geometry(big_voxel2.as_boxes(colors=(0, 1, 0, 0.3)))
    # scene.show()
    are_duplicates = overlap_fraction > threshold
    return are_duplicates, overlap_fraction


def mesh_slice_voxelize(mesh, pitch, plane_origin=[0, 0, 0], plane_normal=[0, 0, 1]):
    slice_result = mesh.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal)
    voxelized_mesh = slice_result.voxelized(pitch)
    return slice_result, voxelized_mesh


pitch = 2
# mesh1_fname = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet/sheet_005.stl")
# mesh1 = trimesh.load(mesh1_fname)
# mesh2_fname = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet/sheet_007.stl")
# mesh2 = trimesh.load(mesh2_fname)
# for th in np.linspace(0, 2 * np.pi, 4, endpoint=False):
#     mesh2.apply_transform(trimesh.transformations.rotation_matrix(th, [0, 0, 1]))
#     test_meshes_are_duplicates(mesh1, mesh2, pitch, 0.9)


stl_list = list(Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet").rglob("*.stl"))
stl_list.sort()
for i in range(len(stl_list)):
    for j in range(i + 1, len(stl_list)):
        mesh1 = trimesh.load(stl_list[i])
        mesh2 = trimesh.load(stl_list[j])
        for th in np.linspace(0, 2 * np.pi, 4, endpoint=False):
            mesh2.apply_transform(trimesh.transformations.rotation_matrix(th, [0, 0, 1]))
            are_duplicates, overlap_fraction = test_meshes_are_duplicates(mesh1, mesh2, pitch, 0.75)
            if are_duplicates:
                print(
                    f"{stl_list[i].name} and {stl_list[j].name} at theta {np.round(th,3)} are duplicates. {overlap_fraction}"
                )
                break
# mesh2.apply_transform(trimesh.transformations.rotation_matrix(0.3, [0, 0, 1]))
# # mesh2_sliced, mesh2_sliced_voxel = mesh_slice_voxelize(mesh2, pitch)

# pitch = 2
# test_meshes_are_duplicates(mesh1, mesh2, pitch, 0.9)

# # # This works!!!
# # for p in mesh1_sliced_voxel.points:
# #     # idx = big_voxel.points_to_indices(p)
# #     # big_voxel.matrix[idx[0], idx[1], idx[2]] = True

# #     print(p in big_voxel.points)
# # len(big_voxel.points)


# # To compare the two meshes, we have to create a larger voxel grid that encompasses both meshes
# big_dims = np.maximum(mesh1_sliced_voxel.shape, mesh2_sliced_voxel.shape)
# big_bounds = np.vstack([np.min([mesh1_sliced_voxel.bounds[0], mesh2_sliced_voxel.bounds[0]], axis=0),
#                         np.max([mesh1_sliced_voxel.bounds[1], mesh2_sliced_voxel.bounds[1]], axis=0)])
# T = np.eye(4)
# T[:3, 3] = big_bounds[0] + pitch/2
# T[0,0] = T[1,1] = T[2,2] = pitch
# big1 = np.ones(big_dims, dtype=bool)
# voxel_big1 = trimesh.voxel.VoxelGrid(big1, transform=T)
# voxel_big1.matrix[:] = False

# # Assign the voxels of the first mesh
# for p in mesh1_sliced_voxel.points:
#     idx  = voxel_big1.points_to_indices(p)
#     voxel_big1.matrix[idx[0], idx[1], idx[2]] = True

# voxel_big1.matrix[voxel_big1.points_to_indices(mesh1_sliced_voxel.points)] = True
#     voxel_big1.matrix[x, y, z
# x0, y0, z0 = np.round((mesh1_sliced_voxel.bounds[0] - big_bounds[0]) / pitch).astype(int)
# voxel_big1.matrix[x0:x0+mesh1_sliced_voxel.shape[0], y0:y0+mesh1_sliced_voxel.shape[1], z0:z0+mesh1_sliced_voxel.shape[2]] = mesh1_sliced_voxel.matrix


# voxel_big1 = trimesh.voxel.VoxelGrid(big1, transform=T)

# # Create a larger voxel grid that encompasses both meshes
# voxel_grid1 = trimesh.voxel.VoxelGrid(voxel_grid1, transform=mesh1_sliced_voxel.transform)
# voxel_grid2 = trimesh.voxel.VoxelGrid(voxel_grid2, transform=mesh2_sliced_voxel.transform)

# voxel_width = 1
# voxelized_mesh = slice_result.voxelized(voxel_width)


# voxel_size = 0.1

# # Get combined bounds
# combined_bounds = trimesh.bounds.union_bounds([mesh1.bounds, mesh1.bounds])

# # Create a voxel grid using the combined bounds
# vox1 = trimesh.voxel.creation.voxelize(mesh1, pitch=voxel_size, bounds=combined_bounds).dense
# vox2 = trimesh.voxel.creation.voxelize(mesh2, pitch=voxel_size, bounds=combined_bounds).dense

# vox1 = trimesh.voxel.creation.voxelize(mesh1, point)
# # # slice_voxel = voxelize(slice_result)


# # # mesh2 = mesh1.copy()

# # # mesh2.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))


# # # def voxelize(mesh):
# # #     # Create voxel grid
# # #     voxel_size =1
# # #     mesh_voxel = mesh.voxelized(voxel_size)

# # #     # Get voxel grid
# # #     voxel_grid = mesh_voxel.matrix

# # #     return voxel_grid


# # def mesh_to_voxel(stl_file, voxel_subdivisions, plane_origin=[0, 0, 0], plane_normal=[0, 0, 1]):
# #     mesh = trimesh.load(stl_file)
# #     slice_result = mesh.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal)
# #     slice_voxel = voxelize(slice_result, voxel_subdivisions)
# #     return slice_result, slice_voxel


# # voxel_subdivisions = 2.0
# # mesh1_fname = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet/sheet_017.stl")
# # mesh1 = trimesh.load(mesh1_fname)
# # mesh1_sliced, mesh1_sliced_voxel = mesh_to_voxel(mesh1, voxel_subdivisions)

# # mesh2 = mesh1.copy()
# # mesh2.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]))
# # mesh2_sliced, mesh2_sliced_voxel = mesh_to_voxel(mesh2, voxel_subdivisions)

# # print(f"{count_overlapping_voxels(mesh1_sliced_voxel, mesh2_sliced_voxel)} / {mesh1_sliced_voxel.matrix.sum()} overlapping voxels")

# # show(mesh1, mesh1_sliced_voxel)

# # import trimesh
# # import numpy as np

# # def voxelize_interval(mesh, bounds=(-50, 50), spacing=2):
# #     """
# #     Voxelizes the mesh within the specified interval for x, y, and z.

# #     Parameters:
# #         mesh (trimesh.Trimesh): The input mesh to voxelize.
# #         bounds (tuple): The lower and upper bounds for x, y, and z (e.g., (-50, 50)).
# #         spacing (float): The spacing between voxel centers.

# #     Returns:
# #         np.ndarray: A 3D boolean array where True represents a filled voxel.
# #         np.ndarray: The origin of the voxel grid (minimum corner of the bounds).
# #     """
# #     # Generate a grid of points in the specified bounds
# #     lower, upper = bounds
# #     grid_coords = np.arange(lower, upper + spacing, spacing)
# #     x, y, z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')

# #     # Stack the coordinates into a (N, 3) array
# #     points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

# #     # Determine which points are inside the mesh
# #     inside = mesh.contains(points)

# #     # Create a 3D boolean array to represent the voxel grid
# #     shape = (len(grid_coords), len(grid_coords), len(grid_coords))
# #     voxel_grid = inside.reshape(shape)

# #     return voxel_grid, lower

# # # Example usage
# # if __name__ == "__main__":
# #     # Load your mesh
# #     mesh = trimesh.load("path_to_your_mesh.stl")

# #     # Voxelize the mesh over the interval [-50, 50] with spacing 2
# #     voxel_grid, origin = voxelize_interval(mesh, bounds=(-50, 50), spacing=2)

# #     print(f"Voxel grid shape: {voxel_grid.shape}")
# #     print(f"Grid origin: {origin}")

# #     # Visualization (optional)
# #     voxelized_mesh = trimesh.voxel.VoxelGrid(encoding=voxel_grid, transform=np.eye(4))
# #     scene = mesh.scene()
# #     scene.add_geometry(voxelized_mesh.as_boxes(colors=(0, 0, 1, 0.3)))
# #     scene.show()
