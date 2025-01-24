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


def test_voxel_grids_are_duplicates(big_voxel1, big_voxel2, threshold):

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


def find_duplicates_in_dict(voxel_dict, threshold):

    mesh_names = list(voxel_dict.keys())

    duplicate_meshes = []
    duplicate_pairs = []
    overlap_fractions = []
    for i in range(len(mesh_names)):

        # # Skip j if already found to be duplicate
        # if mesh_names[i] in duplicate_meshes:
        #     print("Skipping ", mesh_names[i])
        #     continue

        for j in range(i + 1, len(mesh_names)):

            voxel_grid_i = voxel_dict[mesh_names[i]][0]

            for k in range(len(voxel_dict[mesh_names[j]])):
                voxel_grid_j = voxel_dict[mesh_names[j]][k]

                are_duplicates, overlap_fraction = test_voxel_grids_are_duplicates(
                    voxel_grid_i, voxel_grid_j, threshold
                )
                if are_duplicates:
                    print(
                        f"{mesh_names[i]} and {mesh_names[j]} at theta {k} are duplicates. {np.round(overlap_fraction,3)}"
                    )
                    duplicate_meshes.append(mesh_names[j])
                    overlap_fractions.append(overlap_fraction)
                    duplicate_pairs.append((mesh_names[i], mesh_names[j]))

                    break

    print("***********")
    print("Duplicate meshes: ")
    overlap_fractions = np.array(overlap_fractions)
    idx = overlap_fractions.argsort()
    idx = idx[::-1]

    overlap_fractions_sorted = overlap_fractions[idx]
    duplicate_meshes_sorted = np.array(duplicate_meshes)[idx]
    duplicate_pairs_sorted = np.array(duplicate_pairs)[idx]

    # Remove duplicates

    # duplicate_meshes = list(set(duplicate_meshes))
    # duplicate_meshes.sort()
    for i in range(len(duplicate_meshes_sorted)):
        print(duplicate_pairs_sorted[i][0], duplicate_pairs_sorted[i][1], overlap_fractions_sorted[i], sep="\t")


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


def voxelize_given_bounds(mesh, pitch, bounds):

    # Assert bounds are multiple of pitch
    assert np.all(np.diff(bounds, axis=0)[0] % pitch == 0), "Bounds must be multiple of pitch"
    multiple = bounds[0] / pitch

    # Voxelize
    mesh1_voxel = mesh.voxelized(pitch)
    indices = ((mesh1_voxel.points - bounds[0]) / pitch).astype(int)

    dims = np.diff(bounds, axis=0)[0] // pitch

    # Remove indices outside bounds
    mask = np.zeros(indices.shape, dtype=bool)
    mask[:, 0] = np.logical_and(indices[:, 0] >= 0, indices[:, 0] < dims[0] - 1)
    mask[:, 1] = np.logical_and(indices[:, 1] >= 0, indices[:, 1] < dims[1] - 1)
    mask[:, 2] = np.logical_and(indices[:, 2] >= 0, indices[:, 2] < dims[2] - 1)
    mask = np.all(mask, axis=1, keepdims=True)

    # Filter out the excess regions
    valid_indices = indices[mask.reshape(-1)]

    # Create matrix with just the valid indices true
    matrix = np.zeros(np.round((np.diff(bounds, axis=0)[0] / pitch)).astype(int), dtype=bool)
    # matrix[valid_indices] = True
    matrix[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True

    # Convert to voxel grid
    T = np.eye(4)
    T[:3, :3] *= pitch
    T[:3, 3] = bounds[0]
    voxel_grid = trimesh.voxel.VoxelGrid(matrix, transform=T)
    # voxel_grid.show()

    # Test all points the same
    test_points = voxel_grid.points.tolist()
    mesh1_points = mesh1_voxel.points.tolist()
    for tp in test_points:
        assert tp in mesh1_points

    assert [-1000, -1000, -1000] not in mesh1_points

    # Check voxel all good
    assert np.all(voxel_grid.translation == bounds[0])
    assert np.all(voxel_grid.shape == dims)
    assert np.all(voxel_grid.pitch == pitch)
    return voxel_grid


# def generate_voxel_grid_of_stl_in_directory(dirname, pitch, bounds, zrotation=8):

#     # Get list of all stl's in directory
#     stl_list = list(dirname.rglob("*.stl"))
#     stl_list.sort()

#     voxel_dict = {}

#     # Iterate through all stl's
#     for i in range(len(stl_list)):

#         print("Voxelizing ", stl_list[i].name, " at ", zrotation, " rotations...")
#         input_mesh = trimesh.load(stl_list[i])
#         voxel_dict[stl_list[i].name] = {}

#         # Rotate mesh
#         for j in range(zrotation):
#             mesh = input_mesh.copy()
#             mesh.apply_transform(trimesh.transformations.rotation_matrix((j / zrotation) * 2 * np.pi, [0, 0, 1]))

#             # Convert to voxel grid
#             voxel_grid = voxelize_given_bounds(mesh, pitch, bounds)

#             voxel_dict[stl_list[i].name][j] = voxel_grid

#     return voxel_dict

import multiprocessing
from functools import partial
import trimesh
import numpy as np


def process_stl(stl_path, pitch, bounds, zrotation):
    """
    Process a single STL file: rotate, voxelize, and return the results.
    """
    input_mesh = trimesh.load(stl_path)
    result = {}

    for j in range(zrotation):
        mesh = input_mesh.copy()
        mesh.apply_transform(trimesh.transformations.rotation_matrix((j / zrotation) * 2 * np.pi, [0, 0, 1]))
        voxel_grid = voxelize_given_bounds(mesh, pitch, bounds)
        result[j] = voxel_grid

    return stl_path.name, result


def generate_voxel_grid_of_stl_in_directory(dirname, pitch, bounds, zrotation=8):
    """
    Parallelized version to voxelize STL files in a directory.
    """
    # Get list of all STL files in the directory
    stl_list = list(dirname.rglob("*.stl"))
    stl_list.sort()

    voxel_dict = {}

    import tqdm

    with multiprocessing.Pool() as pool:
        process_func = partial(process_stl, pitch=pitch, bounds=bounds, zrotation=zrotation)

        # Wrap pool.map with tqdm for the progress bar
        results = list(tqdm.tqdm(pool.imap(process_func, stl_list), total=len(stl_list), desc="Processing STL files"))

    # Collect results into voxel_dict
    for stl_name, result in results:
        voxel_dict[stl_name] = result

    return voxel_dict


import sys


def get_nested_dict_size(obj):
    """Recursively calculate the size of a dictionary (or any nested structure) in MB."""
    seen_ids = set()

    def _sizeof(o):
        if id(o) in seen_ids:  # Avoid circular references
            return 0
        seen_ids.add(id(o))

        size = sys.getsizeof(o)
        if isinstance(o, dict):
            size += sum(_sizeof(k) + _sizeof(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set)):
            size += sum(_sizeof(i) for i in o)
        return size

    return _sizeof(obj) / (1024**2)


def recurse_stl_directory_for_duplicates(dirname, pitch, bounds, zrotation=8, threshold=0.75):
    voxel_dict = generate_voxel_grid_of_stl_in_directory(dirname, pitch, bounds, zrotation=zrotation)

    # Size of voxel dict in MB
    size = get_nested_dict_size(voxel_dict)
    print(f"Size of voxel dict: {size} MB which is {size / len(voxel_dict)} MB per mesh")

    # Find duplicates
    find_duplicates_in_dict(voxel_dict, threshold)


if __name__ == "__main__":

    pitch = 1
    bounds = np.array([[-26, -26, 0], [26, 26, 80]])
    zrotation = 8

    dirnames = [
        # Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/texture"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/axial_component"),
        Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet"),
    ]

    for dirname in dirnames:
        recurse_stl_directory_for_duplicates(dirname, pitch, bounds, zrotation, threshold=0.70)

    # Compare meshes

    # fname = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet/sheet_005.stl")
    # mesh = trimesh.load(fname)

    # mesh_voxel = voxelize_given_bounds(mesh, pitch, bounds)

    # return voxel_grid
    # # Filter out the excess regions
    # mask = np.zeros(indices.shape, dtype=bool)
    # mask[:, 0] = np.logical_and(indices[:, 0] >= 0, indices[:, 0] < bounds[1, 0])
    # mask[:, 1] = np.logical_and(indices[:, 1] >= 0, indices[:, 1] < bounds[1, 1])
    # mask[:, 2] = np.logical_and(indices[:, 2] >= 0, indices[:, 2] < bounds[1, 2])
    # mask = np.all(mask, axis=1, keepdims=True)
    # mask_args = np.all(mask, axis=1)

    # valid_indices = indices[mask_args]

    # matrix = np.zeros(np.diff(bounds, axis=0)[0] // pitch, dtype=bool)
    # matrix[valid_indices] = True

    # T = np.eye(4)
    # T[:3, :3] *= pitch
    # T[:3, 3] = bounds[0]

    # voxel_grid = trimesh.voxel.VoxelGrid(matrix, transform=T)

    # min_corner = bounds[0]
    # max_corner = bounds[1]

    # big_dims = np.round((max_corner - min_corner) / pitch).astype(int)

    # mesh1_T = mesh1_voxel.transform.copy()

    # # big_dims = np.maximum(mesh1_sliced_voxel.shape, mesh2_sliced_voxel.shape)
    # big_T = mesh1_sliced_voxel.transform.copy()
    # big_T[:3, 3] = np.min([mesh1_sliced_voxel.transform[:3, 3], mesh2_sliced_voxel.transform[:3, 3]], axis=0)
    # big_matrix = np.ones(big_dims, dtype=bool)

    # big_voxel1 = trimesh.voxel.VoxelGrid(big_matrix, transform=big_T)
    # big_voxel1.matrix[:, :, :] = False

    # big_dims = np.round((max_corner - min_corner) / pitch).astype(int)
    # # big_dims = np.maximum(mesh1_sliced_voxel.shape, mesh2_sliced_voxel.shape)
    # big_T = mesh1_sliced_voxel.transform.copy()
    # big_T[:3, 3] = np.min([mesh1_sliced_voxel.transform[:3, 3], mesh2_sliced_voxel.transform[:3, 3]], axis=0)
    # big_matrix = np.ones(big_dims, dtype=bool)

    # big_voxel1 = trimesh.voxel.VoxelGrid(big_matrix, transform=big_T)
    # big_voxel1.matrix[:, :, :] = False

    # sparse_indices = mesh1_sliced_voxel.sparse_indices
    # xshift = round(-(big_voxel1.transform[0, 3] - mesh1_sliced_voxel.transform[0, 3]) / pitch)
    # yshift = round(-(big_voxel1.transform[1, 3] - mesh1_sliced_voxel.transform[1, 3]) / pitch)
    # zshift = round(-(big_voxel1.transform[2, 3] - mesh1_sliced_voxel.transform[2, 3]) / pitch)
    # big_indices = sparse_indices + np.array([xshift, yshift, zshift])
    # big_voxel1.matrix[big_indices[:, 0], big_indices[:, 1], big_indices[:, 2]] = True

    # for p in mesh1_sliced_voxel.points:
    #     if p not in big_voxel1.points:
    #         raise ValueError("Point not in big_voxel1")
    # assert len(mesh1_sliced_voxel.points) == len(big_voxel1.points) > 0

    # pass

    # dir_name = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/sheet")
    # stl_list = list(dir_name.rglob("*.stl"))
    # stl_list.sort()
    # for i in range(len(stl_list)):
    #     for j in range(i + 1, len(stl_list)):

    #         no_duplicates = True

    #         mesh1 = trimesh.load(stl_list[i])
    #         mesh2 = trimesh.load(stl_list[j])
    #         for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    #             mesh2.apply_transform(trimesh.transformations.rotation_matrix(th, [0, 0, 1]))
    #             are_duplicates, overlap_fraction = test_meshes_are_duplicates(mesh1, mesh2, pitch, 0.75)
    #             if are_duplicates:
    #                 print(
    #                     f"{stl_list[i].name} and {stl_list[j].name} at theta {np.round(th,3)} are duplicates. {overlap_fraction}"
    #                 )

    #                 no_duplicates = False
