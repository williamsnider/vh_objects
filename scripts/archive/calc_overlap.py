# Compare two meshes with marching cubes

import trimesh
import trimesh.voxel.creation
import numpy as np

# Find corner
def populate_mask(base_mask, bounds, vox):
    r, c, d = (-(bounds[0, :] - vox.bounds) / vox.pitch)[0].astype("int")

    mask = base_mask.copy()
    mask[
        r : r + vox.shape[0],
        c : c + vox.shape[1],
        d : d + vox.shape[2],
    ] = vox.matrix
    return mask


def find_voxelgrid_overlap(vox1, vox2):
    """Computes the overlap between two voxelgrids"""
    # Move to align origins
    # vox2.apply_translation(vox1.origin - vox2.origin)
    pitch = vox1.pitch

    # Find bounds that cover bounds of vox1 and vox2
    combined_bounds = np.dstack([vox1.bounds, vox2.bounds])
    bounds = np.zeros(vox1.bounds.shape)
    bounds[0, :] = combined_bounds.min(axis=2)[0]
    bounds[1, :] = combined_bounds.max(axis=2)[1]

    # Create mask to transfer vox1 and vox2 matrices into
    base_mask = np.zeros((bounds[1] - bounds[0] / pitch).astype("int"), dtype="bool")

    # Populate mask with voxels
    mask1 = populate_mask(base_mask, bounds, vox1)
    mask2 = populate_mask(base_mask, bounds, vox2)

    overlap = mask1 * mask2

    # Transform matrix to align [0,0,0] voxel to its x,y,z coordinate
    T = np.eye(4)
    T[:3, 3] = bounds[0] + pitch / 2  # bounds refer to edge of voxel, so add half of pitch
    vox3 = trimesh.voxel.VoxelGrid(trimesh.voxel.encoding.DenseEncoding(overlap), transform=T)
    return vox3


# Proportion overlap
def calc_overlap_proportion(vox1, vox2, vox_overlap):
    return 2 * vox_overlap.filled_count / (vox1.filled_count + vox2.filled_count)


def calc_vox_overlap(vox1, vox2):
    vox_overlap = find_voxelgrid_overlap(vox1, vox2)
    return calc_overlap_proportion(vox1, vox2, vox_overlap)


def calc_mesh_overlap(mesh1, mesh2, PITCH):

    vox1 = trimesh.voxel.creation.voxelize(mesh1, PITCH)
    vox1.fill()
    vox2 = trimesh.voxel.creation.voxelize(mesh2, PITCH)
    vox2.fill()

    vox_overlap = find_voxelgrid_overlap(vox1, vox2)

    return calc_overlap_proportion(vox1, vox2, vox_overlap)


PITCH = 1

if __name__ == "__main__":

    # Load meshes
    mesh1 = trimesh.load_mesh("/home/williamsnider/Code/objects/test.stl")
    vox1 = trimesh.voxel.creation.voxelize(mesh1, PITCH)
    vox1.fill()

    mesh2 = trimesh.creation.icosphere(4, 20)
    mesh2 = mesh2.apply_translation([1, 3, 5])
    vox2 = trimesh.voxel.creation.voxelize(mesh2, PITCH)
    vox2.fill()

    vox3 = find_voxelgrid_overlap(vox1, vox2)
    frac = calc_overlap_proportion(vox1, vox2, vox3)
    print(frac)

    # Plot overlaps
    scene = trimesh.Scene()

    vox1_mesh = vox1.as_boxes()
    vox1_mesh.visual.face_colors = [255 * 0 / 3, 255 * (2 - 0) / 3, 255, (0 + 1) / 3 * 255]
    scene.add_geometry(vox1_mesh)

    vox2_mesh = vox2.as_boxes()
    vox2_mesh.visual.face_colors = [255 * 1 / 3, 255 * (2 - 1) / 3, 255, (0 + 1) / 3 * 255]
    scene.add_geometry(vox2_mesh)

    vox3_mesh = vox3.as_boxes()
    vox3_mesh.visual.face_colors = [255 * 2 / 3, 255 * (2 - 2) / 3, 255, 255]
    scene.add_geometry(vox3_mesh)

    scene.show()
