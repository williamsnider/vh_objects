# Code to add a label to the quartet before printing

import trimesh
from scipy.spatial.transform import Rotation
import numpy as np

# from vh_objects.utilities import calc_mesh_boolean_and_edges
from vh_objects.interface import text_to_mesh
from pathlib import Path

root_dir = Path(__file__).parents[1]
quartet_path = Path(root_dir, "assets/cartridge_20250114.stl")
LABEL_DEPTH = 1.5
font_height = 8


def label_mesh(base_mesh, label, T, label_depth, font_height, operation="difference"):

    label_as_meshes = text_to_mesh(label, label_depth, font_height)
    base_mesh = apply_label(base_mesh, label_as_meshes, T, operation)

    return base_mesh


def apply_label(base_mesh, label_as_meshes, T, operation="difference"):

    label_as_meshes = [mesh.copy().apply_transform(T) for mesh in label_as_meshes]

    meshes_to_combine = [base_mesh.copy()]
    meshes_to_combine.extend(label_as_meshes)
    base_mesh = trimesh.boolean.difference(meshes_to_combine)
    # for label_part in label_as_meshes:
    #     base_mesh = trimesh.boolean.difference([base_mesh, label_part])
    #     # base_mesh, _ = calc_mesh_boolean_and_edges(base_mesh, label_part, operation=operation)

    return base_mesh


def label_quartet(stl_path, label, label_depth, font_height):

    # Load quartet
    quartet = trimesh.load_mesh(stl_path)

    # Create label
    label_as_meshes = text_to_mesh(label, LABEL_DEPTH, font_height)

    # Back of interface
    T1 = np.eye(4)
    T1[:3, :3] = Rotation.from_euler("xyz", np.array([0, np.pi / 2, -np.pi / 2])).as_matrix()
    T1[1, 3] = -17.5 + LABEL_DEPTH
    T1[2, 3] = 18

    # Left top of quartet
    T2 = np.eye(4)
    T2[0, 3] = -75
    T2[1, 3] = -1.5
    T2[2, 3] = 45 / 2 - LABEL_DEPTH

    T3 = T2.copy()
    T3[0, 3] *= -1

    T180 = np.eye(4)
    T180[:3, :3] = Rotation.from_euler("xyz", np.array([0, np.pi, 0])).as_matrix()

    T4 = T2.copy()
    T4 = T180 @ T4

    T5 = T3.copy()
    T5 = T180 @ T5

    for T in [T1, T2, T3, T4, T5]:
        quartet = apply_label(quartet, label_as_meshes, T)

    # Rotate quartet to be flat in XY plane, with up-square in +X direction
    T = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]) @ trimesh.transformations.rotation_matrix(
        np.pi / 2, [1, 0, 0]
    )
    T[2, 3] -= 60  # Shift down so quartet's xy plane aligns with origin of shapes (end of cap)
    quartet.apply_transform(T)

    return quartet


from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def process_quartet(i):
    print(i)
    label = "Q" + str(i).zfill(3)
    quartet = label_quartet(quartet_path, label, LABEL_DEPTH, font_height)

    # Export
    export_path = Path(root_dir, f"sample_shapes/stl/quartet/{label}.stl")
    if not export_path.parent.exists():
        export_path.parent.mkdir(parents=True)
    quartet.export(export_path)


if __name__ == "__main__":

    num_processes = 8  # You can set this to the number of CPU cores or desired parallelism level
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(process_quartet, range(175))


# if __name__ == "__main__":

#     for i in range(150):
#         print(i)
#         label = "Q" + str(i).zfill(3)
#         quartet = label_quartet(quartet_path, label, LABEL_DEPTH, font_height)

#         # Export
#         export_path = Path(root_dir, f"sample_shapes/stl/quartet/{label}.stl")
#         if export_path.parent.exists() == False:
#             export_path.parent.mkdir(parents=True)
#         quartet.export(export_path)

#     # Run above in parallel
#     import multiprocessing as m
# scene = trimesh.Scene()
# scene.add_geometry(label_as_meshes)
# scene.add_geometry(quartet)
# scene.show()


# def load_cap():

#     fname = cap_path
#     cap = trimesh.load_mesh(fname)

#     # Shift to origin
#     centroid = cap.centroid
#     cap.apply_translation(-centroid)

#     # Rotate about x-axis to point in +z
#     R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
#     T = np.eye(4)
#     T[:3, :3] = R[:3, :3]
#     cap.apply_transform(T)

#     # Rotate about z-axis so top square is between +x and +y
#     R = trimesh.transformations.rotation_matrix(3 * np.pi / 4, [0, 0, 1])
#     T = np.eye(4)
#     T[:3, :3] = R[:3, :3]
#     cap.apply_transform(T)

#     # Shift so the top is at z=0
#     cap.apply_translation([0, 0, -cap.bounds[1, 2]])
#     return cap
