import trimesh
from pathlib import Path
import numpy as np
import copy

# APPENDAGE_LENGTH = 25  # mm
SHEET_THICKNESS = 6  # mm
K_PERPENDICULAR = 1 / 50
NUM_CS_PER_SHEET = 11
UU, VV = 50, 50
num_edge_cp = 7
base_round_cp = 3
top_round_cp = 3
cap_path = Path("/home/oconnorlab/Code/vh_objects/assets/cap_20241108.stl")
STL_DIR = Path("/home/oconnorlab/Code/vh_objects/sample_shapes/stl")


def create_scene(mesh_dict_or_list, offset=True):

    if isinstance(mesh_dict_or_list, dict):
        mesh_list = list(mesh_dict_or_list.values())
    elif isinstance(mesh_dict_or_list, list):
        mesh_list = mesh_dict_or_list
    else:
        raise ValueError("mesh_dict_or_list must be a dict or list")

    s = trimesh.Scene()
    x_offset = 0
    for sheet in mesh_list:
        sheet_copy = sheet.copy()
        if offset == True:
            sheet_copy.apply_translation([x_offset, 0, 0])
            x_offset += max(20, (sheet_copy.bounds[1, 0] - sheet_copy.bounds[0, 0]) + 10)
        s.add_geometry(sheet_copy)

    s.show(smooth=False)


def load_cap():

    fname = cap_path
    cap = trimesh.load_mesh(fname)

    # Shift to origin
    centroid = cap.centroid
    cap.apply_translation(-centroid)

    # Rotate about x-axis to point in +z
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    T = np.eye(4)
    T[:3, :3] = R[:3, :3]
    cap.apply_transform(T)

    # Rotate about z-axis so top square is between +x and +y
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    T = np.eye(4)
    T[:3, :3] = R[:3, :3]
    cap.apply_transform(T)

    # Shift so the top is at z=0
    cap.apply_translation([0, 0, -cap.bounds[1, 2]])

    # Shift so ball is centered at Z=0
    cap.apply_translation([0, 0, 4.9])
    return cap


def export_shape(s, save_dir, name):

    # Make directory if needed
    save_dir = Path(save_dir)
    if save_dir.exists() == False:
        save_dir.mkdir(parents=True)

    # Export STL
    s.mesh.export(Path(save_dir, name).with_suffix(".stl"))


def slightly_deform_mesh(mesh_list):
    mesh_list_new = []
    for i in range(len(mesh_list)):
        mesh_copy = copy.deepcopy(mesh_list[i])
        mesh_copy.apply_scale(1 + 0.01 * i)
        mesh_copy.apply_translation(np.array([0.01, 0.01, 0.01]) * i)
        mesh_list_new.append(mesh_copy)
    return mesh_list_new


if __name__ == "__main__":
    cap = load_cap()
    cap.show()
