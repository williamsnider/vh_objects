import numpy as np
from scripts.stim_set_common import load_cap
from scipy.spatial.transform import Rotation
from scripts.label_quartet import label_mesh
from pathlib import Path
import trimesh

# Label parameters
font_height = 3.5
LABEL_DEPTH = 1.5
T_label = np.eye(4)
T_label[:3, :3] = Rotation.from_euler("yzx", np.array([np.pi, -np.pi / 2, 0])).as_matrix()
T_label[0, 3] = 9
cap = load_cap()
T_label[2, 3] = cap.bounds[0][2] + LABEL_DEPTH

# Add second label on bottomside with a lesser depth (not sure which is more readable after printing)
T_label_2 = T_label.copy()
T_label_2[0, 3] = -9
T_label_2[2, 3] = cap.bounds[0][2] + LABEL_DEPTH / 2


# Load all meshes
shapes_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl")
stl_list = list(shapes_dir.rglob("*.stl"))

# Check no duplicates file stems
stems = [stl.stem for stl in stl_list]
assert len(stems) == len(set(stems)), "Duplicate file stems found."

# Check labels are in expected format
for stem in stems:
    assert len(stem) == 4
    # Assert first character is letter
    assert stem[0].isalpha()
    # Assert last 3 characters are digits
    assert stem[1:].isdigit()


def process_stl(stl_fname):
    m = trimesh.load_mesh(stl_fname)
    label = stl_fname.stem
    m = label_mesh(m, label, T_label, LABEL_DEPTH, font_height, "difference")
    m = label_mesh(m, label, T_label_2, LABEL_DEPTH, font_height, "difference")
    m.export(stl_fname)
    return m


if __name__ == "__main__":

    # Parallelize
    import tqdm
    import multiprocessing as mp

    with mp.Pool() as pool:
        results = list(tqdm.tqdm(pool.imap(process_stl, stl_list), total=len(stl_list)))

    # for stl_fname in stl_list:
    #     m = process_stl(stl_fname)
    #     m.show()
