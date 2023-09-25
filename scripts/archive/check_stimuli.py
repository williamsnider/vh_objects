from pathlib import Path
from scripts.stimulus_set_params import SAVE_DIR
import trimesh
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

EXTENTS = np.array([120, 45, 70])
bounds = np.array(
    [
        [
            -35,
            -45 / 2,
            -71 / 2,
        ],
        [
            75,
            45 / 2,
            71 / 2,
        ],
    ]
)

stl_list = list(Path(SAVE_DIR).rglob("*.stl"))
stl_list.sort()


def test_mesh(fname):
    # fname = Path(
    #     "/home/oconnorlab/code/objects/sample_shapes/stimulus_set_D/stl/D014.stl"
    # )
    mesh = trimesh.load(fname)

    # Check within bounds
    if ~np.all(mesh.bounds[0] > bounds[0]):
        print("Failure for ", fname.stem)
    elif ~np.all(mesh.bounds[1] < bounds[1]):
        print("Failure for ", fname.stem)


if __name__ == "__main__":

    with Pool() as pool:
        mapped_values = list(
            tqdm(pool.imap_unordered(test_mesh, stl_list), total=len(stl_list))
        )
