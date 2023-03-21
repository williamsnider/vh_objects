# Compare if shapes are identical
from pathlib import Path
import pickle
import scripts.calc_overlap
import numpy as np
import copy
import scipy.spatial.transform
import pandas as pd
import trimesh
from multiprocessing import Pool, Queue
import re
import shutil

PITCH = 1

# Read directory
base_dir = "/home/williamsnider/Code/objects/sample_shapes/stimulus_set_B"
pkl_dir = Path(base_dir, "pkl")
pkl_list = list(Path.glob(pkl_dir, "*"))
pkl_list.sort()


def voxelize_shape(vox_args):
    pkl_path, x_axis_rot, i = vox_args

    # Save as pickle file
    filename = pkl_path.stem + "_" + str(i) + ".pkl"
    vox_path = Path(base_dir, "vox", filename)
    if vox_path.exists():
        return

    print(pkl_path)

    # Load shape
    with open(pkl_path, "rb") as f:
        s2 = pickle.load(f)
    mesh2 = s2.mesh

    # Rotate about x-axis
    R = scipy.spatial.transform.Rotation.from_euler("xyz", [x_axis_rot, 0, 0]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    mesh2 = mesh2.apply_transform(T)

    # Voxelize mesh
    vox2 = trimesh.voxel.creation.voxelize(mesh2, PITCH)
    vox2.fill()

    with open(vox_path, "wb") as f:
        pickle.dump(vox2, f)

    return


rotations = np.linspace(0, 2 * np.pi, 4, endpoint=False)

# Voxelize and save as pickle file
vox_dir = Path(base_dir, "vox")
if vox_dir.exists() == False:
    Path.mkdir(vox_dir, parents=True)
vox_args = []
for pkl_path in pkl_list:
    # print(pkl_path)
    for i, x_axis_rot in enumerate(rotations):
        # voxelize_shape(pkl_path, x_axis_rot, i)
        vox_args.append([pkl_path, x_axis_rot, i])

# with Pool() as pool:
#     pool.map(voxelize_shape, vox_args)
def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


# Read directory
vox_list = list(Path.glob(vox_dir, "*"))
vox_list.sort(key=natural_sort_key)

# Construct df to save results
# rows = []
# for v in vox_list:
#     rows.append(v.stem)
rows = [path.stem for path in vox_list if "_0" in path.stem]
rows.sort(key=natural_sort_key)
columns = [path.stem for path in vox_list]
columns.sort(key=natural_sort_key)
df = pd.DataFrame(columns=columns, index=rows)

combs = []
for r in rows:
    for c in columns:
        if r[:-2] == c[:-2]:
            combs.append([r, c])
        else:

            # Test if r comes before c by sorting
            order1 = [r[:-2], c[:-2]]
            order2 = [r[:-2], c[:-2]]
            order2.sort(key=natural_sort_key)
            if order1 == order2:
                combs.append([r, c])


def compare_voxgrids(comb):
    r, c = comb
    vox1_path = Path(vox_dir, r + ".pkl")
    with open(vox1_path, "rb") as f:
        vox1 = pickle.load(f)

    vox2_path = Path(vox_dir, c + ".pkl")
    with open(vox2_path, "rb") as f:
        vox2 = pickle.load(f)

    prop = scripts.calc_overlap.calc_vox_overlap(vox1, vox2)
    return (r, c, prop)
    # q.put([r, c, prop])


for comb in combs[:1]:
    compare_voxgrids(comb)

# import tqdm
# with Pool() as pool:
#     results = []
#     for result in tqdm.tqdm(pool.imap_unordered(compare_voxgrids, combs), total=len(combs)):
#         results.append(result)

# with open("results.pkl", "wb") as f:
#     pickle.dump(results, f)

# Add to dataframe
# for r, c, prop in results:
#     df.loc[r, c] = prop

# Find overlapping shapes
THRESHOLD = 0.99
r, c = np.where(df > THRESHOLD)
redundant = []

redundant_dict = {}
unique_rows = np.unique(r)
for u in unique_rows:
    mask = r == u
    cols = c[mask]
    r_name = df.index[[u]]
    c_name = df.columns[cols]
    if len(r_name) == len(c_name):
        continue
    else:
        assert r_name[0][:-2] == c_name[0][:-2]
        redundant_dict[r_name[0]] = c_name[1:].tolist()

exclusion_set = set()
for k,v in redundant_dict.items():
    v_short = [i[:-2] for i in v]
    exclusion_set.update(v_short)

inclusion_set = set([k[:-2] for k in redundant_dict.keys()])

exclude_list = list(exclusion_set - inclusion_set)
exclude_list.sort(key=natural_sort_key)


# Copy images less exclude
final_dir = Path(base_dir, "png_final")
if final_dir.exists() == False:
    final_dir.mkdir(parents=True)
for image in vox_list:

    stem = image.stem
    trunc = stem[:-2]

    if trunc in exclude_list:
        continue
    
    src = Path(base_dir, "png", trunc+".png")
    dst = Path(final_dir, trunc+".png")
    shutil.copyfile(src, dst)

