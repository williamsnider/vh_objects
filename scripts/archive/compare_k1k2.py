# Compare k1andk2 for stimulus set
from pathlib import Path
import re
import pickle
import numpy as np
from analysis.calc_curvature import calc_mesh_curvature, plot_curvature


def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


# Get list of images
base_dir = "/home/williamsnider/Code/objects/sample_shapes/stimulus_set_B"
png_dir = Path(base_dir, "png_final")
png_list = [str(i) for i in list(png_dir.glob("*"))]
png_list.sort(key=natural_sort_key)
png_list = [Path(p) for p in png_list]

# Convert to list of shapes
shape_list = [Path(base_dir, "pkl", fname.stem + ".pkl") for fname in png_list]

# Load shape
idx = 0
fname = shape_list[0]
with open(fname, "rb") as f:
    s = pickle.load(f)

# Analyze curvature
COUNT = 1000
DISTANCE = 2


def process_curvature(args):
    fname = args
    if "1614" in str(fname):
        return fname, None, None, None, None

    with open(fname, "rb") as f:
        s = pickle.load(f)

    mesh = s.mesh
    indices = np.random.choice(np.arange(mesh.vertices.shape[0]), COUNT, replace=False)

    try:
        k1, k2, k1_vec, k2_vec = calc_mesh_curvature(mesh, indices, DISTANCE)
    except Exception as e:
        print(fname)
        print(e)
        return fname, None, None, None, None
    return fname, k1, k2, k1_vec, k2_vec


import tqdm
from multiprocessing import Pool

with Pool() as pool:
    results = []
    for result in tqdm.tqdm(pool.imap_unordered(process_curvature, shape_list), total=len(shape_list)):
        results.append(result)

# k1, k2, k1_vec, k2_vec = calc_mesh_curvature(mesh, indices, DISTANCE)
k1_arr = np.array([res[1] for res in results if type(res[1]) != type(None)])
k2_arr = np.array([res[2] for res in results if type(res[2]) != type(None)])

k1 = k1_arr.ravel()
k2 = k2_arr.ravel()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.random import multivariate_normal

THRES = 1
k1_clamp = k1.copy()
k1_clamp[k1 < -THRES] = -THRES
k1_clamp[k1 > THRES] = THRES
k2_clamp = k2.copy()
k2_clamp[k2 < -THRES] = -THRES
k2_clamp[k2 > THRES] = THRES
fig, axes = plt.subplots(nrows=1, ncols=1)

axes.set_title("K1 vs K2")
h = axes.hist2d(k1_clamp, k2_clamp, bins=10, norm=mcolors.PowerNorm(0.2))
axes.set_xlabel("K1 (Max Curv)")
axes.set_ylabel("K2 (Min Curv)")


plt.show()
# Store curvatures


# plot_curvature(mesh, indices, k1_vec)
