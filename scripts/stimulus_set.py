from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
from objects.deformations import (
    deform_ac,
    plane,
    concave_ellipsoid,
    concave_cylinder_vert,
    concave_cylinder_diag_down,
    concave_cylinder_diag_up,
    concave_cylinder_hori,
    hyperboloid_surface_vert,
    hyperboloid_surface_diag_down,
    hyperboloid_surface_diag_up,
    hyperboloid_surface_hori,
    convex_cylinder_vert,
    convex_cylinder_diag_down,
    convex_cylinder_diag_up,
    convex_cylinder_hori,
    convex_ellipsoid,
)
import numpy as np
import copy
from scipy.spatial import cKDTree
from pathlib import Path
import itertools
from joblib import Parallel, delayed
import time


base_dir = Path(Path.cwd(), "sample_shapes", "stimulus_set", "iteration0")
valid_rows = [4, 7, 10]
valid_cols = [5]
MAX_FILTERS_PER_SHAPE = 2
length = 100
curvatures = [0, 1 / 6 * 2 * np.pi / length, 2 / 6 * 2 * np.pi / length]  # 1/6 turn, 1/3 turn
# Create base controlpoints and axial component
# To have 2 bottom/top sections viewable from 1 side, we need 3 cp per section, so 12 total

c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 7 * 2 * np.pi), s(0 / 7 * 2 * np.pi)],
        [c(1 / 7 * 2 * np.pi), s(1 / 7 * 2 * np.pi)],
        [c(2 / 7 * 2 * np.pi), s(2 / 7 * 2 * np.pi)],
        [c(3 / 7 * 2 * np.pi), s(3 / 7 * 2 * np.pi)],
        [c(4 / 7 * 2 * np.pi), s(4 / 7 * 2 * np.pi)],
        [c(5 / 7 * 2 * np.pi), s(5 / 7 * 2 * np.pi)],
        [c(6 / 7 * 2 * np.pi), s(6 / 7 * 2 * np.pi)],
    ]
)
BASE_SCALE = 30

# To have 3 left-to-right intervals, we need 3 cross sections per interval, so 9 total
cs1 = CrossSection(base_cp * BASE_SCALE, 0.1)
cs2 = CrossSection(base_cp * BASE_SCALE, 0.2)
cs3 = CrossSection(base_cp * BASE_SCALE, 0.3)
cs4 = CrossSection(base_cp * BASE_SCALE, 0.4)
cs5 = CrossSection(base_cp * BASE_SCALE, 0.5)
cs6 = CrossSection(base_cp * BASE_SCALE, 0.6)
cs7 = CrossSection(base_cp * BASE_SCALE, 0.7)
cs8 = CrossSection(base_cp * BASE_SCALE, 0.8)
cs9 = CrossSection(base_cp * BASE_SCALE, 0.9)


# for deformation_filter in [
#     plane,
#     concave_ellipsoid,
#     concave_cylinder_vert,
#     concave_cylinder_diag_down,
#     concave_cylinder_diag_up,
#     concave_cylinder_hori,
#     hyperboloid_surface_vert,
#     hyperboloid_surface_diag_down,
#     hyperboloid_surface_diag_up,
#     hyperboloid_surface_hori,
#     convex_cylinder_vert,
#     convex_cylinder_diag_down,
#     convex_cylinder_diag_up,
#     convex_cylinder_hori,
#     convex_ellipsoid,
# ]:

filter_set = [
    plane,
    concave_ellipsoid,
    concave_cylinder_vert,
    concave_cylinder_hori,
    hyperboloid_surface_vert,
    hyperboloid_surface_hori,
    convex_cylinder_vert,
    convex_cylinder_hori,
    convex_ellipsoid,
]

# Make list of all filter pairs (combinations with replacement)
filter_dict = dict()
items = list(locals().items())
for filt in filter_set:

    for name, value in items:

        if type(value) != type(filt):
            continue

        if np.all(value == filt):
            filter_dict[name] = filt
            break  # Escape the loop since the variable was found
filter_dict["None"] = None

all_pairs = []
for d1 in filter_dict.keys():
    for d2 in filter_dict.keys():
        for d3 in filter_dict.keys():

            combination = [d1, d2, d3]

            # Require >=1 intervals to be empty
            if "None" not in combination:
                continue

            all_pairs.append(combination)

# Assemble list of arguments
argument_list = []
count = 0
for curvature in curvatures:

    base_ac = AxialComponent(length, curvature=curvature, cross_sections=[cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8, cs9])

    for deformations in all_pairs:

        # Construct save_dir
        png_save_dir = Path(
            base_dir,
            "png",
            "curvature_" + str(np.round(curvature, decimals=3)).replace(".", "p"),
        )

        stl_save_dir = Path(
            base_dir,
            "stl",
            "curvature_" + str(np.round(curvature, decimals=3)).replace(".", "p"),
        )

        # Construct Label
        label = "_"
        label = label.join(deformations[::-1])  # Flip order to match png

        # Append to argument list
        argument_list.append([base_ac, deformations, label, png_save_dir, stl_save_dir, count])
        count += 1


def carry_out_deformations(base_ac, deformations, label, png_save_dir, stl_save_dir, count):

    print(count)
    ac = copy.deepcopy(base_ac)

    for i, d in enumerate(deformations):

        row = valid_rows[i]
        col = valid_cols[0]
        deformation_filter = filter_dict[d]
        ac = deform_ac(ac, base_ac, row=row, column=col, deformation_filter=deformation_filter)

    ac.make_surface()
    ac.make_mesh()
    s = Shape([ac], align_OBB=False, fuse_to_interface=True, label=label)
    s.save_mesh_as_png(png_save_dir)
    s.export_stl(stl_save_dir)
    # s.mesh.show(smooth=False)


start = time.time()
Parallel(n_jobs=15)(delayed(carry_out_deformations)(*args) for args in argument_list)
end = time.time()
print("Execution time: ", end - start)
# Shift points according to desired patteern
