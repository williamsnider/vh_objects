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
dict = dict()
items = list(locals().items())
for filt in filter_set:

    for name, value in items:

        if type(value) != type(filt):
            continue

        if np.all(value == filt):
            dict[name] = filt
            break  # Escape the loop since the variable was found
dict["None"] = None
all_pairs = list(itertools.combinations_with_replacement(dict.keys(), 2))

# Make list of all locations (i.e. controlpoints) (combinations without replacement)
valid_cps = [[r, c] for r in valid_rows for c in valid_cols]
all_location_pairs = list(itertools.combinations(valid_cps, 2))

for curvature in curvatures[1:]:

    base_ac = AxialComponent(length, curvature=curvature, cross_sections=[cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8, cs9])

    for location in all_location_pairs:

        row1, col1 = location[0]
        row2, col2 = location[1]

        for filt in all_pairs:

            # Print status
            print("Curvature = ", np.round(curvature, decimals=2), " Location = ", location, " Filter pairs = ", filt)

            # Construct save_dir
            save_dir = Path(
                base_dir,
                "curvature_" + str(np.round(curvature, decimals=3)).replace(".", "p"),
                "location_" + str(row1) + "_" + str(row2),
            )
            # Construct Label
            label = ""
            for r in valid_rows:

                if r == row1:
                    label += filt[0]
                elif r == row2:
                    label += filt[1]
                else:
                    label += "None"

                # Add underscore except for final entry
                if r == valid_rows[-1]:
                    continue
                else:
                    label += "_"

            # Create shape
            filt1 = dict[filt[0]]
            filt2 = dict[filt[1]]
            ac = copy.deepcopy(base_ac)
            ac = deform_ac(ac, base_ac, row=row1, column=col1, deformation_filter=filt1)
            ac = deform_ac(ac, base_ac, row=row2, column=col2, deformation_filter=filt2)
            ac.make_surface()
            ac.make_mesh()
            s = Shape([ac], align_OBB=False, fuse_to_interface=True, label=label)
            s.save_mesh_as_png(save_dir)
            # s.mesh.show(smooth=False)

# Shift points according to desired patteern
