from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
from objects.deformations import (
    deform_ac,
    filter_plane,
    filter_concave_ellipsoid_1,
    filter_concave_ellipsoid_2,
    filter_concave_cylinder_1,
    filter_concave_cylinder_2,
    filter_hyperboloid_surface_1,
    filter_hyperboloid_surface_2,
    filter_convex_cylinder_1,
    filter_convex_cylinder_2,
    filter_convex_ellipsoid_1,
    filter_convex_ellipsoid_2,
)
import numpy as np
import copy
from scipy.spatial import cKDTree
from pathlib import Path

save_dir = Path(Path.cwd(), "sample_shapes", "stimulus_set")

# Create base controlpoints and axial component
c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 8 * 2 * np.pi), s(0 / 8 * 2 * np.pi)],
        [c(1 / 8 * 2 * np.pi), s(1 / 8 * 2 * np.pi)],
        [c(2 / 8 * 2 * np.pi), s(2 / 8 * 2 * np.pi)],
        [c(3 / 8 * 2 * np.pi), s(3 / 8 * 2 * np.pi)],
        [c(4 / 8 * 2 * np.pi), s(4 / 8 * 2 * np.pi)],
        [c(5 / 8 * 2 * np.pi), s(5 / 8 * 2 * np.pi)],
        [c(6 / 8 * 2 * np.pi), s(6 / 8 * 2 * np.pi)],
        [c(7 / 8 * 2 * np.pi), s(7 / 8 * 2 * np.pi)],
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
base_ac = AxialComponent(100, curvature=1 / 50, cross_sections=[cs1, cs2, cs3, cs4, cs5, cs6, cs7, cs8, cs9])

# Make list of valid, changeable controlpoints
valid_rows = [4, 7, 10]
valid_cols = [1, 3]
valid_cps = [[r, c] for r in valid_rows for c in valid_cols]

for deformation_filter in [
    filter_plane,
    filter_concave_ellipsoid_1,
    filter_concave_ellipsoid_2,
    filter_concave_cylinder_1,
    filter_concave_cylinder_2,
    filter_hyperboloid_surface_1,
    filter_hyperboloid_surface_2,
    filter_convex_cylinder_1,
    filter_convex_cylinder_2,
    filter_convex_ellipsoid_1,
    filter_convex_ellipsoid_2,
]:
    ac = copy.deepcopy(base_ac)
    ac = deform_ac(ac, base_ac, row=5, column=4, deformation_filter=deformation_filter)
    ac = deform_ac(ac, base_ac, row=8, column=3, deformation_filter=deformation_filter.T)

    # Call these methods again to remake axial component with new controlpoints
    ac.make_surface()
    ac.make_mesh()
    s = Shape([ac])
    s.align_mesh()
    s.save_mesh_as_png(save_dir)
    s.mesh.show(smooth=False)
    break
# Shift points according to desired patteern
