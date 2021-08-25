from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
import numpy as np


c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [0.1, 0.1],
    ]
)


def test_fuse_meshes():

    cs0 = CrossSection(base_cp * 0.5, 0.3)
    cs1 = CrossSection(base_cp * 0.5, 0.7)
    ac1 = AxialComponent(2 * np.pi * 1 * 0.25, curvature=0, cross_sections=[cs0, cs1])
    ac2 = AxialComponent(
        2 * np.pi * 1 * 0.25,
        curvature=1 / 1,
        cross_sections=[cs0, cs1],
        parent_axial_component=ac1,
        position_along_parent=0.75,
        position_along_self=0.0,
        euler_angles=np.array([0, np.pi / 3, 0]),
    )
    s = Shape([ac1, ac2])
    s.fuse_meshes(ac1, ac2)
    # s.merged_meshes.show()


if __name__ == "__main__":
    import pytest

    pytest.main()
