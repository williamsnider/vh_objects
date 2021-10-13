from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
import numpy as np
from pathlib import Path

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
    s.fuse_meshes(ac1.mesh, ac2.mesh)


def test_align_mesh():

    cs0 = CrossSection(base_cp * 20, 0.3)
    cs1 = CrossSection(base_cp * 20, 0.7)
    ac1 = AxialComponent(100 * np.pi * 1 * 0.25, curvature=1 / 100, cross_sections=[cs0, cs1])
    # ac2 = AxialComponent(
    #     40 * np.pi * 1 * 0.25,
    #     curvature=1 / 20,
    #     cross_sections=[cs0, cs1],
    #     parent_axial_component=ac1,
    #     position_along_parent=0.75,
    #     position_along_self=0.0,
    #     euler_angles=np.array([0, np.pi / 3, 0]),
    # )
    s = Shape([ac1])
    s.mesh = ac1.mesh
    s.align_mesh()
    s.fuse_mesh_to_interface()
    # s.merged_meshes.show()


def test_export_stl():

    save_dir = Path(Path.cwd(), "sample_shapes")
    cs0 = CrossSection(base_cp * 20, 0.3)
    cs1 = CrossSection(base_cp * 20, 0.7)
    ac1 = AxialComponent(100 * np.pi * 1 * 0.25, curvature=1 / 100, cross_sections=[cs0, cs1])
    # ac2 = AxialComponent(
    #     40 * np.pi * 1 * 0.25,
    #     curvature=1 / 20,
    #     cross_sections=[cs0, cs1],
    #     parent_axial_component=ac1,
    #     position_along_parent=0.75,
    #     position_along_self=0.0,
    #     euler_angles=np.array([0, np.pi / 3, 0]),
    # )
    s = Shape([ac1])
    s.mesh = ac1.mesh
    s.align_mesh()
    s.fuse_mesh_to_interface()
    s.export_stl(save_dir)


def test_export_png():

    save_dir = Path(Path.cwd(), "sample_shapes")
    cs0 = CrossSection(base_cp * 20, 0.3)
    cs1 = CrossSection(base_cp * 20, 0.7)
    ac1 = AxialComponent(100 * np.pi * 1 * 0.25, curvature=1 / 100, cross_sections=[cs0, cs1])
    # ac2 = AxialComponent(
    #     40 * np.pi * 1 * 0.25,
    #     curvature=1 / 20,
    #     cross_sections=[cs0, cs1],
    #     parent_axial_component=ac1,
    #     position_along_parent=0.75,
    #     position_along_self=0.0,
    #     euler_angles=np.array([0, np.pi / 3, 0]),
    # )
    s = Shape([ac1])
    s.mesh = ac1.mesh
    s.align_mesh()
    s.fuse_mesh_to_interface()
    s.export_png(save_dir)


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_shape.py"])
