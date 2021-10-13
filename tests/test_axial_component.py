from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
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

base_cp_round = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
    ]
)


def test_long_AC():
    """Debugging a sharp boundary when length is long"""
    cs0 = CrossSection(base_cp * 20, 0.3)
    cs1 = CrossSection(base_cp * 20, 0.7)
    ac = AxialComponent(100 * np.pi * 1 * 0.25, curvature=1 / 100, cross_sections=[cs0, cs1])


def test_circular_arc():

    cs = CrossSection(base_cp, 0.0)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs])
    t = np.linspace(0, 1, 3)
    backbone = ac.r(t)
    target = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.70710678, 0.29289322, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    assert np.allclose(backbone, target)


def test_tangent_vectors():

    cs = CrossSection(base_cp, 0.0)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs])
    t = np.linspace(0, 1, 5)
    T = ac.T(t)
    N = ac.N(t)
    B = ac.B(t)
    T_target = np.array(
        [
            [1.00000000e00, 0.00000000e00, 0.00000000e00],
            [9.23879533e-01, 3.82683432e-01, 0.00000000e00],
            [7.07106781e-01, 7.07106781e-01, 0.00000000e00],
            [3.82683432e-01, 9.23879533e-01, 0.00000000e00],
            [6.12323400e-17, 1.00000000e00, 0.00000000e00],
        ]
    )
    N_target = np.array(
        [
            [-0.00000000e00, 1.00000000e00, 0.00000000e00],
            [-3.82683432e-01, 9.23879533e-01, 0.00000000e00],
            [-7.07106781e-01, 7.07106781e-01, 0.00000000e00],
            [-9.23879533e-01, 3.82683432e-01, 0.00000000e00],
            [-1.00000000e00, 6.12323400e-17, 0.00000000e00],
        ]
    )
    B_target = np.array(
        [
            [0.0, -0.0, 1.0],
            [0.0, -0.0, 1.0],
            [0.0, -0.0, 1.0],
            [0.0, -0.0, 1.0],
            [0.0, -0.0, 1.0],
        ]
    )
    assert np.allclose(T, T_target)
    assert np.allclose(N, N_target)
    assert np.allclose(B, B_target)


def test_get_controlpoints():

    cs = CrossSection(base_cp, 0.5)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs])
    ac.get_controlpoints()


def test_make_surface():

    cs0 = CrossSection(base_cp * 0.5, 0.3)
    cs1 = CrossSection(base_cp * 0.5, 0.7)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=1 / 1, cross_sections=[cs0, cs1])

    ac.get_controlpoints()
    ac.make_surface()


def test_make_mesh():

    cs0 = CrossSection(base_cp_round * 0.5, 0.3)
    cs1 = CrossSection(base_cp_round * 0.5, 0.7)
    ac = AxialComponent(2 * np.pi * 1 * 0.25, curvature=0, cross_sections=[cs0, cs1])

    ac.get_controlpoints()
    ac.make_surface()
    ac.make_mesh()


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_axial_component.py"])
