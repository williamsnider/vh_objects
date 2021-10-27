from objects.backbone import Backbone
from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
import numpy as np

# Create base backbone
cp = np.array(
    [
        [0, 0, 0],
        [0, 10, 0],
        [0, 20, 0],
        [0, 30, 0],
        [10, 30, 0],
        [20, 30, 0],
        [30, 30, 0],
    ]
)
backbone1 = Backbone(cp, reparameterize=False)

# Create base cross section
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
    ac = AxialComponent(backbone=backbone1, cross_sections=[cs0, cs1])
    s = Shape([ac])


def test_connect_axial_components():

    position_along_parent = 0.75
    position_along_self = 0.25

    cs0 = CrossSection(base_cp * 20, 0.3)
    cs1 = CrossSection(base_cp * 20, 0.7)
    ac1 = AxialComponent(backbone=backbone1, cross_sections=[cs0, cs1])
    ac2 = AxialComponent(
        backbone=backbone1,
        cross_sections=[cs0, cs1],
        parent_axial_component=ac1,
        position_along_parent=position_along_parent,
        position_along_self=position_along_self,
    )

    # Test that join coordinate is correct
    assert np.all(np.isclose(ac1.r(position_along_parent), ac2.r(position_along_self)))

    # Test that tangent at join point is correct (since euler angles are (0,0,0))
    assert np.all(np.isclose(ac1.T(position_along_parent), ac2.T(position_along_self)))

    # Test that normal at join point is correct (since euler angles are (0,0,0))
    assert np.all(np.isclose(ac1.N(position_along_parent), ac2.N(position_along_self)))

    # Test that binormal at join point is correct (since euler angles are (0,0,0))
    assert np.all(np.isclose(ac1.B(position_along_parent), ac2.B(position_along_self)))


def test_get_controlpoints():
    cs0 = CrossSection(base_cp * 20, 0.3)
    cs1 = CrossSection(base_cp * 20, 0.7)
    ac = AxialComponent(backbone=backbone1, cross_sections=[cs0, cs1])
    ac.get_controlpoints()


def test_make_surface():

    cs0 = CrossSection(base_cp * 20, 0.3)
    cs1 = CrossSection(base_cp * 20, 0.7)
    ac = AxialComponent(backbone=backbone1, cross_sections=[cs0, cs1])

    ac.get_controlpoints()
    ac.make_surface()


def test_make_mesh():

    cs0 = CrossSection(base_cp * 20, 0.3)
    cs1 = CrossSection(base_cp * 20, 0.7)
    ac = AxialComponent(backbone=backbone1, cross_sections=[cs0, cs1])

    ac.get_controlpoints()
    ac.make_surface()
    ac.make_mesh()


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_axial_component.py"])
