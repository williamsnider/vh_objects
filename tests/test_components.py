# Test that the segments listed in objects.components produce correct shapes

from objects.components import segment_flat, segment_slight_curve, segment_sharp_bend, segment_hook
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.parameters import BACKBONE_LENGTH

import numpy as np

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


def test_segments():

    for segment in [segment_hook]:

        cs_list = [CrossSection(base_cp_round * BACKBONE_LENGTH / 6, i) for i in np.linspace(0.1, 0.9, 10)]
        # cs0 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.1)
        # cs1 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.2)
        # cs2 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.3)
        # cs3 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.4)
        # cs4 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.5)
        # cs5 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.6)
        # cs6 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.7)
        # cs7 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.8)
        # cs8 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.9)
        ac = AxialComponent(backbone=segment, cross_sections=cs_list)
        s = Shape([ac])
        s.mesh.show()


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_components.py"])
