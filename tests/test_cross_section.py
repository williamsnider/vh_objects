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


def test_align_controlpoints():

    # Roll out of alignment
    cp = np.roll(base_cp, 2, axis=0)
    cs = CrossSection(cp, 0.5, rotation=0, tilt=0)

    assert np.allclose(cs.controlpoints[:, :2], base_cp)


if __name__ == "__main__":
    import pytest

    pytest.main(["tests"])
