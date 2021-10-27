from objects.backbone import Backbone
from objects.backbone_from_digits import BackboneFromDigits
import numpy as np
import matplotlib.pyplot as plt


def test_align_segments():
    cp0 = np.stack(
        [
            np.zeros(5),
            np.linspace(0, 1, 5),
            np.zeros(5),
        ]
    ).T  # Transpose so that cp are along rows

    backbone0 = Backbone(cp0, reparameterize=False)
    cp1 = np.stack(
        [
            np.linspace(0, 1, 5),
            np.linspace(0, 1, 5),
            np.zeros(5),
        ]
    ).T  # Transpose so that cp are along rows
    backbone1 = Backbone(cp1, reparameterize=False)
    t = np.linspace(0, np.pi / 2, 5)
    cp2 = np.stack(
        [
            np.sin(t),
            1 - np.cos(t),
            np.zeros(5),
        ]
    ).T  # Transpose so that cp are along rows
    backbone2 = Backbone(cp2, reparameterize=False)
    digit_segments = [backbone0, backbone1, backbone2]
    angles_between_segments = np.array([[0, 0, 0], [0, 0, 0]])
    bfd = BackboneFromDigits(digit_segments=digit_segments, angles_between_segments=angles_between_segments)

    for i, segment in enumerate(bfd.digit_segments):

        if i == 0:
            continue

        prev = bfd.digit_segments[i - 1]
        curr = bfd.digit_segments[i]

        # Ensure cp align
        assert np.all(
            np.isclose(prev.controlpoints[-1], curr.controlpoints[0])
        ), "Last cp of previous segment not aligned with first cp of current segment.0"

        # Ensure TNB aligns
        assert np.all(np.isclose(prev.T(1), curr.T(0))), "Tangent vectors not aligned."
        assert np.all(np.isclose(prev.N(1), curr.N(0))), "Normal vectors not aligned."
        assert np.all(np.isclose(prev.B(1), curr.B(0))), "Binormal vectors not aligned."

    # Create new backbone with the aligned controlpoints
    backbone = Backbone(bfd.controlpoints, reparameterize=True)
    u = np.linspace(0, 1, 100)
    r = backbone.r(u)
    cp = backbone.controlpoints

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    maxcp = cp.max()
    ax.set_xlim([-maxcp, maxcp])
    ax.set_ylim([-maxcp, maxcp])
    ax.set_zlim([-maxcp, maxcp])
    ax.view_init(elev=-90, azim=90)
    ax.plot(r[:, 0], r[:, 1], r[:, 2], "k.")
    plt.show()


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_backbone_from_digits.py"])
