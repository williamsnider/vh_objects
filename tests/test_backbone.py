from objects.backbone import Backbone
import numpy as np


def test_backbone_tangent_normal_binormal():
    cp = np.array([[0, 0, 0], [0, 10, 0], [0, 20, 0], [0, 30, 0], [10, 30, 0], [20, 30, 0], [30, 30, 0]])
    backbone = Backbone(cp, reparameterize=True)

    # TODO: Figure out why binormal(0) results in nan. First you need to enable debugging breakpoints in other files.
    t = np.linspace(0, 1, 100)
    T = backbone.T(t)
    N = backbone.N(t)
    B = backbone.B(t)

    assert np.all(T[0] == [0, 1, 0])
    assert np.all(N[0] == [-1, 0, 0])
    assert np.all(B[0] == [0, 0, 1])
    assert np.all(T[-1] == [1, 0, 0])
    assert np.all(N[-1] == [0, 1, 0])
    assert np.all(B[-1] == [0, 0, 1])


def test_reparameterize():
    cp = np.array([[0, 0, 0], [0, 10, 0], [0, 20, 0], [0, 30, 0], [10, 30, 0], [20, 30, 0], [30, 30, 0]])
    backbone = Backbone(cp, reparameterize=False)
    new_backbone = backbone.reparameterize()

    # Test if arc length / total length = t
    NUM_SAMPLES = 100
    t = np.linspace(0, 1, NUM_SAMPLES)
    total_length = new_backbone.length()

    for i, val in enumerate(t):

        dist_ratio = new_backbone.length(0, val) / total_length

        assert np.isclose(
            dist_ratio - val, 0, atol=0.001
        ), "Reparameterized backbone is not arc length parameterized within atol."


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_backbone.py"])
