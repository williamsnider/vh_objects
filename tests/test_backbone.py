from objects.backbone import Backbone
import numpy as np


def test_reparameterize():

    cp = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0]])
    NUM_CONTROLPOINTS = cp.shape[0]

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
