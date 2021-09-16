from objects.utilities import distribute_indices
import numpy as np

points1 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [-0.5, 0.5, 0.0],
        [-1.0, 0.0, 0.0],
    ]
)

points2 = np.array(
    [
        [1.0, 0.0, 2.0],
        [0.5, 0.5, 2.0],
        [0.0, 1.0, 2.0],
        [-0.5, 0.5, 2.0],
        [-1.0, 0.0, 2.0],
        [-0.5, -0.5, 2.0],
        [0.0, -1, 2.0],
        [0.5, -0.5, 2.0],
    ]
)


def test_distribute_indices(points1, points2):

    # len(points1) < len(points2)
    len1 = points1.shape[0]
    len2 = points2.shape[0]
    ratio_1_2 = len1 / len2
    edge1 = np.round(ratio_1_2 * np.arange(len2)).astype("int").T
    edge2 = np.arange(len2).T
    expected = np.stack([edge1, edge2], axis=1)
    pairings = distribute_indices(points1, points2)
    assert np.all(expected == pairings)

    # len(points1) < len(points2); order flipped
    expected = np.stack([edge2, edge1], axis=1)
    pairings = distribute_indices(points2, points1)
    assert np.all(expected == pairings)

    # len(points1) == len(points1)
    pts1 = points1
    pts2 = np.roll(points1, 2, axis=0)
    expected = np.stack([np.arange(len1).T, np.roll(np.arange(len1), -2)], axis=1)
    pairings = distribute_indices(pts1, pts2)
    assert np.all(expected == pairings)


if __name__ == "__main__":
    # import pytest

    test_distribute_indices(points1, points2)
    # pytest.main()
