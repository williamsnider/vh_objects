import numpy as np
import matplotlib.pyplot as plt
from objects.components import (
    cp_round_low,
    cp_round_high,
    cp_concave_high,
    cp_concave_low,
    cp_plane,
    cp_convex_low,
    cp_convex_med,
    cp_convex_high,
    cp_convex_point_low,
    cp_convex_point_med,
    cp_convex_point_high,
)
from objects.parameters import ORDER
from splipy import BSplineBasis, Curve
from pathlib import Path

maxcp = cp_convex_high.max()
save_dir = Path(Path.cwd(), "sample_shapes", "plots")
if save_dir.is_dir() == False:
    Path.mkdir(save_dir)


def make_fig(cp, title):

    # Make curve
    degree = ORDER - 1
    num_cp_per_cross_section = cp.shape[0]
    num_knots = num_cp_per_cross_section + ORDER + degree
    knot = np.linspace(0, 1, num_knots)
    basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
    curve = Curve(basis1, controlpoints=cp, rational=False)

    # Sample curve
    curve.reparam()
    t = np.linspace(0, 1, 1000)
    pts = curve(t)

    # Make figure
    fig, ax = plt.subplots()
    ax.set_title(title, fontdict={"fontsize": 20})
    ax.set_xlim([-maxcp, maxcp])
    ax.set_ylim([-maxcp, maxcp])
    ax.set_aspect("equal")

    # Plot controlpoints
    x, y = cp.T
    ax.plot(x, y, "g.", markersize=20)

    # Plot curve
    x, y = pts.T
    ax.plot(x, y, "b-")

    fname = Path(save_dir, title).with_suffix(".png")
    plt.savefig(fname, bbox_inches="tight")


make_fig(cp_round_high, "Round")
make_fig(cp_round_low, "Round (small)")
make_fig(cp_concave_high, "Concave (high)")
make_fig(cp_concave_low, "Concave (low)")
make_fig(cp_plane, "Plane")
make_fig(cp_convex_low, "Convex (low)")
make_fig(cp_convex_med, "Convex (med)")
make_fig(cp_convex_high, "Convex (high)")
make_fig(cp_convex_point_low, "Convex Point (low)")
make_fig(cp_convex_point_med, "Convex Point (med)")
make_fig(cp_convex_point_high, "Convex Point (high)")
