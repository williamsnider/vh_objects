import numpy as np
import matplotlib.pyplot as plt
from objects.components import (
    cp_round,
    cp_concave_high,
    cp_plane,
    cp_convex,
    cp_elliptical,
    backbone_flat,
)
from objects.parameters import ORDER, BACKBONE_LENGTH
from splipy import BSplineBasis, Curve
from pathlib import Path
from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape

cs_list = [
    cp_concave_high,
    cp_plane,
    cp_round,
    cp_convex,
    cp_elliptical,
]

# Make 6x3 Figure
fig, axs = plt.subplots(len(cs_list), 3)
padding = 10
for row, cs in enumerate(cs_list):

    # Label top axes with column titles
    if row == 0:
        axs[row, 0].set_title("Spline Control Points")
        axs[row, 1].set_title("Cross Section")
        axs[row, 2].set_title("Sample Shape")

    # Label left axes with row titles
    cs_name = [k for k, v in locals().items() if v is cs][0]
    cs_name = cs_name.replace("cp_", "")
    axs[row, 0].set_ylabel(cs_name, fontsize=15)

    # Plot controlpoints (first column)
    x, y = cs[:, :2].T
    axs[row, 0].plot(x, y, "g.", markersize=20)
    axs[row, 0].plot(x, y, "g-", markersize=20)
    axs[row, 0].plot(x[[-1, 0]], y[[-1, 0]], "g-", markersize=20)
    y_mean = y.mean()
    axs[row, 0].set_ylim([-40, 40])
    axs[row, 0].set_xlim([-40, 40])
    axs[row, 0].set_aspect("equal")
    axs[row, 0].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )

    # # Plot Cross Section (second column)

    # def plot_comparison_curve(comp):
    #     degree = ORDER - 1
    #     num_cp_per_cross_section = comp.shape[0]
    #     num_knots = num_cp_per_cross_section + ORDER + degree
    #     knot = np.linspace(0, 1, num_knots)
    #     basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
    #     curve = Curve(basis1, controlpoints=comp, rational=False)

    #     # Sample curve
    #     curve.reparam()
    #     t = np.linspace(0, 1, 1000)
    #     x, y = curve(t).T
    #     axs[row, 1].plot(x, y, "r-", linewidth=10)

    # # Plot comparison curve
    # if cs is cp_concave_high:
    #     comp = cp_convex
    #     plot_comparison_curve(comp)
    # elif cs is cp_concave_low:
    #     comp = cp_round_high
    #     plot_comparison_curve(comp)

    # elif cs is cp_round_high:
    #     comp = cp_concave_low
    #     plot_comparison_curve(comp)

    # elif cs is cp_convex:
    #     comp = cp_concave_high
    #     plot_comparison_curve(comp)

    # Make curve
    degree = ORDER - 1
    num_cp_per_cross_section = cs.shape[0]
    num_knots = num_cp_per_cross_section + ORDER + degree
    knot = np.linspace(0, 1, num_knots)
    basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
    curve = Curve(basis1, controlpoints=cs, rational=False)

    # Sample curve
    curve.reparam()
    t = np.linspace(0, 1, 1000)
    x, y = curve(t).T
    axs[row, 1].plot(x, y, "b-", linewidth=10)
    axs[row, 1].set_ylim([-40, 40])
    axs[row, 1].set_xlim([-40, 40])
    axs[row, 1].set_aspect("equal")
    axs[row, 1].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )

    # Plot render of shape
    cs_for_shape = [CrossSection(cs, i, rotation=np.pi / 2) for i in np.linspace(0.1, 0.9, 10)]
    ac = AxialComponent(backbone=backbone_flat, cross_sections=cs_for_shape)
    s = Shape([ac])
    color = s.save_mesh_as_png(save_dir="dummy", return_img=True)
    color = np.flip(color, axis=1)  # Reverse y axis so that the render aligns with the above plots
    axs[row, 2].imshow(color / 2 ** 16)  # Convert to range (0,1)
    axs[row, 2].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )

plt.show()


# def make_fig(cp, title):

#     # Make curve
#     degree = ORDER - 1
#     num_cp_per_cross_section = cp.shape[0]
#     num_knots = num_cp_per_cross_section + ORDER + degree
#     knot = np.linspace(0, 1, num_knots)
#     basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
#     curve = Curve(basis1, controlpoints=cp, rational=False)

#     # Sample curve
#     curve.reparam()
#     t = np.linspace(0, 1, 1000)
#     pts = curve(t)

#     # Make figure
#     fig, ax = plt.subplots()
#     ax.set_title(title, fontdict={"fontsize": 20})
#     ax.set_xlim([-maxcp, maxcp])
#     ax.set_ylim([-maxcp, maxcp])
#     ax.set_aspect("equal")

#     # Plot controlpoints
#     x, y = cp.T
#     ax.plot(x, y, "g.", markersize=20)

#     # Plot curve
#     x, y = pts.T
#     ax.plot(x, y, "b-")

#     fname = Path(save_dir, title).with_suffix(".png")
#     # plt.savefig(fname, bbox_inches="tight")
#     plt.show()


# make_fig(cp_round_high, "Round")
# make_fig(cp_round_low, "Round (small)")
# make_fig(cp_concave_high, "Concave (high)")
# make_fig(cp_concave_low, "Concave (low)")
# make_fig(cp_plane, "Plane")
# make_fig(cp_convex_low, "Convex (low)")
# make_fig(cp_convex_med, "Convex (med)")
# make_fig(cp_convex_high, "Convex (high)")
# make_fig(cp_convex_point_low, "Convex Point (low)")
# make_fig(cp_convex_point_med, "Convex Point (med)")
# make_fig(cp_convex_point_high, "Convex Point (high)")
