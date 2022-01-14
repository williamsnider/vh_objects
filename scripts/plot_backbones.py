# Test that the segments listed in objects.components produce correct shapes

from objects.components import (
    segment_flat,
    segment_weak_curve,
    segment_strong_curve,
    segment_sharp_bend,
    segment_hook,
    segment_weak_s,
    segment_strong_s,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.parameters import BACKBONE_LENGTH
import matplotlib.pyplot as plt
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

backbone_list = [
    segment_flat,
    segment_weak_curve,
    segment_strong_curve,
    segment_sharp_bend,
    segment_hook,
    segment_weak_s,
    segment_strong_s,
]

# Make 6x3 Figure
fig, axs = plt.subplots(len(backbone_list), 3)
padding = 10
for row, backbone in enumerate(backbone_list):

    # Label top axes with column titles
    if row == 0:
        axs[row, 0].set_title("Spline Control Points")
        axs[row, 1].set_title("Backbone")
        axs[row, 2].set_title("Sample Shape")

    # Label left axes with row titles
    backbone_name = [k for k, v in locals().items() if v is backbone][0]
    backbone_name = backbone_name.replace("segment_", "")
    backbone_name = backbone_name.replace("_", "\n")  # Add newline
    axs[row, 0].set_ylabel(backbone_name, fontsize=15)

    # Plot controlpoints (first column)
    x, y = backbone.controlpoints[:, :2].T
    axs[row, 0].plot(x, y, "g.", markersize=20)
    y_mean = y.mean()
    axs[row, 0].set_ylim([y_mean - BACKBONE_LENGTH / 2 - padding, y_mean + BACKBONE_LENGTH / 2 + padding])
    axs[row, 0].set_aspect("equal")
    axs[row, 0].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )

    # Plot backbone (second column)
    t = np.linspace(0, 1, 100)
    x, y, _ = backbone.r(t).T
    axs[row, 1].plot(x, y, "b-", linewidth=10)
    axs[row, 1].set_ylim([y_mean - BACKBONE_LENGTH / 2 - padding, y_mean + BACKBONE_LENGTH / 2 + padding])
    axs[row, 1].set_aspect("equal")
    axs[row, 1].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )

    # Plot render of shape
    cs_list = [CrossSection(base_cp_round * BACKBONE_LENGTH / 6, i) for i in np.linspace(0.1, 0.9, 10)]
    ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
    s = Shape([ac])
    color = s.save_mesh_as_png(save_dir="dummy", return_img=True)
    color = np.flip(color, axis=1)  # Reverse y axis so that the render aligns with the above plots
    axs[row, 2].imshow(color / 2 ** 16)  # Convert to range (0,1)
    axs[row, 2].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )

plt.show()


# def test_segments():

#     t = np.linspace(0, 1, 100)
#     for segment in [segment_sharp_bend]:

#         maxcp = segment.controlpoints.max() * 1.2
#         # Make figure
#         fig, ax = plt.subplots()
#         # ax.set_title(title, fontdict={"fontsize": 20})
#         ax.set_xlim([-maxcp, maxcp])
#         ax.set_ylim([-maxcp, maxcp])
#         ax.set_aspect("equal")

#         # Plot controlpoints
#         x, y, _ = segment.controlpoints.T
#         ax.plot(x, y, "g.", markersize=20)

#         # Plot backbone
#         x, y, _ = segment.r(t).T
#         ax.plot(x, y, "b-", markersize=20)
#         plt.show()

#         cs_list = [CrossSection(base_cp_round * BACKBONE_LENGTH / 6, i) for i in np.linspace(0.1, 0.9, 10)]
#         # cs0 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.1)
#         # cs1 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.2)
#         # cs2 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.3)
#         # cs3 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.4)
#         # cs4 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.5)
#         # cs5 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.6)
#         # cs6 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.7)
#         # cs7 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.8)
#         # cs8 = CrossSection(base_cp_round * BACKBONE_LENGTH / 6, 0.9)
#         ac = AxialComponent(backbone=segment, cross_sections=cs_list)
#         s = Shape([ac])
#         s.mesh.show()


# test_segments()
