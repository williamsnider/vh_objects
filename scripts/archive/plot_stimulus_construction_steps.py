from objects.components import cp_concave_high, cp_round_high, segment_arc_1_4, segment_flat
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.backbone_from_digits import BackboneFromDigits
from objects.backbone import Backbone
from objects.shape import Shape
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Construct backbone
backbone0 = segment_flat.copy()
backbone1 = segment_arc_1_4.copy()
backbone2 = segment_flat.copy()
backbone3 = segment_arc_1_4.copy()
digit_segments = [backbone0, backbone1, backbone2, backbone3]
angles_between_segments = np.array([[0, 0, 0], [np.pi / 4, np.pi / 4, 0], [0, 0, 0]])
bfd = BackboneFromDigits(digit_segments, angles_between_segments)
backbone = Backbone(bfd.controlpoints, reparameterize=True)

# Construct cross sections
rotation = np.pi / 4
cs0 = CrossSection(cp_concave_high, 0.2, rotation=rotation)
cs1 = CrossSection(cp_round_high, 0.5, rotation=rotation)
cs2 = CrossSection(cp_concave_high, 0.8, rotation=rotation)

# Construct axial component
ac1 = AxialComponent(backbone, cross_sections=[cs0, cs1, cs2])

# Construct shape
s = Shape([ac1], align_OBB=False, fuse_to_interface=False, label="example_shape")
s.mesh.show()
# png_save_dir = Path(Path.cwd(), "sample_shapes", "example_shape")
# s.save_mesh_as_png(png_save_dir)

##########################
# Plot backbone
##########################

# Make figure
# maxcp = backbone.controlpoints.max()
# title = "Backbone"
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# # ax.set_title(title, fontdict={"fontsize": 20})
# ax.set_xlim([-maxcp, maxcp])
# ax.set_ylim([-maxcp, maxcp])
# ax.set_zlim([-maxcp, maxcp])
# ax.view_init(elev=-132, azim=29)

# # Plot controlpoints
# # x, y, z = backbone.controlpoints.T
# # ax.plot(x, y, z, "g.", markersize=20)

# # Plot curve
# t = np.linspace(0, 1, 1000)
# pts = backbone.r(t)
# x, y, z = pts.T
# ax.plot(x, y, z, "k-")

# # Plot surface points
# for pos in [0.2, 0.5, 0.8]:
#     u = np.linspace(ac1.surface.start()[0], ac1.surface.end()[0], 1000)
#     pts = np.squeeze(ac1.surface(u, pos))

#     x, y, z = pts.T
#     ax.plot(x, y, z, "b-")

# plt.show()

# fname = Path(save_dir, title).with_suffix(".png")
# plt.savefig(fname, bbox_inches="tight")
