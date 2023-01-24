# Test interface joining on curved cylinder
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.utilities import approximate_arc, get_deformation_vertex, get_deformation_points_along_plane
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy

from objects.utilities import approximate_arc


def align_backbone_center(backbone):
    # Rotate to reach T(0.5) == +X axis
    original = np.vstack([backbone.T(0.5), backbone.N(0.5), backbone.B(0.5)])
    goal = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    rot = goal @ np.linalg.inv(original)
    cp = arc_array @ rot
    return Backbone(controlpoints=cp, reparameterize=True)


BACKBONE_LENGTH = 40
CS_SCALE = BACKBONE_LENGTH / 3
c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 8 * 2 * np.pi), s(0 / 8 * 2 * np.pi)],
        [c(1 / 8 * 2 * np.pi), s(1 / 8 * 2 * np.pi)],
        [c(2 / 8 * 2 * np.pi), s(2 / 8 * 2 * np.pi)],
        [c(3 / 8 * 2 * np.pi), s(3 / 8 * 2 * np.pi)],
        [c(4 / 8 * 2 * np.pi), s(4 / 8 * 2 * np.pi)],
        [c(5 / 8 * 2 * np.pi), s(5 / 8 * 2 * np.pi)],
        [c(6 / 8 * 2 * np.pi), s(6 / 8 * 2 * np.pi)],
        [c(7 / 8 * 2 * np.pi), s(7 / 8 * 2 * np.pi)],
    ]
)
cp_round = base_cp.copy()
cp_round *= CS_SCALE


# Varying curvature of constant curvature medial-axis
scale = 0.75
NUM_CS = 10
angle = np.pi / 4
arc_array = approximate_arc(angle, 40)
backbone = Backbone(controlpoints=arc_array, reparameterize=True)


backbone = align_backbone_center(backbone)

cs_list = [CrossSection(cp_round * scale, i) for i in np.linspace(0.05, 0.95, NUM_CS)]
ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
s = Shape([ac], label="cylinder_curve_{}".format(str(round(angle, 2)).replace(".", "-")))


# Add surface deformations
s.label = "NeedsALabel"
t = 1 / 3
ang = 0
magnitude = 4
sigma = 1

pts, normals = get_deformation_vertex(s.mesh, s.ac_list[0], t, N_rotation=ang)


# Flatten points around deformation so that bump will be consistent
# s.flatten_around_vertex(pts, normals, magnitude, sigma)

# Apply deformation
s.apply_gaussian_deformation(pts, normals, magnitude, sigma)

s.create_interface()
s.fuse_mesh_to_interface()
s.mesh_with_interface.show()
