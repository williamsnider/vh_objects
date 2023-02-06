import numpy as np
from objects.utilities import (
    find_cp_for_desired_radius,
    approximate_arc,
    get_deformation_vertex,
    fuse_meshes,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.backbone import Backbone
from objects.shape import Shape
import trimesh
import scipy
from multiprocessing import Pool
import pickle

### Construct projections ###
NUM_CP_PER_CROSS_SECTION = 8
NUM_CP_PER_BACKBONE = 5
LENGTH = 20
RADIUS = 2.5

# Straight
backbone_cp = np.hstack(
    [
        np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
backbone = Backbone(backbone_cp, reparameterize=True)
cp_radius = find_cp_for_desired_radius(RADIUS, NUM_CP_PER_CROSS_SECTION)
cs_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(
    -1, 1
)
cs_cp = np.hstack((cp_radius * np.cos(cs_th), cp_radius * np.sin(cs_th)))

cs_list = [
    CrossSection(controlpoints=cs_cp, position=position)
    for position in np.linspace(0.1, 0.9, 6)
]
p_straight = AxialComponent(backbone, cs_list, smooth_with_post=False)
p_straight.mesh.show(smooth=False)

# Sharp bend
backbone_cp = np.hstack(
    [
        np.linspace(0, LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
shift = backbone_cp - 