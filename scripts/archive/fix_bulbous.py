import numpy as np
from objects.utilities import (
    find_cp_for_desired_radius,
    approximate_arc,
    get_deformation_vertex,
    fuse_meshes,
    fair_mesh,
    angle_between,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.backbone import Backbone
from objects.shape import Shape
import trimesh
import scipy
import scipy.spatial
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import copy
import itertools

#########################
### Common parameters ###
#########################
NUM_CP_PER_CROSS_SECTION = 8
NUM_CP_PER_BACKBONE = 5
NUM_CP_PER_BACKBONE_3SEG = 7
NUM_CROSS_SECTION_2SEG = 20
NUM_CROSS_SECTION_3SEG = int(round(NUM_CROSS_SECTION_2SEG * 1.5))
SEGMENT_LENGTH = 20
RADIUS = 2

#################
### Backbones ###
#################

b_3seg_bend90_cp = np.hstack(
    [
        np.linspace(0, 3 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE_3SEG).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE_3SEG, 1)),
        np.zeros((NUM_CP_PER_BACKBONE_3SEG, 1)),
    ]
)
shift = b_3seg_bend90_cp - b_3seg_bend90_cp[NUM_CP_PER_BACKBONE_3SEG - 3, :]
R = scipy.spatial.transform.Rotation.from_euler("xyz", np.array([0, 0, -2 * np.pi / 4])).as_matrix()
shift_R = shift @ R
transformed = shift_R + b_3seg_bend90_cp[NUM_CP_PER_BACKBONE_3SEG - 3, :]
b_3seg_bend90_cp[NUM_CP_PER_BACKBONE_3SEG - 3 :, :] = transformed[NUM_CP_PER_BACKBONE_3SEG - 3 :, :]
b_3seg_bend90 = Backbone(b_3seg_bend90_cp, reparameterize=True)

######################
### Cross sections ###
######################
cp_2 = find_cp_for_desired_radius(2, NUM_CP_PER_CROSS_SECTION)
# cp_5 = find_cp_for_desired_radius(5, NUM_CP_PER_CROSS_SECTION)
# cp_0 = 0

cs_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
cs_2 = np.hstack((cp_2 * np.cos(cs_th), cp_2 * np.sin(cs_th)))
# cs_5 = np.hstack((cp_5 * np.cos(cs_th), cp_5 * np.sin(cs_th)))
# cs_0 = np.hstack((cp_0 * np.cos(cs_th), cp_0 * np.sin(cs_th)))

# Football
scale = np.geomspace(1, 3, NUM_CROSS_SECTION_2SEG // 2)
scale = np.concatenate([scale, scale[-1::-1]])
scale = np.concatenate([np.ones(NUM_CROSS_SECTION_3SEG - NUM_CROSS_SECTION_2SEG), scale])
position = np.linspace(0.0, 0.95, NUM_CROSS_SECTION_3SEG)
football_3seg = [
    CrossSection(controlpoints=scale[i] * cs_2, position=position[i]) for i in range(NUM_CROSS_SECTION_3SEG)
]


cs_3seg_dict = {
    "football": football_3seg,
}

# cs_list = [cylinder]  # , point, football, hourglass]
# one_segment_backbone_list = [b_straight1, b_arc1]
two_segment_backbone_dict = {
    "b_bend135": b_3seg_bend90,
}

ac = AxialComponent(b_3seg_bend90, football_3seg, smooth_with_post=False)
ac.mesh.show(smooth=False)
