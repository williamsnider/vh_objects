# Linear segment
import copy
import numpy as np
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.utilities import approximate_arc
from objects.shape import Shape

### Parameters ###
NUM_CP_PER_BACKBONE = 5
SEGMENT_LENGTH = 15
NUM_CS = 11
CS_RADII = np.array([0, 7.5, 15])

#################
### Backbones ###
#################

# Linear segment
b_lin2_cp = np.hstack(
    [
        np.linspace(0, 2 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_lin2 = Backbone(b_lin2_cp, reparameterize=True)

# Curved segment
b_cur2_cp = approximate_arc(np.pi / 2, 2 * SEGMENT_LENGTH * np.pi / 2)
b_cur2 = Backbone(b_cur2_cp, reparameterize=True)

######################
### Cross sections ###
######################

pos = np.linspace(0, 1, NUM_CS)
t = np.linspace(0, 2 * np.pi, 8, endpoint=False).reshape(-1, 1)
base_cp = np.hstack([np.cos(t), np.sin(t)])

# 1_1_1 cylindrical
scale_1_1_1 = np.ones(NUM_CS) * CS_RADII[2]
cs_1_1_1 = [CrossSection(scale_1_1_1[i] * base_cp, pos[i]) for i in range(NUM_CS)]

########################
### Axial components ###
########################

ac_lin2_1_1_1 = AxialComponent(b_lin2, cs_1_1_1, hemisphere_ends=True)
ac_cur2_1_1_1 = AxialComponent(b_cur2, cs_1_1_1, hemisphere_ends=True)

##############
### Shapes ###
##############

ac1 = copy.deepcopy(ac_lin2_1_1_1)
ac2 = copy.deepcopy(ac_cur2_1_1_1)
ac2.parent_axial_component = ac1
ac2.position_along_parent = 1.0
ac2.calc_points()
ac3 = copy.deepcopy(ac_cur2_1_1_1)
ac3.parent_axial_component = ac2
ac3.position_along_parent = 1.0
ac3.calc_points()

s = Shape([ac1, ac2, ac3])
s.mesh.show(smooth=False)
