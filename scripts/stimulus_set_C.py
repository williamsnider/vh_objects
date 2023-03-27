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

# 1_2_1 football
x = np.array([0, 0.5, 1])
y = CS_RADII[[1, 2, 1]]
poly = np.polyfit(x, y, 2)
scale_1_2_1 = np.polyval(poly, pos)
cs_1_2_1 = [CrossSection(scale_1_2_1[i] * base_cp, pos[i]) for i in range(NUM_CS)]

# 1_2_1 sheet
sheet_thickness_to_width = 4
sheet_cp = base_cp.copy()
sheet_cp[[0, 1, 7], 0] = 1 / sheet_thickness_to_width
sheet_cp[[3, 4, 5], 0] = -1 / sheet_thickness_to_width
scale_1_1_1 = np.ones(NUM_CS) * CS_RADII[2]
cs_sheet_1_1_1 = [CrossSection(scale_1_1_1[i] * sheet_cp, pos[i]) for i in range(NUM_CS)]
endpoint_offset = np.linalg.norm(sheet_cp[-1] - sheet_cp[1]) / 2
sheet = AxialComponent(
    b_cur2,
    cs_sheet_1_1_1,
    endpoint_offsets=np.array([endpoint_offset * scale_1_1_1[0], endpoint_offset * scale_1_1_1[1]]),
)
sheet.mesh.show(smooth=False)

# sheet_cp =
# t = np.linspace(0,1,100)
# vals = np.polyval(poly, t)
# der = np.polyder(poly,1)

# # Implicitly differentiate equation of circle (x-a)^2 +y^2 = r^2
# x = 1
# y = np.polyval(poly,x)
# m = np.polyval(der, x)
# a = x + y * m  # Solve for a
# r = np.sqrt((x-a)**2 + y**2)  # Solve for r

# # Check result
# assert np.isclose((x-a)**2 + y**2, r**2)


# # a1 = -(np.sqrt(r**2-y**2) - x)
# # a2 = -(-np.sqrt(r**2-y**2) - x)
# th = np.arctan2(y,x-a)
# tt = np.linspace(0,2*th,10000)
# xx = r*np.cos(tt) + a
# yy = r*np.sin(tt)

# (2*a - 2*x)/(2*y)

# (x-a)**2 + y**2
# # Find where poly hits zero
# y0 = 0
# p = np.poly1d(poly)
# rts = (p-y0).roots

# t = np.linspace(rts.min(), rts.max())
# vals = np.polyval(poly, t)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(t, vals)
# ax.plot(xx,yy,"-k")
# ax.plot(x,y,"*r")
# ax.set_aspect("equal")
# plt.show()


########################
### Axial components ###
########################

ac_dict = {
    "ac_lin2_1_1_1": AxialComponent(b_lin2, cs_1_1_1, hemisphere_ends=True),
    "ac_cur2_1_1_1": AxialComponent(b_cur2, cs_1_1_1, hemisphere_ends=True),
    "ac_lin2_1_2_1": AxialComponent(b_lin2, cs_1_2_1, hemisphere_ends=True),
    "ac_cur2_1_2_1": AxialComponent(b_cur2, cs_1_2_1, hemisphere_ends=True),
    "None": None,
}

ac_dict["ac_lin2_1_2_1"].mesh.show(smooth=False)
##############
### Shapes ###
##############

# combs = [[ac_list], [parent_list], [position_along_parent], [euler_angles]]
class ShapeCombinations:
    def __init__(
        self,
        ac_names,
        parent_axial_component_names,
        position_along_parent_list,
        euler_angles_list,
    ):
        self.ac_names = ac_names
        self.parent_axial_component_names = parent_axial_component_names
        self.position_along_parent_list = position_along_parent_list
        self.euler_angles_list = euler_angles_list

        self.ac_list = [ac_dict[s] for s in ac_names]
        self.parent_axial_component_list = [ac_dict[s] for s in parent_axial_component_names]

    def __str__(self):
        print("ac_names: \t\t\t", "\t".join(self.ac_names))
        print("parent_axial_component_names: \t", "\t".join(self.parent_axial_component_names))
        print("position_along_parent: \t\t", "\t".join(map(str, self.position_along_parent_list)))
        print("euler_angles: \t\t\t", "\t".join(map(str, self.euler_angles_list)))
        return ""


comb = ShapeCombinations(
    ac_names=["ac_lin2_1_1_1", "ac_cur2_1_1_1"],
    parent_axial_component_names=["None", "ac_lin2_1_1_1"],
    position_along_parent_list=[0.0, 1.0],
    euler_angles_list=[np.zeros(3), np.zeros(3)],
)


###################
### Single limb ###
###################

# for ac1 in ac_dict.keys():
#     for

# ac1 = copy.deepcopy(ac_lin2_1_1_1)
# ac2 = copy.deepcopy(ac_cur2_1_1_1)
# ac2.parent_axial_component = ac1
# ac2.position_along_parent = 1.0
# ac2.calc_points()
# ac3 = copy.deepcopy(ac_cur2_1_1_1)
# ac3.parent_axial_component = ac2
# ac3.position_along_parent = 1.0
# ac3.calc_points()

# s = Shape([ac1, ac2, ac3])
# s.mesh.show(smooth=False)
