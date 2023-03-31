# Linear segment
import copy
import numpy as np
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.utilities import approximate_arc
from objects.shape import Shape

# TODO: Check RADII
# TODO: Ensure parity between orthogonal and colinear components -- probably best to fuse complete components
# TODO: Think about slanting everything away for ease of 3D printing (without supports)


### Parameters ###
NUM_CP_PER_BACKBONE = 5
SEGMENT_LENGTH = 20
NUM_CS = 11
BASE_RADIUS = 7  # Radius off which other features are derived
VOLUMETRIC_MIDDLE_FACTOR = 3  # Middle cs radius is this times larger than edge cs
LEAF_SCALE_FACTOR = 2
SHEET_THICKNESS = BASE_RADIUS / 2
SHEET_WIDTH = BASE_RADIUS  # Width at base of sheet, will be larger with scaling
POS_SHEET = np.linspace(0.05, 0.95, NUM_CS)  # Scale shorter to allow for roundover

# Derived parameters
x_width = BASE_RADIUS / 2
cs_radii = np.arange(3) * BASE_RADIUS

##############################
### Cross Section Profiles ###
##############################

# Circular cross sections
t = np.linspace(0, 2 * np.pi, 8, endpoint=False).reshape(-1, 1)
round_cp = np.hstack([np.cos(t), np.sin(t)])

# Sheet cross sections # TODO: May want this to have better rounded edges
sheet_cp = np.array(
    [
        [SHEET_THICKNESS / 2, 0],
        [SHEET_THICKNESS / 2, SHEET_WIDTH / 2],
        [0, SHEET_WIDTH / 2 + SHEET_THICKNESS / 2],  # Roundover edge
        [-SHEET_THICKNESS / 2, SHEET_WIDTH / 2],
        [-SHEET_THICKNESS / 2, 0],
        [-SHEET_THICKNESS / 2, -SHEET_WIDTH / 2],
        [0, -SHEET_WIDTH / 2 - SHEET_THICKNESS / 2],  # Roundover edge
        [SHEET_THICKNESS / 2, -SHEET_WIDTH / 2],
    ]
)


#################
### Backbones ###
#################

# Linear 0.5 segment
b_lin05_cp = np.hstack(
    [
        np.linspace(0, SEGMENT_LENGTH / 2, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_lin05 = Backbone(b_lin05_cp, reparameterize=True)

# Linear 1 segment
b_lin1_cp = np.hstack(
    [
        np.linspace(0, SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_lin1 = Backbone(b_lin1_cp, reparameterize=True)

# Linear 2 segment (L1+L2)
b_lin2_cp = np.hstack(
    [
        np.linspace(0, 2 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
        np.zeros((NUM_CP_PER_BACKBONE, 1)),
    ]
)
b_lin2 = Backbone(b_lin2_cp, reparameterize=True)

# Curved 0.5 segment
b_cur05_cp = approximate_arc(np.pi / 2, SEGMENT_LENGTH / 2)
b_cur05 = Backbone(b_cur05_cp, reparameterize=True)

# Curved 1 segment
b_cur1_cp = approximate_arc(np.pi / 2, SEGMENT_LENGTH)
b_cur1 = Backbone(b_cur1_cp, reparameterize=True)

# # Curved segment + stick (used for appendages to volumetric)
# STICK_LENGTH = 4 * X_WIDTH
# NUM_CP_STICK = 3
# b_stick_cp = np.hstack(
#     [
#         np.linspace(0, STICK_LENGTH, NUM_CP_STICK).reshape(-1, 1),
#         np.zeros((NUM_CP_STICK, 1)),
#         np.zeros((NUM_CP_STICK, 1)),
#     ]
# )
# b_cur1_s_cp = np.vstack(
#     [
#         b_stick_cp[:-1],
#         b_cur1_cp + b_stick_cp[-1],
#     ]
# )
# b_cur1_s = Backbone(b_cur1_s_cp, reparameterize=True)

########################
### Axial Components ###
########################

### Common variables ###
pos = np.linspace(0, 1, NUM_CS)  # Position from (0,1)
pos_seg1 = pos * SEGMENT_LENGTH  # Position from (0, SEGMENT_LENGTH)
pos_seg2 = pos_seg1 * 2  # Position from (0, 2*SEGMENT_LENGTH)
pos_seg05 = pos * SEGMENT_LENGTH / 2  # Position from (0, SEGMENT_LENGTH/2)

### Volumetric ("football") ###

# Fit quadratic polynomial to determine scaling of cross sections
x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH * 2
y = np.array([x_width, VOLUMETRIC_MIDDLE_FACTOR * x_width, x_width])
volumetric_poly = np.polyfit(x, y, 2)
volumetric_scale = np.polyval(volumetric_poly, pos_seg2)
volumetric_cs = [
    CrossSection(volumetric_scale[i] * round_cp, pos[i]) for i in range(NUM_CS)
]

# Construct axial component
volumetric = AxialComponent(
    b_lin2,
    volumetric_cs,
    hemispherical_ends=True,
    hemispherical_polynomial=volumetric_poly,
    hemisphere_x=[0, 2 * SEGMENT_LENGTH],
)

### Sheet - Leaf - Zero Curvature ###

# Fit quadratic polynomial to determine scaling of cross sections
x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH
y = np.array([1, LEAF_SCALE_FACTOR, 1])  # Width of sheet already factored in sheet_cp
sheet_leaf_0K_poly = np.polyfit(x, y, 2)
sheet_leaf_0K_scale = np.hstack(
    [
        np.ones((len(pos_seg1), 1)),
        np.polyval(sheet_leaf_0K_poly, pos_seg1).reshape(-1, 1),
    ]
)
sheet_leaf_0K_cs = [
    CrossSection(sheet_leaf_0K_scale[i] * sheet_cp, POS_SHEET[i]) for i in range(NUM_CS)
]

# Construct axial component
sheet_leaf_0K = AxialComponent(
    b_lin1,
    sheet_leaf_0K_cs,
)

### Sheet - Leaf - With Curvature ###

# Fit quadratic polynomial to determine scaling of cross sections
x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH
y = np.array([1, LEAF_SCALE_FACTOR, 1])  # Width of sheet already factored in sheet_cp
sheet_leaf_1K_poly = np.polyfit(x, y, 2)
sheet_leaf_1K_scale = np.hstack(
    [
        np.ones((len(pos_seg1), 1)),
        np.polyval(sheet_leaf_0K_poly, pos_seg1).reshape(-1, 1),
    ]
)
sheet_leaf_1K_cs = [
    CrossSection(sheet_leaf_1K_scale[i] * sheet_cp, POS_SHEET[i]) for i in range(NUM_CS)
]

# Construct axial component
sheet_leaf_1K = AxialComponent(
    b_cur1,
    sheet_leaf_1K_cs,
)

### Sheet - Point - Zero Curvature ###

# Fit quadratic polynomial to determine scaling of cross sections
x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH / 2
y = np.geomspace(1, 0.2, 3)
sheet_point_0K_poly = np.polyfit(x, y, 2)
sheet_point_0K_scale = np.hstack(
    [
        np.ones((len(pos_seg1), 1)),
        np.polyval(sheet_point_0K_poly, pos_seg05).reshape(-1, 1),
    ]
)
sheet_point_0K_cs = [
    CrossSection(sheet_point_0K_scale[i] * sheet_cp, POS_SHEET[i])
    for i in range(NUM_CS)
]

# Construct axial component
sheet_point_0K = AxialComponent(
    b_lin05,
    sheet_point_0K_cs,
)

### Sheet - Point - With Curvature ###

# Fit quadratic polynomial to determine scaling of cross sections
x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH / 2
y = np.geomspace(1, 0.2, 3)
sheet_point_1K_poly = np.polyfit(x, y, 2)
sheet_point_1K_scale = np.hstack(
    [
        np.ones((len(pos_seg1), 1)),
        np.polyval(sheet_point_1K_poly, pos_seg05).reshape(-1, 1),
    ]
)
sheet_point_1K_cs = [
    CrossSection(sheet_point_1K_scale[i] * sheet_cp, POS_SHEET[i])
    for i in range(NUM_CS)
]

# Construct axial component
sheet_point_1K = AxialComponent(
    b_cur05,
    sheet_point_1K_cs,
)
sheet_point_1K.mesh.show()


### Sheet - Bump - Zero Curvature ###
# XXX: This is wrong XXX

# Create scale of cross sections
th = np.linspace(0, np.pi / 2, NUM_CS, endpoint=False)
sheet_bump_0K_scale = np.hstack(
    [
        np.ones((len(pos_seg1), 1)),
        np.cos(th).reshape(-1, 1),
    ]
)
sheet_bump_0K_cs = [
    CrossSection(sheet_bump_0K_scale[i] * sheet_cp, pos[i]) for i in range(NUM_CS)
]

# Construct axial component
sheet_bump_0K = AxialComponent(
    b_lin05,
    sheet_bump_0K_cs,
)
sheet_bump_0K.mesh.show()

### Bump - Zero Curvature ###
from objects.utilities import calc_hemisphere_controlpoints


# Calculate control points for hemisphere
hemi_base_cp = np.hstack([np.zeros((round_cp.shape[0], 1)), round_cp])
x = np.array([0, 1, 2])
y = np.array([1, 1, 1])
poly = np.polyfit(x, y, 2)  # Fit a linear polynomial for function
cp = calc_hemisphere_controlpoints(
    hemi_base_cp, np.array([1, 0, 0]), np.array([0, 0, 0]), poly, 0
)

cp = cp[1:]
# full = np.vstack([cp, cp[-2::-1] * np.array([-1,1,1])])
bump_0K = copy.deepcopy(sheet_bump_0K)
bump_0K.controlpoints = cp * np.array([-1, 1, 1])
bump_0K.num_rows = cp.shape[0]
bump_0K.make_surface()
bump_0K.make_mesh()
bump_0K.mesh.show()


sheet_bump_0K_cp = cp * np.array([-1, 1, 1])  # Copy
sheet_bump_0K_cp *= (SHEET_THICKNESS + SHEET_WIDTH) / 2  # Scale
# Squash

import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection="3d")
arr = sheet_bump_0K_cp
for i in range(arr.shape[0]):
    ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")

# Set scale
xs = arr[:, :, 0].ravel()
ys = arr[:, :, 1].ravel()
zs = arr[:, :, 2].ravel()
ax.set_box_aspect(
    (np.ptp(xs), np.ptp(ys), np.ptp(zs))
)  # aspect ratio is 1:1:1 in data space
plt.show()


# # volumetric.mesh.show()


# ######################
# ### Cross sections ###
# ######################

# pos = np.linspace(0, 1, NUM_CS)
# pos_seg = pos * SEGMENT_LENGTH
# pos_seg2 = pos_seg * 2
# t = np.linspace(0, 2 * np.pi, 8, endpoint=False).reshape(-1, 1)
# base_cp = np.hstack([np.cos(t), np.sin(t)])

# # 1_1_1 cylindrical
# scale_1_1_1 = np.ones(NUM_CS) * CS_RADII[2]
# cs_1_1_1 = [CrossSection(scale_1_1_1[i] * base_cp, pos[i]) for i in range(NUM_CS)]

# # 1_4_1 football
# x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH * 2
# y = np.array([X_WIDTH, 4 * X_WIDTH, X_WIDTH])
# poly_1_4_1 = np.polyfit(x, y, 2)
# scale_1_4_1 = np.polyval(poly_1_4_1, pos_seg2)
# cs_1_4_1 = [CrossSection(scale_1_4_1[i] * base_cp, pos[i]) for i in range(NUM_CS)]
# # Calculate when polynomial would hit zero

# # 1_2_1 sheet
# sheet_thickness_to_width = 4
# sheet_cp = base_cp.copy()
# sheet_cp[[0, 1, 7], 0] = 1 / sheet_thickness_to_width
# sheet_cp[[3, 4, 5], 0] = -1 / sheet_thickness_to_width
# scale_1_1_1 = np.ones(NUM_CS) * CS_RADII[1]
# cs_sheet_1_1_1 = [
#     CrossSection(scale_1_1_1[i] * sheet_cp, pos_sheet[i]) for i in range(NUM_CS)
# ]
# # endpoint_offset = np.linalg.norm(sheet_cp[-1] - sheet_cp[1]) / 2

# # 1_2_1
# x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH
# y = np.array([X_WIDTH, 2 * X_WIDTH, X_WIDTH])
# poly_1_2_1 = np.polyfit(x, y, 2)
# scale_1_2_1 = np.polyval(poly_1_2_1, pos_seg)
# cs_1_2_1 = [CrossSection(scale_1_2_1[i] * base_cp, pos[i]) for i in range(NUM_CS)]

# ########################
# ### Axial components ###
# ########################

# volumetric = AxialComponent(
#     b_lin2,
#     cs_1_4_1,
#     hemispherical_ends=True,
#     hemispherical_polynomial=poly_1_4_1,
#     hemisphere_x=[0, 2 * SEGMENT_LENGTH],
# )
# # volumetric.mesh.show()

# sheet = AxialComponent(
#     b_cur1,
#     cs_sheet_1_1_1,
#     # parent_axial_component=volumetric,
#     # position_along_parent=0.5,
#     # euler_angles=np.array([0, np.pi / 2, 0]),
# )

# sheet2 = AxialComponent(
#     b_cur1,
#     cs_sheet_1_1_1,
#     # parent_axial_component=volumetric,
#     # position_along_parent=0.5,
#     # euler_angles=np.array([0, np.pi / 2, 0]),
# )

# bulb = AxialComponent(
#     b_lin1,
#     cs_1_2_1,
#     hemispherical_ends=True,
#     hemispherical_polynomial=poly_1_2_1,
#     hemisphere_x=[0, SEGMENT_LENGTH],
#     parent_axial_component=volumetric,
#     position_along_parent=1.0,
# )

# # head = AxialComponent(
# #     b_cur1,
# #     cs_sheet_1_2_1,
# #     parent_axial_component=volumetric,
# #     position_along_parent=0.5,
# #     euler_angles=np.array([0, np.pi / 2, 0]),
# # )


# # s = Shape([volumetric, sheet])
# # s.mesh.show()

# # # Affix sheet to volumetric
# xyz = volumetric.r(0.5) + volumetric.N(0.5) * 2.5 * X_WIDTH
# curr = np.hstack(
#     [
#         sheet.T(0.0).reshape(-1, 1),
#         sheet.N(0.0).reshape(-1, 1),
#         sheet.B(0.0).reshape(-1, 1),
#     ]
# )
# import scipy.spatial.transform

# euler = np.array([np.pi / 2, 0, np.pi / 2])
# goal = scipy.spatial.transform.Rotation.from_euler("xyz", euler).as_matrix()
# R = (goal.T @ np.linalg.inv(curr.T)).T
# T1 = np.eye(4)
# T1[:3, :3] = R
# T1[:3, 3] = xyz
# new_sheet = sheet.mesh.copy()
# new_sheet = new_sheet.apply_transform(T1)
# sheet.mesh.apply_transform(T1)

# xyz = volumetric.r(0.5) + -volumetric.N(0.5) * 2.5 * X_WIDTH
# curr = np.hstack(
#     [
#         sheet.T(0.0).reshape(-1, 1),
#         sheet.N(0.0).reshape(-1, 1),
#         sheet.B(0.0).reshape(-1, 1),
#     ]
# )
# euler = np.array([np.pi / 2, 0, -np.pi / 2])
# goal = scipy.spatial.transform.Rotation.from_euler("xyz", euler).as_matrix()
# R = (goal.T @ np.linalg.inv(curr.T)).T
# T1 = np.eye(4)
# T1[:3, :3] = R
# T1[:3, 3] = xyz
# new_sheet = sheet.mesh.copy()
# new_sheet = new_sheet.apply_transform(T1)
# sheet2.mesh.apply_transform(T1)

# # import trimesh

# # scene = trimesh.Scene()
# # scene.add_geometry(volumetric.mesh)
# # scene.add_geometry(new_sheet)
# # scene.show()

# s = Shape([volumetric, sheet, sheet2, bulb])
# s.mesh.show()
# # ac_dict = {
# #     "ac_lin2_1_1_1": AxialComponent(b_lin2, cs_1_1_1, hemisphere_ends=True),
# #     "ac_cur2_1_1_1": AxialComponent(b_cur2, cs_1_1_1, hemisphere_ends=True),
# #     "ac_lin2_1_2_1": AxialComponent(b_lin2, cs_1_2_1, hemisphere_ends=True),
# #     "ac_cur2_1_2_1": AxialComponent(b_cur2, cs_1_2_1, hemisphere_ends=True),
# #     "None": None,
# # }

# # ac_dict["ac_lin2_1_2_1"].mesh.show(smooth=False)
# ##############
# ### Shapes ###
# ##############

# # # combs = [[ac_list], [parent_list], [position_along_parent], [euler_angles]]
# # class ShapeCombinations:
# #     def __init__(
# #         self,
# #         ac_names,
# #         parent_axial_component_names,
# #         position_along_parent_list,
# #         euler_angles_list,
# #     ):
# #         self.ac_names = ac_names
# #         self.parent_axial_component_names = parent_axial_component_names
# #         self.position_along_parent_list = position_along_parent_list
# #         self.euler_angles_list = euler_angles_list

# #         self.ac_list = [ac_dict[s] for s in ac_names]
# #         self.parent_axial_component_list = [
# #             ac_dict[s] for s in parent_axial_component_names
# #         ]

# #     def __str__(self):
# #         print("ac_names: \t\t\t", "\t".join(self.ac_names))
# #         print(
# #             "parent_axial_component_names: \t",
# #             "\t".join(self.parent_axial_component_names),
# #         )
# #         print(
# #             "position_along_parent: \t\t",
# #             "\t".join(map(str, self.position_along_parent_list)),
# #         )
# #         print("euler_angles: \t\t\t", "\t".join(map(str, self.euler_angles_list)))
# #         return ""


# # comb = ShapeCombinations(
# #     ac_names=["ac_lin2_1_1_1", "ac_cur2_1_1_1"],
# #     parent_axial_component_names=["None", "ac_lin2_1_1_1"],
# #     position_along_parent_list=[0.0, 1.0],
# #     euler_angles_list=[np.zeros(3), np.zeros(3)],
# # )


# ###################
# ### Single limb ###
# ###################

# # for ac1 in ac_dict.keys():
# #     for

# # ac1 = copy.deepcopy(ac_lin2_1_1_1)
# # ac2 = copy.deepcopy(ac_cur2_1_1_1)
# # ac2.parent_axial_component = ac1
# # ac2.position_along_parent = 1.0
# # ac2.calc_points()
# # ac3 = copy.deepcopy(ac_cur2_1_1_1)
# # ac3.parent_axial_component = ac2
# # ac3.position_along_parent = 1.0
# # ac3.calc_points()

# # s = Shape([ac1, ac2, ac3])
# # s.mesh.show(smooth=False)
