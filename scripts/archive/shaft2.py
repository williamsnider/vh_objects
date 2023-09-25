import numpy as np
import matplotlib.pyplot as plt
from objects.utilities import approximate_arc
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.utilities import approximate_arc, calc_hemisphere_controlpoints
from objects.shape import Shape
import copy


# Parameters
X_WIDTH = 2.5
NUM_CS = 11
SEGMENT_LENGTH = 15
NUM_CP_PER_BACKBONE = 5

pos = np.linspace(0, 1, NUM_CS)
num_cs_adjusted = len(pos)
pos_seg = pos * SEGMENT_LENGTH
t = np.linspace(0, 2 * np.pi, 8, endpoint=False).reshape(-1, 1)
base_cp = np.hstack([np.cos(t), np.sin(t)])

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

# Test this on axial component
ac_x = np.array([0, 0.5, 1])
ac_y = np.array([2 * X_WIDTH, 1 * X_WIDTH, 1 * X_WIDTH])
poly = np.polyfit(ac_x, ac_y, 2)
poly_seg = np.polyfit(ac_x * SEGMENT_LENGTH * 2, ac_y, 2)
scale_1_4_1 = np.polyval(poly, pos)
cs_1_4_1 = [
    CrossSection(scale_1_4_1[i] * base_cp, pos[i]) for i in range(num_cs_adjusted)
]
football = AxialComponent(
    b_cur2,
    cs_1_4_1,
    hemispherical_ends=True,
    hemispherical_polynomial=poly_seg,
    hemisphere_x=[0, 2 * SEGMENT_LENGTH],
)

football.mesh.show()

# Plot controlpoints
ax = plt.figure().add_subplot(projection="3d")
arr = football.controlpoints
for i in range(arr.shape[0]):
    ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")
plt.show()

# Plot profile to assess smoothness
section = football.mesh.section(np.array([0, 1, 0]), np.array([0, 0, 0]))
slice_2D, to_3D = section.to_planar()
polygon = slice_2D.polygons_full[0]
yy, xx = polygon.exterior.coords.xy
xx = np.array(xx)
yy = np.array(yy)
xx = xx[yy > 0]
yy = yy[yy > 0]
yy = yy[xx.argsort()]
xx = xx[xx.argsort()]

# Analyze slopes
pts = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
dydx = np.diff(pts[:, 1]) / np.diff(pts[:, 0])


ax = plt.figure().add_subplot()
# ax.plot(xx[:-1], yy[:-1], "b-*")
ax.plot(xx[1:], dydx, "b-")
ax.set_aspect("equal")
plt.show()


# # Fit profile with quadartic
# x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH
# y = np.array([1 * X_WIDTH, 4 * X_WIDTH, 1 * X_WIDTH])
# poly_seg = np.polyfit(x, y, 2)

# # Sample profile
# t = np.linspace(0, SEGMENT_LENGTH, 100)
# vals = np.polyval(poly, t)
# der = np.polyder(poly, 1)

# # Find arc with G1 continuity

# # Implicitly differentiate equation of circle (x-a)^2 +y^2 = r^2
# x = 0
# y = np.polyval(poly_seg, x)
# m = np.polyval(der, x)
# a = x + y * m  # Solve for a
# r = np.sqrt((x - a) ** 2 + y**2)  # Solve for r

# assert np.isclose((x - a) ** 2 + y**2, r**2)  # Check result


# if x == 0:
#     th = np.arctan2(y, (x - a))
#     arc_length = np.pi - th
#     tt = np.linspace(th, np.pi, 10000)
#     x_shift = 0
# else:
#     th = np.arctan2(y, (x - a))
#     arc_length = th
#     tt = np.linspace(0, th, 10000)
#     x_shift = x

# # Plot smooth arc
# xx = r * np.cos(tt) + a
# yy = r * np.sin(tt)

# # Calculate controlpoint arc

# cp = approximate_arc(arc_length, r * arc_length)
# cp_rot = cp[:, [1, 0]]  # Rotate 45deg
# if x == 0:
#     cp_rot[:, 0] *= -1
# cp_T = cp_rot - np.array(
#     [cp_rot[-1, 0] - x_shift, 0]
# )  # Shift to align with end of quadratic

# # Redo controlpoint arc using ratios
# new_cp = np.tile([x, y], (5, 1))
# scale_ratio = cp_T[:, 1] / y
# vec = np.array([1, 0]).reshape(1, -1)

# # Scale points based on leftmost cross section
# new_cp[:, 1] *= scale_ratio

# # Translate based on falloff
# shift = vec * cp_T[:, [0]]
# new_cp += shift

# fig, ax = plt.subplots()
# ax.plot(t, vals)
# ax.plot(xx, yy, "-k")  # Arc
# ax.plot(x, y, "*r")  # Intersection point
# ax.plot(new_cp[:, 0], new_cp[:, 1], "g-")
# ax.set_aspect("equal")
# plt.show()

# pos = np.linspace(0, 1, NUM_CS)
# pos_seg = pos * SEGMENT_LENGTH
# t = np.linspace(0, 2 * np.pi, 8, endpoint=False).reshape(-1, 1)
# base_cp = np.hstack([np.cos(t), np.sin(t)])

# # Linear segment

# b_lin2_cp = np.hstack(
#     [
#         np.linspace(0, 2 * SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(-1, 1),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#         np.zeros((NUM_CP_PER_BACKBONE, 1)),
#     ]
# )
# b_lin2 = Backbone(b_lin2_cp, reparameterize=True)

# # Test this on axial component
# ac_x = np.array([0, 0.5, 1])
# ac_y = np.array([X_WIDTH, 4 * X_WIDTH, X_WIDTH])
# poly = np.polyfit(ac_x, ac_y, 2)
# scale_1_4_1 = np.polyval(poly, pos)
# cs_1_4_1 = [CrossSection(scale_1_4_1[i] * base_cp, pos[i]) for i in range(NUM_CS)]
# football = AxialComponent(
#     b_lin2,
#     cs_1_4_1,
#     fair_ends=False,
# )
# # football.mesh.show(smooth=False)


# # def calc_cp_hemisphere(base_cp, endpoint, tan_vec, radius):

# #     # Rotate base_cp to line in yz plane
# #     vecA = base_cp[0] - base_cp[1]
# #     vecB = base_cp[0] - base_cp[2]
# #     N = np.cross(vecA, vecB) / np.linalg.norm(np.cross(vecA, vecB))
# #     assert np.all(np.isclose(N, tan_vec)) or np.all(np.isclose(N, -tan_vec))
# #     T = np.cross(vecA, N) / np.linalg.norm(np.cross(vecA, N))
# #     B = np.cross(T, N)
# #     curr = np.vstack([N.reshape(1, -1), T.reshape(1, -1), B.reshape(1, -1)])
# #     goal = np.eye(3)
# #     R = (goal @ np.linalg.inv(curr.T)).T

# #     yz_cp = (base_cp - endpoint) @ R

# #     cp = np.tile(yz_cp, (5, 1, 1))
# #     cp_scale = cp * out1[::-1, 0].reshape(-1, 1, 1)
# #     cp_scale[:, :, 0] = yz_cp[:, 0]

# #     vec_rotated = tan_vec @ R
# #     cp_shift = cp_scale + vec_rotated * vec_frac * radius

# #     # Transform back to original plane
# #     result = (cp_shift) @ R.T + endpoint

# #     return result


# # Update controlpoints
# old_cp = football.controlpoints
# mid_cp = old_cp[2:-2]
# new_cp = np.zeros((mid_cp.shape[0] + 2 * 4, *mid_cp.shape[1:]))
# new_cp[4:-4] = mid_cp

# # Rotate base_cp to be in yz plane
# ac = football
# base_cp = mid_cp[0]
# tan_vec = ac.T(0.0)
# endpoint = ac.r(0.0)
# vecA = base_cp[0] - base_cp[1]
# vecB = base_cp[0] - base_cp[2]
# N = np.cross(vecA, vecB) / np.linalg.norm(np.cross(vecA, vecB))
# assert np.all(np.isclose(N, tan_vec)) or np.all(np.isclose(N, -tan_vec))
# T = np.cross(vecA, N) / np.linalg.norm(np.cross(vecA, N))
# B = np.cross(T, N)
# curr = np.vstack([N.reshape(1, -1), T.reshape(1, -1), B.reshape(1, -1)])
# goal = np.eye(3)
# R = (goal @ np.linalg.inv(curr.T)).T
# yz_cp = (base_cp - endpoint) @ R

# scale_ratio = cp_T[:, 1] / y

# # Scale these controlpoints
# out_cp = np.tile(yz_cp, (5, 1, 1))
# cp_scale = out_cp * scale_ratio.reshape(-1, 1, 1)
# cp_scale[:, :, 0] = yz_cp[:, 0]

# # Translate these controlpoints
# vec_rotated = tan_vec @ R
# cp_shift = cp_scale + vec_rotated * cp_T[:, 0].reshape(-1, 1, 1)

# result = (cp_shift) @ R.T + endpoint

# # Scale points
# base_cp = mid_cp[0]
# new_cp[:4] = result[:4]  # Omit overlapping controlpoints


# # Plot controlpoints
# ax = plt.figure().add_subplot(projection="3d")
# arr = new_cp
# for i in range(arr.shape[0]):
#     ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")
# plt.show()

# # Make new mesh
# new_ac = copy.deepcopy(football)
# new_ac.controlpoints = new_cp
# new_ac.num_rows = new_cp.shape[0]
# new_ac.make_surface()
# new_ac.make_mesh()
# new_ac.mesh.show(smooth=False)
