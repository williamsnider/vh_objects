import numpy as np
import matplotlib.pyplot as plt
from objects.utilities import approximate_arc
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.utilities import approximate_arc
from objects.shape import Shape
import copy

# Parameters
X_WIDTH = 2.5
NUM_CS = 11
SEGMENT_LENGTH = 15
NUM_CP_PER_BACKBONE = 5

# Fit profile with quadartic
x = np.array([0, 0.5, 1]) * SEGMENT_LENGTH
y = np.array([1 * X_WIDTH, 4 * X_WIDTH, 1 * X_WIDTH])
poly = np.polyfit(x, y, 2)

# Sample profile
t = np.linspace(0, SEGMENT_LENGTH, 100)
vals = np.polyval(poly, t)
der = np.polyder(poly, 1)

# Find arc with G1 continuity

# Implicitly differentiate equation of circle (x-a)^2 +y^2 = r^2
x = 0
y = np.polyval(poly, x)
m = np.polyval(der, x)
a = x + y * m  # Solve for a
r = np.sqrt((x - a) ** 2 + y**2)  # Solve for r

assert np.isclose((x - a) ** 2 + y**2, r**2)  # Check result


if x == 0:
    th = np.arctan2(y, (x - a))
    arc_length = np.pi - th
    tt = np.linspace(th, np.pi, 10000)
    x_shift = 0
else:
    th = np.arctan2(y, (x - a))
    arc_length = th
    tt = np.linspace(0, th, 10000)
    x_shift = x

# Plot smooth arc
xx = r * np.cos(tt) + a
yy = r * np.sin(tt)

# Calculate controlpoint arc

cp = approximate_arc(arc_length, r * arc_length)
cp_rot = cp[:, [1, 0]]  # Rotate 45deg
if x == 0:
    cp_rot[:, 0] *= -1
cp_T = cp_rot - np.array([cp_rot[-1, 0] - x_shift, 0])  # Shift to align with end of quadratic

fig, ax = plt.subplots()
ax.plot(t, vals)
ax.plot(xx, yy, "-k")  # Arc
ax.plot(x, y, "*r")  # Intersection point
ax.plot(cp_T[:, 0], cp_T[:, 1], "g-")
ax.set_aspect("equal")
plt.show()

pos = np.linspace(0, 1, NUM_CS)
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

# Test this on axial component
x = np.array([0, 0.5, 1])
y = np.array([X_WIDTH, 4 * X_WIDTH, X_WIDTH])
poly = np.polyfit(x, y, 2)
scale_1_4_1 = np.polyval(poly, pos)
cs_1_4_1 = [CrossSection(scale_1_4_1[i] * base_cp, pos[i]) for i in range(NUM_CS)]
football = AxialComponent(
    b_lin2,
    cs_1_4_1,
    fair_ends=False,
)
football.mesh.show(smooth=False)


def calc_cp_hemisphere(base_cp, endpoint, tan_vec, radius):

    # Rotate base_cp to line in yz plane
    vecA = base_cp[0] - base_cp[1]
    vecB = base_cp[0] - base_cp[2]
    N = np.cross(vecA, vecB) / np.linalg.norm(np.cross(vecA, vecB))
    assert np.all(np.isclose(N, tan_vec)) or np.all(np.isclose(N, -tan_vec))
    T = np.cross(vecA, N) / np.linalg.norm(np.cross(vecA, N))
    B = np.cross(T, N)
    curr = np.vstack([N.reshape(1, -1), T.reshape(1, -1), B.reshape(1, -1)])
    goal = np.eye(3)
    R = (goal @ np.linalg.inv(curr.T)).T

    yz_cp = (base_cp - endpoint) @ R

    cp = np.tile(yz_cp, (5, 1, 1))
    cp_scale = cp * out1[::-1, 0].reshape(-1, 1, 1)
    cp_scale[:, :, 0] = yz_cp[:, 0]

    vec_rotated = tan_vec @ R
    cp_shift = cp_scale + vec_rotated * vec_frac * radius

    # Transform back to original plane
    result = (cp_shift) @ R.T + endpoint

    return result


# Update controlpoints
old_cp = football.controlpoints
mid_cp = old_cp[2:-2]
new_cp = np.zeros((mid_cp.shape[0] + 2 * 4, *mid_cp.shape[1:]))

cs_cp = mid_cp[0]
calc_cp_hemisphere(cs_cp, football.r(0.0), football.T(0.0), 3)
# Make new mesh
new_ac = copy.deepcopy(football)
new_ac.controlpoints = new_cp
new_ac.num_rows = new_cp.shape[0]
new_ac.make_surface()
new_ac.make_mesh()
new_ac.mesh.show(smooth=False)
