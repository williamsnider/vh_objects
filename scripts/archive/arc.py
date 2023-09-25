import numpy as np
from objects.backbone import Backbone
from scripts.stimulus_set_params import (
    SEGMENT_LENGTH,
    X_WIDTH,
    NUM_CP_PER_BACKBONE,
    NUM_CP_PER_CROSS_SECTION,
)
from objects.utilities import approximate_arc
import matplotlib.pyplot as plt

from scipy.interpolate import CubicHermiteSpline, CubicSpline
from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection

join_radius = 10

# Flat
flat_cp = np.vstack(
    [
        np.linspace(0, SEGMENT_LENGTH, NUM_CP_PER_BACKBONE).reshape(1, -1),
        np.zeros((1, NUM_CP_PER_BACKBONE)),
        np.zeros((1, NUM_CP_PER_BACKBONE)),
    ]
).T

# Arc
arc_cp = approximate_arc(np.pi / 2, SEGMENT_LENGTH, 25)
arc_cp[:, 1] *= -1  # Flip direction across yz axis

# Combined
comb_cp = np.vstack([arc_cp, arc_cp[1:] + arc_cp[-1]])

# mid_idx = (comb_cp.shape[0]-1)//2
# comb_cp[mid_idx] = comb_cp[[mid_idx-1, mid_idx, mid_idx+1]].mean(axis=0)

b = Backbone(comb_cp, reparameterize=True)
t = np.linspace(0, 1, 1000)
pts = b.r(t)

th = np.linspace(0, 2 * np.pi)
cx = join_radius * np.cos(th) + arc_cp[-1, 0]
cy = join_radius * np.sin(th) + arc_cp[-1, 1]

# Find t where radius intersects backbone
from scipy.optimize import minimize_scalar


def calc_dydx(b, t):

    return (b.r(t) - b.r(t - 0.001)) / (0.001)


def cost_func(t):

    # Calc pts
    xy = b.r(t)[0, :2]

    # Calc dist to sphere
    dist_x = (cx - xy[0]) ** 2
    dist_y = (cy - xy[1]) ** 2

    return (dist_x + dist_y).min()


t1 = minimize_scalar(cost_func, bounds=[0.0, 0.5], method="bounded")
t2 = minimize_scalar(cost_func, bounds=[0.5, 1.0], method="bounded")

# Plot hits
h1 = b.r(t1.x)
h2 = b.r(t2.x)

# Fit Cubic Hermite Splien
x = np.array([0, 1])
y = np.vstack([b.r(t1.x)[0, :2], b.r(t2.x)[0, :2]])
dydx = np.vstack([calc_dydx(b, t1.x)[0, :2], calc_dydx(b, t2.x)[0, :2]])
# dydx = np.vstack([b.dx(t1.x)[0, :2], b.dx(t2.x)[0, :2]])
chs = CubicHermiteSpline(x, y, dydx)

# Fit cubic spline
outside = np.logical_or(t < t1.x, t > t2.x)
inside = ~outside
x = t[outside]
y = b.r(x)
cs = CubicSpline(x, y)
cs_pts = cs(np.linspace(0, 1, 50))

# New backbone
b = Backbone(cs_pts, reparameterize=True)

# New Axial component
t = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
base_cp = np.hstack([np.cos(t), np.sin(t)])

NUM_CS = 10
half_scale = np.linspace(2 * join_radius, join_radius, NUM_CS)
scale = np.concatenate([half_scale, half_scale[::-1]])
pos = np.linspace(0, 1, 2 * NUM_CS)
cs_list = [CrossSection(base_cp * scale[i], pos[i]) for i in range(NUM_CS * 2)]
ac = AxialComponent(b, cs_list)

cpts = chs(np.linspace(0, 1, 1000))
ax = plt.figure().add_subplot()
# ax.plot(pts[:, 0], pts[:, 1], "k-")
# ax.plot(comb_cp[:, 0], comb_cp[:, 1], "r*")
# ax.plot(cx, cy, "g-")
# ax.plot(h1[:, 0], h1[:, 1], "b*")
# ax.plot(h2[:, 0], h2[:, 1], "b*")
ax.plot(cs_pts[:, 0], cs_pts[:, 1], "k*-")
ax.set_aspect("equal")
plt.show()
