import igl
import scipy as sp
import numpy as np
from meshplot import plot, subplot, interact
import meshplot

meshplot.offline()

import matplotlib.pyplot as plt
import os
from objects import *

root_folder = os.getcwd()

v, f = igl.read_triangle_mesh(os.path.join(root_folder, "bump_domain.obj"))
u = v.copy()

# Find boundary vertices outside annulus
vrn = np.linalg.norm(v, axis=1)
is_outer = [vrn[i] - 1.00 > -1e-15 for i in range(v.shape[0])]
is_inner = [vrn[i] - 0.15 < 1e-15 for i in range(v.shape[0])]
in_b = [is_outer[i] or is_inner[i] for i in range(len(is_outer))]

b = np.array([i for i in range(v.shape[0]) if (in_b[i])]).T
bc = np.zeros(b.size)

for bi in range(b.size):
    bc[bi] = 0.0 if is_outer[b[bi]] else 1.0

c = np.array(is_outer)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=-90, azim=90)

# Entire mesh
x, y, z = v.T
ax.plot(x, y, z, ".", color="black")

x, y, z = v[is_outer].T
ax.plot(x, y, z, ".", color="green")

x, y, z = v[is_inner].T
ax.plot(x, y, z, ".", color="red")

x, y, z = v[b[bc.astype("bool")]].T
ax.plot(x, y, z, ".", color="purple")
plt.show()
for i in range(1, 5):
    z = igl.harmonic_weights(v, f, b, bc, int(i))
    u[:, 2] = z
    if i == 1:
        p = subplot(u, f, c, shading={"wire_width": 0.01, "colormap": "tab10"}, s=[1, 4, i - 1])
    else:
        subplot(u, f, c, shading={"wire_width": 0.01, "colormap": "tab10"}, s=[1, 4, i - 1], data=p)
p
