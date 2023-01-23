# Stimulus Set A
import numpy as np
from objects.utilities import fit_radius, approximate_arc
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.backbone import Backbone

# Parameters
RADIUS_LIST =  [5,8,9,12,15]  # mm
BACKBONE_ANGLE_LIST = np.linspace(0,1,5)*np.pi  # radians
LENGTH = 80  # mm; does not include post or interface
NUM_CP_PER_CROSS_SECTION = 8



# Hook

# Points along curve
angle = np.pi
dist = LENGTH
radius = dist/angle
t = np.linspace(angle,0,15, endpoint=False).reshape(-1,1)
t = np.flip(t)
c = np.cos
s = np.sin
curve_cp = np.hstack([radius*s(t), radius*c(t), np.zeros(t.shape)])
curve_cp -= np.array([0, radius, 0 ])

# Smoothed hook backbone
j = 4
backbone_angle = BACKBONE_ANGLE_LIST[j]
backbone_cp = approximate_arc(backbone_angle, LENGTH)
s = backbone_cp[1,0]
s=0
POST_LENGTH = 15
post_cp = np.array([[-s, POST_LENGTH, 0],[-s, POST_LENGTH*2/3, 0],[-s,POST_LENGTH/3, 0]])
comb_cp = np.concatenate([post_cp, curve_cp])
backbone = Backbone(comb_cp, reparameterize=True)
xy = backbone.r(np.linspace(0,1,100))

# Find midpoint of curve
samples = backbone.r(np.linspace(0,1,1000))
s = np.sin
midpoint = np.array([radius*s(backbone_angle/2), radius*c(backbone_angle/2)-radius, 0])
dists = np.linalg.norm(samples-midpoint, axis=1)
idx = dists.argmin()
mid_t = np.linspace(0,1,1000)[idx]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1)
# axs.plot(backbone_cp[:,0], backbone_cp[:,1], "b.", linewidth=10)
# axs.plot(post_cp[:,0], post_cp[:,1], "g.")
axs.plot(xy[:,0], xy[:,1], "r-", linewidth=10)
axs.plot(comb_cp[:,0], comb_cp[:,1], "k.")

axs.set_aspect("equal")
plt.show()


i = 4
radius = RADIUS_LIST[i]
cp_radius = fit_radius(radius, NUM_CP_PER_CROSS_SECTION)
th = np.linspace(0, 2*np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1,1)
cs_cp = np.hstack((cp_radius*np.cos(th), cp_radius*np.sin(th)))
post_radius = fit_radius(5, NUM_CP_PER_CROSS_SECTION)
post_cp = np.hstack((post_radius*np.cos(th), post_radius*np.sin(th)))
curve_t = np.flip(np.linspace(1,1-2*(1-mid_t),20, endpoint=False))
cs_list_curve = [CrossSection(controlpoints=cs_cp, position=position) for position in curve_t]
cs_list_post = [CrossSection(controlpoints=post_cp, position=position) for position in np.linspace(0,(1-2*(1-mid_t))*1/2,3, endpoint=True)]
cs_list = [*cs_list_post,*cs_list_curve]
# Axial Component
ac = AxialComponent(backbone, cs_list, smooth_with_post=False)
ac.mesh.show()


post_begin = 1-2*(1-mid_t)
new_t = np.linspace(0,1,20)
new_cs_curve = [CrossSection(controlpoints=cs_cp, position=position) for position in new_t[new_t>=post_begin]]
new_cs_post = [CrossSection(controlpoints=post_cp, position=position) for position in new_t[new_t<post_begin]]
new_ac = AxialComponent(backbone, [*new_cs_post, *new_cs_curve], smooth_with_post=False)
new_ac.mesh.show(smooth=False)

fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(ac.controlpoints.shape[0]):
    ax.plot(ac.controlpoints[i,:,0],ac.controlpoints[i,:,1],ac.controlpoints[i,:,2],"b-")
plt.show()


## Fuse to interface

# Generate ac
new_t = np.linspace(0,1,20)
cs_curve = [CrossSection(controlpoints=cs_cp, position=position) for position in new_t]
backbone_cp = approximate_arc(backbone_angle, LENGTH)
backbone = Backbone(backbone_cp, reparameterize=True)
ac = AxialComponent(backbone, cs_curve, smooth_with_post=False)
ac.mesh.show(smooth=False)

# Shift ac so that furthest vertex along -x axis are at origin
old_verts = ac.mesh.vertices
bottommost_vert = old_verts[old_verts[:,1].argmin()]
new_verts = old_verts - bottommost_vert

# Add in interface



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(new_verts[:,0],new_verts[:,1], new_verts[:,2], "k.")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# plt.show()
# # Backbone Angles / Curvature
# for j in range(len(BACKBONE_ANGLE_LIST)):
#     j = 4
#     backbone_angle = BACKBONE_ANGLE_LIST[j]
#     backbone_cp = approximate_arc(backbone_angle, LENGTH)
#     c = np.cos
#     s = np.sin
#     R = np.array([[c(backbone_angle/2), s(backbone_angle/2), 0],[-s(backbone_angle/2), c(backbone_angle/2), 0],[0,0,1]])
#     backbone_R = backbone_cp @ R
#     backbone = Backbone(backbone_R, reparameterize=True)

#     # Radii
#     for i in range(len(RADIUS_LIST)):
#         radius = RADIUS_LIST[i]
#         cp_radius = fit_radius(radius, NUM_CP_PER_CROSS_SECTION)
#         th = np.linspace(0, 2*np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1,1)
#         cs_cp = np.hstack((cp_radius*np.cos(th), cp_radius*np.sin(th)))
#         cs_list = [CrossSection(controlpoints=cs_cp, position=position) for position in np.linspace(0.1, 1,20)]

#         # Axial Component
#         ac = AxialComponent(backbone, cs_list, smooth_with_post=True)
#         ac.mesh.show()
#         break
#     break