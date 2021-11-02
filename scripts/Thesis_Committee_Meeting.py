from objects.digit_segments import segment_flat, segment_arc_1_4, segment_arc_1_8
from objects.backbone_from_digits import BackboneFromDigits
from objects.backbone import Backbone
import numpy as np
import itertools
import matplotlib.pyplot as plt

######################################
### Make backbone combinations
######################################

# Segments
valid_segments = [segment_flat, segment_arc_1_4]
segment_combinations = [list(comb) for comb in itertools.product(valid_segments, repeat=4)]

# Angles
T_euler = np.linspace(0, 2 * np.pi, 4, endpoint=False)  # These rotations introduce kinks
N_euler = np.array([0])  # These rotations introduce kinks
B_euler = np.array([0])  # These rotations introduce kinks
valid_angles = [[t, n, b] for t in T_euler for n in N_euler for b in B_euler]
angle_combinations = []

# Allow only angles in MIDDLE position
for pos in [1]:
    for angle in valid_angles:
        euler_angles = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        euler_angles[pos] = angle
        angle_combinations.append(euler_angles)
angle_combinations = np.array(angle_combinations)

# Backbone combinations
backbone_list = []
for digit_segments in segment_combinations:
    for angles_between_segments in angle_combinations:
        bfd = BackboneFromDigits(digit_segments, angles_between_segments)
        backbone = Backbone(bfd.controlpoints, reparameterize=True)

        print(backbone.length())
        backbone_list.append(backbone)

# # Detect redundant
# cp_all_backbones = np.zeros((len(backbone_list), (4 * 4 + 1) * 3))
# for i, backbone in enumerate(backbone_list):
#     cp_all_backbones[i, :] = backbone.controlpoints.ravel()
# _, indices = np.unique(cp_all_backbones, axis=0, return_index=True)
# backbone_list_pruned = [backbone_list[i] for i in np.sort(indices)]


# TODO: I'm not convinced that a chain of arcs rotate properly about the tangent axis.

# Make random shape
backbone = backbone_list[45]

# Plot
t = np.linspace(0, 1, 100)
for backbone in backbone_list:

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    x, y, z = backbone.r(t).T
    ax.plot(x, y, z, "b-")
    axis_max = np.max([x, y, z])

    ax.set_xlim([-axis_max, axis_max])
    ax.set_ylim([-axis_max, axis_max])
    ax.set_zlim([-axis_max, axis_max])
    plt.show()
