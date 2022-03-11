# Steps to constructions shapes, rendering png and forming stl
from objects.components import (
    backbone_flat,
    backbone_weak_curve,
    backbone_strong_curve,
    backbone_sharp_bend,
    backbone_hook_f,
    backbone_hook_r,
    backbone_s,
    cp_round,
    cp_concave_high,
    cp_plane,
    cp_convex,
    cp_elliptical,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

# Inputs
base_dir = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set")
backbone_list = [
    backbone_flat,
    backbone_weak_curve,
    backbone_strong_curve,
    backbone_sharp_bend,
    backbone_hook_f,
    backbone_hook_r,
    backbone_s,
]

# Controlpoints used for cross sections
cp_list = [cp_round, cp_concave_high, cp_plane, cp_convex, cp_elliptical]

##########################
### Shape Construction ###
##########################
shapes = []

# Varying backbones
for backbone in backbone_list:
    cp = cp_round
    rotation = 0
    cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 10)]
    ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
    s = Shape([ac])
    shapes.append(s)

# Varying cross sections
# #TODO: rotations?
for backbone in [backbone_flat, backbone_weak_curve]:
    for cp in cp_list:
        rotation = 0
        cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 10)]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac])
        shapes.append(s)

# Blending cross sections
# #TODO: rotations?
for cp_a in [cp_round, cp_elliptical]:
    for cp_b in cp_list:

        # Skip when cp_a and cp_b are the same cross section, as its already formed above
        if cp_a is cp_b:
            continue

        # Blend in both directions (i.e. shape AB and shape BA)
        for i in range(2):
            if i == 1:
                cp_a, cp_b = cp_b, cp_a  # Flip order

            backbone = backbone_flat
            rotation = 0
            cs_list = [CrossSection(cp_a, 0.1, rotation=rotation), CrossSection(cp_b, 0.9, rotation=rotation)]
            ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
            s = Shape([ac])
            shapes.append(s)

# Varying cross section relative sizes
sizes = np.array(
    [
        [0.5, 1, 1.5],  # linear increase (cone)
        [1.5, 1, 0.5],  # linear decrease (cone)
        [0.5, 1, 2],  # quadratic increase (flare)
        [2, 1, 0.5],  # quadratic decrease (flare)
        [0.5, 1.5, 0.5],  # spindle
        [1.5, 0.5, 1.5],  # hourglass
    ]
)
for size in sizes:
    for cp in [cp_round]:
        backbone = backbone_flat
        rotation = 0
        cs_list = [
            CrossSection(cp * size[0], 0.1, rotation=rotation),
            CrossSection(cp * size[1], 0.5, rotation=rotation),
            CrossSection(cp * size[2], 0.9, rotation=rotation),
        ]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac])
        shapes.append(s)

now = datetime.now()
timestamp = now.strftime("%d_%m_%y__%H_%M_%S")
pickle_name = Path(base_dir, "stimulus_set", "stimulus_set", timestamp + ".pickle")
with open(pickle_name, "wb") as f:
    pickle.dump(shapes, f)
# pickle.dump(shapes, open(""))
