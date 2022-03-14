# Steps to constructions shapes, rendering png and forming stl
from re import S
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
from objects.parameters import BACKBONE_LENGTH, ORDER
from splipy import BSplineBasis, Curve
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

##################
### Parameters ###
##################
set_version = "Z"  # Typically a single letter
base_dir = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set", set_version)

# Backbones used
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

        # Blend in both directions (i.e. shape AB)
        backbone = backbone_flat
        rotation = 0
        cs_list = [CrossSection(cp_a, 0.1, rotation=rotation), CrossSection(cp_b, 0.9, rotation=rotation)]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac])
        shapes.append(s)

        # Blend in both directions (i.e. shape BA)
        backbone = backbone_flat
        rotation = 0
        cs_list = [CrossSection(cp_b, 0.1, rotation=rotation), CrossSection(cp_a, 0.9, rotation=rotation)]
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

# Labeling the shapes TODO: make these more descriptive
for i, s in enumerate(shapes):
    s.label = set_version + "_" + str(i).zfill(4)


###########################
### Pickling the shapes ###
###########################

# Make base_dir directory
if base_dir.is_dir() is False:
    base_dir.mkdir(parents=True)

# Save with timestamp
now = datetime.now()
timestamp = now.strftime("%d_%m_%y__%H_%M_%S")
pickle_name = Path(base_dir, "stimulus_set_ " + timestamp + ".pickle")
with open(pickle_name, "wb") as f:
    pickle.dump(shapes, f)

# Load from pickled file
with open(pickle_name, "rb") as f:
    shapes = pickle.load(f)


###################################
### Save shapes as .stl meshses ###
###################################
for s in shapes:

    # Make save_dir
    save_dir = Path(base_dir, "stl")
    if save_dir.is_dir() is False:
        save_dir.mkdir(parents=True)

    # Export mesh
    filename = Path(save_dir, s.label).with_suffix(".stl")
    s.mesh.export(filename)

#####################################
### Save shapes as png renderings ###
#####################################
for s in shapes:

    backbone = s.ac_list[0].backbone
    ac = s.ac_list[0]
    shape = s

    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].set_title(backbone.name)

    # Plot backbone
    x, y = backbone.controlpoints[:, :2].T
    y_mean = y.mean()
    padding = 10
    t = np.linspace(0, 1, 100)
    x, y, _ = backbone.r(t).T
    axs[0].plot(x, y, "b-", linewidth=10)
    axs[0].set_ylim([y_mean - BACKBONE_LENGTH / 2 - padding, y_mean + BACKBONE_LENGTH / 2 + padding])
    axs[0].set_aspect("equal")
    axs[0].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )

    # # Plot cross sections
    # # plot_name = "\n".join(cs_names)
    # # axs[1].set_title(plot_name)
    # for i, cross_section in enumerate(cs):
    #     # Make curve
    #     degree = ORDER - 1
    #     num_cp_per_cross_section = cross_section.shape[0]
    #     num_knots = num_cp_per_cross_section + ORDER + degree
    #     knot = np.linspace(0, 1, num_knots)
    #     basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
    #     curve = Curve(basis1, controlpoints=cross_section, rational=False)

    #     # Sample curve
    #     curve.reparam()
    #     t = np.linspace(0, 1, 1000)
    #     x, y = curve(t).T
    #     x += i * 60  # Shift to right
    #     axs[1].set_ylim([-40, 40])
    #     axs[1].plot(x, y, "b-", linewidth=10)
    #     axs[1].set_aspect("equal")
    #     axs[1].tick_params(
    #         axis="both",
    #         which="both",
    #         bottom=False,
    #         top=False,
    #         left=False,
    #         right=False,
    #         labelbottom=False,
    #         labelleft=False,
    #     )

    # Plot shape

    # rotation about x-axis
    r = rotation - np.pi / 2
    R = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(r), -np.sin(r), 0],
            [0, np.sin(r), np.cos(r), 0],
            [0, 0, 0, 1],
        ]
    )
    color = shape.save_mesh_as_png(save_dir="dummy", return_img=True, rotation=R)
    color = np.flip(color, axis=1)  # Reverse y axis so that the render aligns with the above plots
    axs[2].imshow(color / 2 ** 16)  # Convert to range (0,1)
    axs[2].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    # axs[2].set_title(rotation_name)

    # Make save_dir
    save_dir = Path(base_dir, "png")
    if save_dir.is_dir() is False:
        save_dir.mkdir(parents=True)

    # Export mesh
    filename = Path(save_dir, s.label).with_suffix(".png")
    fig.suptitle(s.label)
    fig.savefig(filename)
    plt.close(fig)
