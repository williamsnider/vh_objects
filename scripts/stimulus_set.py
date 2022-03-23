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
    cp_concave,
    cp_plane,
    cp_convex,
    cp_elliptical,
    sd_cylinder,
    sd_cylinder_spherical_top,
    sd_curved_cylinder,
    sd_sphere,
    sd_ellipsoid,
    sd_cone,
    sd_curved_elliptical_cylinder,
    sd_elliptical_cylinder,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.parameters import BACKBONE_LENGTH, ORDER
from objects.utilities import transform_sd_mesh
from splipy import BSplineBasis, Curve
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy

##################
### Parameters ###
##################
set_version = "Z"  # Typically a single letter
base_dir = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set", set_version)

# # Backbones used
# backbone_list = [
#     backbone_flat,
#     backbone_weak_curve,
#     backbone_strong_curve,
#     backbone_sharp_bend,
#     backbone_hook_f,
#     backbone_hook_r,
#     backbone_s,
# ]

# # Controlpoints used for cross sections
# cp_list = [
#     cp_round,
#     cp_concave,
#     cp_plane,
#     cp_convex,
#     cp_elliptical,
# ]

# # Surface deformations used
# sd_list = [
#     sd_cylinder,
#     sd_cylinder_spherical_top,
#     sd_curved_cylinder,
#     sd_sphere,
#     sd_ellipsoid,
#     sd_cone,
# ]

# Dictionary linking names and arrays (helps with constructing description of each shape)
d = {
    "b_flat": (backbone_flat),
    "b_weak_curve": (backbone_weak_curve),
    "b_strong_curve": (backbone_strong_curve),
    "b_sharp_bend": (backbone_sharp_bend),
    "b_hook_f": (backbone_hook_f),
    "b_hook_r": (backbone_hook_r),
    "b_s": (backbone_s),
    "cs_round": (cp_round),
    "cs_concave": (cp_concave),
    "cs_plane": (cp_plane),
    "cs_convex": (cp_convex),
    "cs_elliptical": (cp_elliptical),
    "sd_cylinder": (sd_cylinder),
    "sd_cylinder_spherical_top": (sd_cylinder_spherical_top),
    "sd_curved_cylinder": (sd_curved_cylinder),
    "sd_sphere": (sd_sphere),
    "sd_ellipsoid": (sd_ellipsoid),
    "sd_cone": (sd_cone),
}
##########################
### Shape Construction ###
##########################
shapes = []

# Varying backbones
for b_name in ["b_flat", "b_weak_curve", "b_strong_curve", "b_sharp_bend", "b_hook_f", "b_hook_r"]:
    cs_name = "cs_round"
    backbone = d[b_name]
    cp = d[cs_name]
    rotation = 0
    cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 10)]
    ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
    s = Shape([ac])
    s.description = "{0}-{1}".format(b_name, cs_name)
    s.cs_name = [cs_name]
    s.sd_name = None
    shapes.append(s)

# Varying cross sections
# #TODO: rotations?
for b_name in ["b_flat", "b_weak_curve"]:
    for cs_name in ["cs_round", "cs_concave", "cs_plane", "cs_convex", "cs_elliptical"]:

        if b_name == "b_flat" and cs_name == "cs_round":
            continue

        backbone = d[b_name]
        cp = d[cs_name]
        rotation = 0
        cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 10)]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac])
        s.description = "{0}-{1}".format(b_name, cs_name)
        s.cs_name = [cs_name]
        s.sd_name = None
        shapes.append(s)

# Blending cross sections
# #TODO: rotations?
for cs_name_a in ["cs_round", "cs_elliptical"]:
    for cs_name_b in ["cs_round", "cs_concave", "cs_plane", "cs_convex", "cs_elliptical"]:

        # Skip when cp_a and cp_b are the same cross section, as its already formed above
        if cs_name_a == cs_name_b:
            continue

        # Do not duplicate cs_elliptical + cs_round
        if cs_name_a == "cs_elliptical" and cs_name_b == "cs_round":
            continue

        b_name = "b_flat"
        cp_a = d[cs_name_a]
        cp_b = d[cs_name_b]

        # Blend in both directions (i.e. shape AB)
        backbone = d[b_name]
        rotation = 0
        cs_list = [CrossSection(cp_a, 0.1, rotation=rotation), CrossSection(cp_b, 0.9, rotation=rotation)]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac])
        s.description = "{0}-{1}-{2}".format(b_name, cs_name_a, cs_name_b)
        s.cs_name = [cs_name_a, cs_name_b]
        s.sd_name = None
        shapes.append(s)

        # Blend in both directions (i.e. shape BA)
        backbone = backbone_flat
        rotation = 0
        cs_list = [CrossSection(cp_b, 0.1, rotation=rotation), CrossSection(cp_a, 0.9, rotation=rotation)]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac])
        s.description = "{0}-{1}-{2}".format(b_name, cs_name_b, cs_name_a)
        s.cs_name = [cs_name_b, cs_name_a]
        s.sd_name = None
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
    for cs_name in ["cs_round"]:
        b_name = "b_flat"
        backbone = d[b_name]
        cp = d[cs_name]
        rotation = 0
        cs_list = [
            CrossSection(cp * size[0], 0.1, rotation=rotation),
            CrossSection(cp * size[1], 0.5, rotation=rotation),
            CrossSection(cp * size[2], 0.9, rotation=rotation),
        ]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac])
        size_as_str = [str(int(s * 100)) + "%" for s in size]
        s.description = "{0}-{1}-{2}".format(b_name, cs_name, "-".join(size_as_str))
        s.cs_name = [cs_name]
        s.sd_name = None
        shapes.append(s)

# Varying surface deformations
for b_name in ["b_flat"]:
    backbone = d[b_name]
    for cs_name in ["cs_round"]:

        cp = d[cs_name]

        # Generate base shape
        rotation = 0
        cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 9)]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        base_shape = Shape([ac])

        # Apply surface deformations
        for sd_name in [
            "sd_sphere",
            "sd_ellipsoid",
            "sd_cylinder",
            "sd_cylinder_spherical_top",
            "sd_curved_cylinder",
            "sd_cone",
        ]:

            sd = d[sd_name]

            # Rotations around round axis of axial component
            for theta_backbone in np.linspace(0, 2 * np.pi, 4, endpoint=False):

                if theta_backbone != 0 and backbone == backbone_flat:
                    continue

                # Rotations about vector from backbone to surface point where surface deformation is attached
                for theta_linear_segment in np.linspace(0, 2 * np.pi, 4, endpoint=False):

                    if ~np.isclose(theta_linear_segment, 0) and sd not in [
                        sd_curved_cylinder,
                        sd_ellipsoid,
                    ]:
                        continue

                    # Don't duplicate the ellipsoid with rotation
                    if sd == sd_ellipsoid:
                        pass
                    if np.any(np.isclose(theta_linear_segment, [np.pi, 3 * np.pi / 2])) and sd in [
                        sd_ellipsoid,
                    ]:
                        continue

                    # Apply "difference" operation to make concavity
                    for operation in ["union", "difference"]:
                        if operation == "difference" and sd not in [sd_sphere, sd_ellipsoid]:
                            continue

                        # Allow surface deformation to be in 1-2 positions
                        for pos_list in [[0.33], [0.66], [0.33, 0.66]]:
                            s = copy.deepcopy(base_shape)
                            for pos in pos_list:
                                sd_mesh, origin = sd
                                sd_mesh_transformed = transform_sd_mesh(
                                    sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment
                                )
                                # sd_mesh_rotations = [transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment) for theta_linear_segment in np.linspace(0, 2*np.pi, 4, endpoint=False)]
                                s.combine_meshes([s.mesh, sd_mesh_transformed], operation=operation)
                            # s.mesh.show()

                            # b_flat-cs_round-sd_sphere-rot_b_90-rot_ls_90-pos_0.33_0.66
                            if operation == "union":
                                curv = "convex"
                            elif operation == "difference":
                                curv = "concave"
                            s.description = "{0}-{1}-{2}-rot_b_{3}-rot_ls_{4}-pos_{5}_{6}".format(
                                b_name,
                                cs_name_a,
                                sd_name,
                                str(np.round(theta_backbone / np.pi * 180).astype("int")),
                                str(np.round(theta_linear_segment / np.pi * 180).astype("int")),
                                "_".join([str(p) for p in pos_list]),
                                curv,
                            )
                            s.cs_name = [cs_name]
                            s.sd_name = sd_name
                            shapes.append(s)

# Labeling the shapes
for i, s in enumerate(shapes):
    s.label = set_version + "_" + str(i).zfill(4)

# Attach interface
for i, s in enumerate(shapes):
    print(i)
    s.create_interface()
    s.fuse_mesh_to_interface()
    s.mesh.show()


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
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax0.set_title(backbone.name)

    # Plot backbone
    x, y = backbone.controlpoints[:, :2].T
    y_mean = y.mean()
    padding = 10
    t = np.linspace(0, 1, 100)
    x, y, _ = backbone.r(t).T
    ax0.plot(x, y, "b-", linewidth=10)
    ax0.set_ylim([y_mean - BACKBONE_LENGTH / 2 - padding, y_mean + BACKBONE_LENGTH / 2 + padding])
    ax0.set_aspect("equal")
    ax0.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )

    # Plot cross sections
    plot_name = "\n".join(s.cs_name)
    ax1.set_title(plot_name)
    for i, cs_name in enumerate(s.cs_name):
        cp = d[cs_name]

        # Make curve
        degree = ORDER - 1
        num_cp_per_cross_section = cp.shape[0]
        num_knots = num_cp_per_cross_section + ORDER + degree
        knot = np.linspace(0, 1, num_knots)
        basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
        curve = Curve(basis1, controlpoints=cp, rational=False)

        # Sample curve
        curve.reparam()
        t = np.linspace(0, 1, 1000)
        x, y = curve(t).T
        x += i * 60  # Shift to right
        ax1.set_ylim([-40, 40])
        ax1.plot(x, y, "b-", linewidth=10)
        ax1.set_aspect("equal")
        ax1.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

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
    color = shape.save_mesh_as_png(save_dir="", return_img=True, rotation=R)
    color = np.flip(color, axis=1)  # Reverse y axis so that the render aligns with the above plots
    ax2.imshow(color / 2**16)  # Convert to range (0,1)
    ax2.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    if s.sd_name != None:
        ax2.set_title(s.sd_name)

    # Make save_dir
    save_dir = Path(base_dir, "png")
    if save_dir.is_dir() is False:
        save_dir.mkdir(parents=True)

    filename = Path(save_dir, s.label).with_suffix(".png")
    fig.suptitle(s.label + "_" + s.description)
    fig.savefig(filename)
    # plt.show()
    plt.close(fig)
