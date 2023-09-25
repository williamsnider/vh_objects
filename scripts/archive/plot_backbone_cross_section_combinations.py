# Test that the backbones listed in objects.components produce correct shapes

from objects.components import (
    backbone_flat,
    backbone_weak_curve,
    backbone_strong_curve,
    backbone_sharp_bend,
    backbone_hook_f,
    backbone_hook_r,
    backbone_s,
    cp_round_high,
    cp_concave_high,
    cp_plane,
    cp_convex,
    cp_elliptical,
)
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.parameters import BACKBONE_LENGTH
import matplotlib.pyplot as plt
import numpy as np
from objects.parameters import ORDER, BACKBONE_LENGTH
from splipy import BSplineBasis, Curve
from pathlib import Path
from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
from pathlib import Path

# Inputs
base_dir = Path(
    r"C:\Users\William\Files\OConnor\Code\Projects\objects\sample_shapes\backbone_cross_section_combinations_1"
)


# Create base cross section
c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [0.1, 0.1],
    ]
)

base_cp_round = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
    ]
)

backbone_list = [
    backbone_flat,
    backbone_weak_curve,
    backbone_strong_curve,
    backbone_sharp_bend,
    backbone_hook_f,
    backbone_hook_r,
    backbone_s,
]

cs_list = [
    cp_concave_high,
    cp_plane,
    cp_round_high,
    cp_convex,
    cp_round_low,
    cp_elliptical,
]

cs_dict = {
    "concave_high": cp_concave_high,
    "plane": cp_plane,
    "round_high": cp_round_high,
    "convex": cp_convex,
    "elliptical": cp_elliptical,
}

rotation_dict = {
    0: "0_degrees",
    np.pi / 2: "90_degrees",
    np.pi: "180_degrees",
    3 * np.pi / 2: "270_degrees",
}


def render_combination(backbone, cs_names, pattern, rotation):

    # Construct save_dir
    rotation_name = rotation_dict[rotation]
    save_dir = Path(
        base_dir,
        rotation_name,
        pattern,
    )
    if save_dir.is_dir() is False:
        save_dir.mkdir(parents=True)

    # Generate list of cross sections
    cs = [cs_dict[name] for name in cs_names]
    if pattern == "A":
        assert len(cs) == 1
        cs_list = [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.9, 10)]
    elif pattern == "AB":
        assert len(cs) == 2
        cs_list = [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.1 / 0.9 * 4, 5)] + [
            CrossSection(cs[1], i, rotation=rotation) for i in np.linspace(0.1 / 0.9 * 5, 0.9, 5)
        ]
    elif pattern == "ABB":
        assert len(cs) == 3
        assert cs_names[1] == cs_names[2]
        cs_list = (
            [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.3, 3)]
            + [CrossSection(cs[1], i, rotation=rotation) for i in np.linspace(0.4, 0.6, 3)]
            + [CrossSection(cs[2], i, rotation=rotation) for i in np.linspace(0.7, 0.9, 3)]
        )
    elif pattern == "BAB":
        assert len(cs) == 3
        assert cs_names[0] == cs_names[2]
        cs_list = (
            [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.3, 3)]
            + [CrossSection(cs[1], i, rotation=rotation) for i in np.linspace(0.4, 0.6, 3)]
            + [CrossSection(cs[2], i, rotation=rotation) for i in np.linspace(0.7, 0.9, 3)]
        )
    elif pattern == "BBA":
        assert len(cs) == 3
        assert cs_names[0] == cs_names[1]
        cs_list = (
            [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.3, 3)]
            + [CrossSection(cs[1], i, rotation=rotation) for i in np.linspace(0.4, 0.6, 3)]
            + [CrossSection(cs[2], i, rotation=rotation) for i in np.linspace(0.7, 0.9, 3)]
        )
    else:
        raise NotImplementedError

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

    # Plot cross sections
    plot_name = "\n".join(cs_names)
    axs[1].set_title(plot_name)
    for i, cross_section in enumerate(cs):
        # Make curve
        degree = ORDER - 1
        num_cp_per_cross_section = cross_section.shape[0]
        num_knots = num_cp_per_cross_section + ORDER + degree
        knot = np.linspace(0, 1, num_knots)
        basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
        curve = Curve(basis1, controlpoints=cross_section, rational=False)

        # Sample curve
        curve.reparam()
        t = np.linspace(0, 1, 1000)
        x, y = curve(t).T
        x += i * 60  # Shift to right
        axs[1].set_ylim([-40, 40])
        axs[1].plot(x, y, "b-", linewidth=10)
        axs[1].set_aspect("equal")
        axs[1].tick_params(
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
    ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
    s = Shape([ac])

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
    color = s.save_mesh_as_png(save_dir="dummy", return_img=True, rotation=R)
    color = np.flip(color, axis=1)  # Reverse y axis so that the render aligns with the above plots
    axs[2].imshow(color / 2 ** 16)  # Convert to range (0,1)
    axs[2].tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    axs[2].set_title(rotation_name)

    # Export
    filename = "_".join([pattern, backbone.name, *cs_names])
    savename = Path(save_dir, filename).with_suffix(".png")
    fig.suptitle(savename.parts[-3:])
    fig.savefig(savename)
    plt.close(fig)


# Construct combinations
combinations = []

for rotation in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:

    # A pattern
    for b in backbone_list:

        # Skip redundant - flat backbone same regardless of rotation
        if rotation != 0 and b == backbone_flat:
            continue

        for c in cs_dict.keys():

            # Skip redundant - round cs same regardless of rotation
            if rotation != 0 and (c == "round_high" or c == "round_low"):
                continue

            # Skip redundant - elliptical cs same at 0/180 and 90/270deg rotations
            if c == "elliptical" and (rotation == np.pi or rotation == 3 * np.pi / 2):
                continue

            backbone = b
            cs_names = [c]
            pattern = "A"
            combinations.append([backbone, cs_names, pattern, rotation])

    # # AB pattern
    # for b in backbone_list:

    #     # Skip redundant - flat backbone same regardless of rotation
    #     if rotation != 0 and b == backbone_flat:
    #         continue

    #     for c in cs_dict.keys():

    #         # Skip redundant - A and B cannot both be round_high
    #         if c == "round_high":
    #             continue

    #         # Skip redundant - round cs same regardless of rotation
    #         if rotation != 0 and (c == "round_high" or c == "round_low"):
    #             continue

    #         # Skip redundant - elliptical cs same at 0/180 and 90/270deg rotations
    #         if c == "elliptical" and (rotation == np.pi or rotation == 3 * np.pi / 2):
    #             continue

    #         backbone = b
    #         cs_names = [c, "round_high"]
    #         pattern = "AB"
    #         combinations.append([backbone, cs_names, pattern, rotation])

    #         backbone = b
    #         cs_names = ["round_high", c]
    #         pattern = "AB"
    #         combinations.append([backbone, cs_names, pattern, rotation])

    # ABB pattern
    for b in backbone_list:

        # Skip redundant - flat backbone same regardless of rotation
        if rotation != 0 and b == backbone_flat:
            continue

        for c in cs_dict.keys():

            # Skip redundant - A and B cannot both be round_high
            if c == "round_high":
                continue

            # Skip redundant - round cs same regardless of rotation
            if rotation != 0 and (c == "round_high" or c == "round_low"):
                continue

            # Skip redundant - elliptical cs same at 0/180 and 90/270deg rotations
            if c == "elliptical" and (rotation == np.pi or rotation == 3 * np.pi / 2):
                continue

            backbone = b
            cs_names = [c, "round_high", "round_high"]
            pattern = "ABB"
            combinations.append([backbone, cs_names, pattern, rotation])

    # ABA pattern
    for b in backbone_list:

        # Skip redundant - flat backbone same regardless of rotation
        if rotation != 0 and b == backbone_flat:
            continue

        for c in cs_dict.keys():

            # Skip redundant - A and B cannot both be round_high
            if c == "round_high":
                continue

            # Skip redundant - round cs same regardless of rotation
            if rotation != 0 and (c == "round_high" or c == "round_low"):
                continue

            # Skip redundant - elliptical cs same at 0/180 and 90/270deg rotations
            if c == "elliptical" and (rotation == np.pi or rotation == 3 * np.pi / 2):
                continue

            backbone = b
            cs_names = ["round_high", c, "round_high"]
            pattern = "BAB"
            combinations.append([backbone, cs_names, pattern, rotation])

    # AAB pattern
    for b in backbone_list:

        # Skip redundant - flat backbone same regardless of rotation
        if rotation != 0 and b == backbone_flat:
            continue

        for c in cs_dict.keys():

            # Skip redundant - A and B cannot both be round_high
            if c == "round_high":
                continue

            # Skip redundant - round cs same regardless of rotation
            if rotation != 0 and (c == "round_high" or c == "round_low"):
                continue

            # Skip redundant - elliptical cs same at 0/180 and 90/270deg rotations
            if c == "elliptical" and (rotation == np.pi or rotation == 3 * np.pi / 2):
                continue

            backbone = b
            cs_names = ["round_high", "round_high", c]
            pattern = "BBA"
            combinations.append([backbone, cs_names, pattern, rotation])

# for c in combinations:
#     render_combination(*c)


def export_stl(backbone, cs_names, pattern, rotation):

    # Generate list of cross sections
    cs = [cs_dict[name] for name in cs_names]
    if pattern == "A":
        assert len(cs) == 1
        cs_list = [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.9, 10)]
    elif pattern == "AB":
        assert len(cs) == 2
        cs_list = [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.1 / 0.9 * 4, 5)] + [
            CrossSection(cs[1], i, rotation=rotation) for i in np.linspace(0.1 / 0.9 * 5, 0.9, 5)
        ]
    elif pattern == "ABB":
        assert len(cs) == 3
        assert cs_names[1] == cs_names[2]
        cs_list = (
            [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.3, 3)]
            + [CrossSection(cs[1], i, rotation=rotation) for i in np.linspace(0.4, 0.6, 3)]
            + [CrossSection(cs[2], i, rotation=rotation) for i in np.linspace(0.7, 0.9, 3)]
        )
    elif pattern == "BAB":
        assert len(cs) == 3
        assert cs_names[0] == cs_names[2]
        cs_list = (
            [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.3, 3)]
            + [CrossSection(cs[1], i, rotation=rotation) for i in np.linspace(0.4, 0.6, 3)]
            + [CrossSection(cs[2], i, rotation=rotation) for i in np.linspace(0.7, 0.9, 3)]
        )
    elif pattern == "BBA":
        assert len(cs) == 3
        assert cs_names[0] == cs_names[1]
        cs_list = (
            [CrossSection(cs[0], i, rotation=rotation) for i in np.linspace(0.1, 0.3, 3)]
            + [CrossSection(cs[1], i, rotation=rotation) for i in np.linspace(0.4, 0.6, 3)]
            + [CrossSection(cs[2], i, rotation=rotation) for i in np.linspace(0.7, 0.9, 3)]
        )
    else:
        raise NotImplementedError

    ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
    s = Shape([ac])
    filename = "_".join([pattern, backbone.name, *cs_names])
    s.label = filename

    # Construct save_dir
    save_dir = Path(
        base_dir,
        "sample_shapes",
    )
    if save_dir.is_dir() is False:
        save_dir.mkdir(parents=True)
    s.export_stl(save_dir)


# Sample shapes to print
samples = [
    [backbone_weak_curve, ["concave_high"], "A", np.pi / 2],
    [backbone_weak_curve, ["round_high"], "A", np.pi / 2],
    [backbone_weak_curve, ["convex"], "A", np.pi / 2],
    [backbone_flat, ["convex"], "A", 0],
    [backbone_weak_curve, ["convex"], "A", 0],
    [backbone_strong_curve, ["convex"], "A", 0],
    [backbone_s, ["concave_high", "round_high", "round_high"], "ABB", 0],
    [backbone_s, ["round_high", "concave_high", "round_high"], "BAB", 0],
    [backbone_s, ["round_high", "round_high", "concave_high"], "BBA", 0],
]

for c in samples:
    export_stl(*c)
