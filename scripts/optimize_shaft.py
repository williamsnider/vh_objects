# Create axial component
import numpy as np

import copy
import numpy as np
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.utilities import (
    approximate_arc,
    make_mesh,
    make_surface,
    calc_hemisphere_controlpoints,
    angle_between,
    fuse_meshes,
    calc_mesh_boolean_and_edges,
)
from objects.parameters import INTERFACE_PATH, INTERFACE_SHIFT
from objects.interface import load_interface
from scripts.sheets import construct_sheet, bend_sheet, make_base_cp
from scripts.hemi import calc_sphere_controlpoints
import trimesh
from scripts.sheets import plot_arr
from scipy.spatial.transform.rotation import Rotation
from pathlib import Path
from scipy.optimize import minimize, basinhopping

# Change backbone length until desired length is reached

NUM_CS = 11
NUM_CP_PER_CROSS_SECTION = 50
NUM_CP_PER_BACKBONE = 5


def make_ac(backbone_length, r1, r2, r3):
    pos = np.linspace(0, 1, NUM_CS)
    pos_app = np.linspace(0, backbone_length, NUM_CS)

    b_cp = np.hstack(
        [
            np.linspace(0, backbone_length, NUM_CP_PER_BACKBONE).reshape(-1, 1),
            np.zeros((NUM_CP_PER_BACKBONE, 1)),
            np.zeros((NUM_CP_PER_BACKBONE, 1)),
        ]
    )
    b_appendage_K0 = Backbone(b_cp, reparameterize=True)

    t = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(
        -1, 1
    )
    round_cp = np.hstack([np.cos(t), np.sin(t)])

    x = np.linspace(0, 1, 3) * backbone_length
    # y = np.array([0.5 * X_WIDTH, 0.5 * X_WIDTH, 1.0 * X_WIDTH])
    # y = np.array([0.5 * X_WIDTH, 1 * X_WIDTH, 0.1 * X_WIDTH])
    y = np.array([r1, r2, r3])
    # y = np.array([0.5 * X_WIDTH, 0.5 * X_WIDTH, 0.1 * X_WIDTH])

    poly = np.polyfit(x, y, 2)
    scale = np.polyval(poly, pos_app)
    cs_list = [
        CrossSection(controlpoints=round_cp * scale[i], position=pos[i])
        for i in range(NUM_CS)
    ]
    ac = AxialComponent(
        b_appendage_K0,
        cs_list,
        smooth_with_post=False,
        hemispherical_ends=True,
        hemispherical_polynomial=poly,
        hemisphere_x=[pos_app[0], pos_app[-1]],
    )
    return ac


def objective_function(*inputs):
    backbone_length, DESIRED_LENGTH, r1, r2, r3 = inputs

    ac = make_ac(backbone_length[0], r1, r2, r3)
    length = ac.mesh.extents[0]
    # print(length)

    return (length - DESIRED_LENGTH) ** 2


def optimize_backbone_length(DESIRED_LENGTH, r1, r2, r3):

    print("Optimizing backbone length...")

    # Find close best guess
    NUM_GUESSES = 20
    guesses = np.linspace(0.00001, DESIRED_LENGTH, NUM_GUESSES)
    guess_result = np.zeros(NUM_GUESSES)
    for i in range(NUM_GUESSES):
        guess_result[i] = objective_function([guesses[i]], DESIRED_LENGTH, r1, r2, r3)
    x0 = guesses[guess_result.argmin()]
    print("best guess: {}".format(x0))

    res = minimize(
        objective_function,
        x0=x0,
        args=(DESIRED_LENGTH, r1, r2, r3),
        bounds=[(0.0001, DESIRED_LENGTH)],
    )

    # res = basinhopping(objective_function, DESIRED_LENGTH)
    backbone_length = res.x[0]
    if res.fun > 1:
        print("Failed to find good solution")
        return None
    else:
        # ac = make_ac(backbone_length, r1, r2, r3)
        # ac.mesh.show()
        return backbone_length


if __name__ == "__main__":
    DESIRED_LENGTH = 22.5
    X_WIDTH = 6
    r1, r2, r3 = np.array([0.5 * X_WIDTH, 0.5 * X_WIDTH, 0.5 * X_WIDTH])
    optimize_backbone_length(DESIRED_LENGTH, r1, r2, r3)


# if res.success == True:
#     return res.x
# else:
#     return None
