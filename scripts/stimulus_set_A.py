import numpy as np
from objects.shaft import Shaft
from objects.shape import Shape
from scipy.spatial.transform.rotation import Rotation
from multiprocessing import Pool
from tqdm import tqdm

from scripts.stimulus_set_params import (
    SEGMENT_LENGTH,
    NUM_CS,
    X_WIDTH,
    NUM_CP_PER_CROSS_SECTION,
    ac_radii,
    ac_theta_dict,
    ac_junc_angles,
    ac_junc_rotations,
    SPHERE_ALIGNMENT_OFFSET,
    SAVE_DIR,
)


#########################################
### Shafts for Limb1 and Limb2 Shapes ###
#########################################

# Construct dict containing the different shafts that serve as Limb1 and Limb2
shaft_dict = {}
for theta_name in ac_theta_dict.keys():
    for r1_name in ["1"]:
        for r2_name in ["1", "2"]:
            for r3_name in ["0", "1", "2"]:

                s_name = "_".join(["s", theta_name, r1_name, r2_name, r3_name])
                r1 = ac_radii[int(r1_name)]
                r2 = ac_radii[int(r2_name)]
                r3 = ac_radii[int(r3_name)]
                theta = ac_theta_dict[theta_name]

                shaft_dict[s_name] = Shaft(
                    SEGMENT_LENGTH,
                    r1,
                    r2,
                    r3,
                    theta,
                    lengthtype="two_hemi",  # Length takes into account both spherical ends
                    num_cs=NUM_CS,
                    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
                )

##############################
### Limb1 and Limb2 Shapes ###
##############################


def construct_shapes(inputs):

    # Get inputs
    (
        L1_name,
        L2_name,
        junc_rotation_name,
        junc_angle_name,
        count,
        fairing_distance,
        SAVE_DIR,
    ) = inputs

    # Load shafts
    shaft1 = shaft_dict[L1_name].copy()
    shaft1.mesh.visual.vertex_colors = np.array([255, 255, 0, 50])

    # Handle single limb case
    if L2_name == None:
        mesh_list = [shaft1.mesh]
        T_list = [np.eye(4)]
        description = "-".join([L1_name])
    else:
        shaft2 = shaft_dict[L2_name].copy()
        shaft2.mesh.visual.vertex_colors = np.array([255, 0, 255, 75])

        # Align center of sphere to origin
        TA = np.eye(4)
        TA[:3, 3] = (
            -shaft2.l_sphere_origin + SPHERE_ALIGNMENT_OFFSET
        )  # Improves success of boolean to slightly misalign

        # Rotate according to junction rotation and angle
        TB = np.eye(4)
        if junc_rotation_name == "r0":
            TB[:3, :3] = Rotation.from_euler(
                "xyz",
                np.array(
                    [
                        ac_junc_rotations[junc_rotation_name],
                        0,
                        -ac_junc_angles[junc_angle_name],
                    ]
                ),
            ).as_matrix()
        elif junc_rotation_name == "r1":
            TB[:3, :3] = Rotation.from_euler(
                "xyz",
                np.array(
                    [
                        ac_junc_rotations[junc_rotation_name],
                        0,
                        -ac_junc_angles[junc_angle_name],
                    ]
                ),
            ).as_matrix()  # Flip junc_angle sign to keep (choose shape to be longer)
        else:
            raise NotImplementedError

        # Align to r sphere of shaft1
        TC = shaft1.get_T(1.0)
        TC[:3, 3] = shaft1.r_sphere_origin
        T = TC @ (TB @ TA)

        mesh_list = [shaft1.mesh, shaft2.mesh]
        T_list = [np.eye(4), T]
        description = "-".join([L1_name, junc_rotation_name, junc_angle_name, L2_name])

    label = "A" + str(count).zfill(3)

    # Transform final shape to align correctly with interface
    T_final = np.eye(4)

    if "th0" in L1_name:
        T_final[:3, :3] = Rotation.from_euler(
            "xyz", np.array([-np.pi / 2, 0, 0])
        ).as_matrix()

    # Rotate shapes with a curved Limb1 (endpoint of L1 will be at midpoint in Z-dimension which is most efficient for storage on shelves)
    elif "th1" in L1_name:
        T_final[:3, :3] = Rotation.from_euler(
            "xyz", np.array([-np.pi / 2, -np.pi / 4, 0])
        ).as_matrix()
    else:
        raise NotImplementedError

    # Shift up shapes for all limb1s
    # if "s_th1" == description[:5]:
    all_z_shift = -2
    T_final[2, 3] = all_z_shift

    # Shift down post for curved limb1s
    if "s_th1" == description[:5]:
        post_z_shift = +X_WIDTH * 0.6 + all_z_shift
    else:
        post_z_shift = all_z_shift

    boolean_list = ["union" for _ in mesh_list]
    s = Shape(
        mesh_list,
        T_list,
        boolean_list,
        label,
        description,
        SAVE_DIR,
        T_final=T_final,
        fairing_distance=fairing_distance,
        post_z_shift=post_z_shift,
    )

    return s


count = 0
combs = []
# Single limb
for L1_name in [
    "s_th0_1_1_0",
    "s_th0_1_1_1",
    "s_th0_1_1_2",
    "s_th0_1_2_0",
    "s_th0_1_2_1",
    "s_th0_1_2_2",
    "s_th1_1_1_0",
    "s_th1_1_1_1",
    "s_th1_1_1_2",
    "s_th1_1_2_0",
    "s_th1_1_2_1",
    "s_th1_1_2_2",
]:
    L2_name = None
    combs.append(
        [
            L1_name,
            L2_name,
            0,
            0,
            count,
            0,
            SAVE_DIR,
        ]
    )
    count += 1

# Double limb

for L1_name in [
    "s_th0_1_1_1",
    "s_th0_1_2_1",
    "s_th1_1_1_1",
    "s_th1_1_2_1",
]:
    for L2_name in [
        "s_th0_1_1_0",
        "s_th0_1_1_1",
        "s_th0_1_1_2",
        "s_th0_1_2_0",
        "s_th0_1_2_1",
        "s_th0_1_2_2",
        "s_th1_1_1_0",
        "s_th1_1_1_1",
        "s_th1_1_1_2",
        "s_th1_1_2_0",
        "s_th1_1_2_1",
        "s_th1_1_2_2",
    ]:
        for junc_rotation_name, junc_rotation in ac_junc_rotations.items():

            # Skip shapes with straight L1 and nonzero rotation since these rotations are redundant
            if ("th0" in L1_name) and (junc_rotation_name != "r0"):
                continue

            # Skip shapes with straight L2 in second rotation since these rotations are redudant
            if ("th0" in L2_name) and (junc_rotation_name != "r0"):
                continue

            for junc_angle_name, junc_angle in ac_junc_angles.items():

                # Set fairing distance differently for straight-straight and other shapes
                if "th0" in L1_name and "th0" in L2_name and junc_angle_name == "ja0":
                    fairing_distance = 1
                else:
                    fairing_distance = 3

                combs.append(
                    [
                        L1_name,
                        L2_name,
                        junc_rotation_name,
                        junc_angle_name,
                        count,
                        fairing_distance,
                        SAVE_DIR,
                    ]
                )
                count += 1


for i, comb in enumerate(combs[:]):
    construct_shapes(comb)

# a = construct_shapes(combs[1])

if __name__ == "__main__":

    with Pool() as pool:
        mapped_values = list(
            tqdm(pool.imap_unordered(construct_shapes, combs), total=len(combs))
        )
