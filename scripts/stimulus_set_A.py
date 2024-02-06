import numpy as np
from objects.shaft import Shaft
from objects.shape import Shape
from scipy.spatial.transform.rotation import Rotation
from multiprocessing import Pool
from tqdm import tqdm
from scripts.stimulus_set_common import mesh_dict
from pathlib import Path
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
from scipy.spatial.transform.rotation import Rotation
import trimesh

##############
### Params ###
##############

SAVE_DIR = Path(SAVE_DIR, "stimulus_set_A")

def construct_shapes(inputs):
    """Constructs shapes for stimulus set A."""

    # Get inputs
    (
        L1_name,
        L2_name,
        junc_rotation_name,
        junc_angle_name,
        count,
        mesh_fairing_distance,
        SAVE_DIR,
    ) = inputs

    # Load shafts
    L1 = mesh_dict[L1_name].copy()
    L1.visual.vertex_colors = np.array([255, 255, 0, 50])
    L1_T = np.eye(4)
    if "K1" in L1_name:
        L1_T[:3,:3] = Rotation.from_euler("zyx", np.array([0, np.pi/4, 0])).as_matrix()
    else:
        L1_T[:3,:3] = Rotation.from_euler("zyx", np.array([0, np.pi/2, 0])).as_matrix()
    # L1.apply_transform(L1_T)

    # Handle single limb case
    from scripts.stimulus_set_common import A_APPENDAGE_LENGTH
    if L2_name == None:
        mesh_list = [L1]
        T_list = [L1_T]
        description = "-".join([L1_name])
    else:
        L2 = mesh_dict[L2_name].copy()

        L2.apply_scale(0.999) # Shrink L2 slightly to avoid boolean union issues
        L2.visual.vertex_colors = np.array([255, 0, 255, 50])
        L2_T = np.eye(4)

        if (junc_rotation_name == "r0") and ("_K1" in L1_name):
            L2_T[:3,:3] = Rotation.from_euler(
                "zyx",
                np.array(
                    [
                        ac_junc_rotations[junc_rotation_name],
                        -ac_junc_angles[junc_angle_name]+3*np.pi/4,
                        0,
                    ]
                ),
            ).as_matrix()
        elif (junc_rotation_name == "r1") and ("_K1" in L2_name):
            L2_T[:3,:3] = Rotation.from_euler(
                "zyx",
                np.array(
                    [
                        ac_junc_rotations[junc_rotation_name],
                        -ac_junc_angles[junc_angle_name]+np.pi,
                        0,
                    ]
                ),
            ).as_matrix()
        else:
            L2_T[:3,:3] = Rotation.from_euler(
                "zyx",
                np.array(
                    [
                        ac_junc_rotations[junc_rotation_name],
                        -ac_junc_angles[junc_angle_name]+np.pi/2,
                        0,
                    ]
                ),
            ).as_matrix()

        # else:

        #     L2_T[:3,:3] = Rotation.from_euler(
        #         "zyx",
        #         np.array(
        #             [
        #                 ac_junc_rotations[junc_rotation_name],
        #                 -ac_junc_angles[junc_angle_name]+np.pi/2,
        #                 0,
        #             ]
        #         ),
        #     ).as_matrix()
        
        if "_K1" in L1_name:
            SHIFT = 6.75
        else:
            SHIFT = 2.5
        L2_T[:3, 3] = [A_APPENDAGE_LENGTH - SHIFT, 0, 0]

        mesh_list = [L1, L2]
        T_list = [L1_T, L2_T]
        description = "-".join([L1_name, junc_rotation_name, junc_angle_name, L2_name])

    label = "A" + str(count).zfill(3)

    # Transform final shape to align correctly with interface
    T_final = np.eye(4)


    # Set fairing distance for shape and post.
    if L1_name == "app0":
        post_fairing_distance = 0.25  # Small fairing distance for straight-straight shapes
    else:
        post_fairing_distance = 2

    boolean_list = ["union" for _ in mesh_list]
    s  = Shape(mesh_list, T_list, boolean_list, label, description, SAVE_DIR, T_final=T_final, mesh_fairing_distance=mesh_fairing_distance, post_fairing_distance=post_fairing_distance, post_z_shift=0)
    # s.mesh.show(smooth=False)
    # s.mesh.color = [255, 0, 0, 255]


    


####################
### Combinations ###
####################

count = 0
combs = []

# # Single limb
# for L1_name in [  "app0",
#     "app0_K1",
#     "app1",
#     "app1_K1",
#     "app2", 
#     "app2_K1",
#     "app3",
#     "app3_K1",
#     "app4",
#     "app4_K1",
#     "app5",
#     "app5_K1",

# ]:
#     L2_name = None
#     junc_rotation_name = "r0"
#     junc_angle_name = "ja1"
#     mesh_fairing_distance = 0
#     inputs = [L1_name, L2_name, junc_rotation_name, junc_angle_name, count, mesh_fairing_distance, SAVE_DIR]
    
#     s = construct_shapes(inputs)
#     combs.append(inputs)
#     count += 1
 

# Double limb 
for L1_name in ["app0", "app0_K1", "app3", "app3_K1"]:

    for L2_name in ["app0", "app0_K1", "app1", "app1_K1", "app2","app2_K1", "app3","app3_K1", "app4", "app4_K1", "app5", "app5_K1"]:

        for junc_rotation_name, junc_rotation in ac_junc_rotations.items():

            # Skip shapes with straight L1 and nonzero rotation since these rotations are redundant
            if ("_K1" not in L1_name) and (junc_rotation_name != "r0"):
                continue

            # Skip shapes with straight L2 in second rotation since these rotations are redudant
            if ("_K1" not in L2_name) and (junc_rotation_name != "r0"):
                continue

            
            for junc_angle_name, junc_angle in ac_junc_angles.items():

                # Set fairing distance differently for straight-straight and other shapes
                if ("_K1" not in L1_name) and ("_K1" not in L2_name) and (junc_angle_name == "ja0"):
                    mesh_fairing_distance = 1
                elif (L1_name == "app0") and ("_K1" in L2_name) and (junc_angle_name == "ja0"):
                    mesh_fairing_distance = 1
                else:
                    mesh_fairing_distance = 3#3

                inputs =   [
                        L1_name,
                        L2_name,
                        junc_rotation_name,
                        junc_angle_name,
                        count,
                        mesh_fairing_distance,
                        SAVE_DIR,
                    ]
                combs.append(inputs
                )    

                count += 1


if __name__ == "__main__":

    # with Pool() as pool:
    #     mapped_values = list(
    #         tqdm(pool.imap_unordered(construct_shapes, combs), total=len(combs))
    #     )

    for i, comb in enumerate(combs[:]):

        construct_shapes(comb)
        print("Finished {} of {}".format( str(i).zfill(3), len(combs)))
##############################
### Debugging Example ###
##############################

# # Get inputs
# (
#     L1_name,
#     L2_name,
#     junc_rotation_name,
#     junc_angle_name,
#     count,
#     mesh_fairing_distance,
#     SAVE_DIR,
# ) = inputs

# L1_name = "app3"
# L2_name = "app2_K1"
# junc_rotation_name = "r0"
# junc_angle_name = "ja1"
# count = 0
# mesh_fairing_distance = 6
# SAVE_DIR = Path("./sample_shapes/stimulus_set_A/")

# inputs = [L1_name, L2_name, junc_rotation_name, junc_angle_name, count, mesh_fairing_distance, SAVE_DIR]
# construct_shapes(inputs)
        
#         # L2.apply_transform(L2_T)
# # Double limb

# for L1_name in [
#     "s_th0_1_1_1",
#     "s_th0_1_2_1",
#     "s_th1_1_1_1",
#     "s_th1_1_2_1",
# ]:
#     for L2_name in [
#         "s_th0_1_1_0",
#         "s_th0_1_1_1",
#         "s_th0_1_1_2",
#         "s_th0_1_2_0",
#         "s_th0_1_2_1",
#         "s_th0_1_2_2",
#         "s_th1_1_1_0",
#         "s_th1_1_1_1",
#         "s_th1_1_1_2",
#         "s_th1_1_2_0",
#         "s_th1_1_2_1",
#         "s_th1_1_2_2",
#     ]:
#         for junc_rotation_name, junc_rotation in ac_junc_rotations.items():

#             # Skip shapes with straight L1 and nonzero rotation since these rotations are redundant
#             if ("_K1" not in L1_name) and (junc_rotation_name != "r0"):
#                 continue

#             # Skip shapes with straight L2 in second rotation since these rotations are redudant
#             if ("_K1" not in L2_name) and (junc_rotation_name != "r0"):
#                 continue

#             for junc_angle_name, junc_angle in ac_junc_angles.items():

#                 # Set fairing distance differently for straight-straight and other shapes
#                 if "_K1" not in L1_name and "_K1" not in L2_name and junc_angle_name == "ja0":
#                     mesh_fairing_distance = 1
#                 else:
#                     mesh_fairing_distance = 3

#                 combs.append(
#                     [
#                         L1_name,
#                         L2_name,
#                         junc_rotation_name,
#                         junc_angle_name,
#                         count,
#                         mesh_fairing_distance,
#                         SAVE_DIR,
#                     ]
#                 )
#                 count += 1


#         # L2.apply_transform(L2_T)

#     # scene = trimesh.Scene()
#     # scene.add_geometry(L1)
#     # scene.add_geometry(L2)

#     # from objects.interface import load_interface
#     # from objects.parameters import INTERFACE_PATH
#     # label = "1111"
#     # interface = load_interface(INTERFACE_PATH, label)
#     # scene.add_geometry(interface)

#     # scene.show()

#         # # Align center of sphere to origin
#         # TA = np.eye(4)
#         # TA[:3, 3] = (
#         #
# # Load shafts
# L1 = mesh_dict[L1_name].copy()
# L1.visual.vertex_colors = np.array([255, 255, 0, 50])
# L1_T = np.eye(4)
# if "K1" in L1_name:
#     L1_T[:3,:3] = Rotation.from_euler("zyx", np.array([0, np.pi/4, 0])).as_matrix()
# else:
#     L1_T[:3,:3] = Rotation.from_euler("zyx", np.array([0, np.pi/2, 0])).as_matrix()
# # L1.apply_transform(L1_T)

# # Handle single limb case
# from scripts.stimulus_set_params import APPENDAGE_LENGTH
# if L2_name == None:
#     mesh_list = [L1.mesh]
#     T_list = [L1_T]
#     description = "-".join([L1_name])
# else:
#     L2 = mesh_dict[L2_name].copy()
#     L2.visual.vertex_colors = np.array([255, 0, 255, 50])
#     L2_T = np.eye(4)
#     L2_T[:3,:3] = Rotation.from_euler(
#             "zyx",
#             np.array(
#                 [
#                     ac_junc_rotations[junc_rotation_name],
#                     -ac_junc_angles[junc_angle_name]+np.pi/2,
#                     0,
#                 ]
#             ),
#         ).as_matrix()
#     L2_T[:3, 3] = [APPENDAGE_LENGTH, 0, 0]
#     # L2.apply_transform(L2_T)

# # scene = trimesh.Scene()
# # scene.add_geometry(L1)
# # scene.add_geometry(L2)

# # from objects.interface import load_interface
# # from objects.parameters import INTERFACE_PATH
# # label = "1111"
# # interface = load_interface(INTERFACE_PATH, label)
# # scene.add_geometry(interface)

# # scene.show()

#     # # Align center of sphere to origin
#     # TA = np.eye(4)
#     # TA[:3, 3] = (
#     #     -L2.l_sphere_origin + SPHERE_ALIGNMENT_OFFSET
#     # )  # Improves success of boolean to slightly misalign

#     # # Rotate according to junction rotation and angle
#     # TB = np.eye(4)
#     # if junc_rotation_name == "r0":
#     #     TB[:3, :3] = Rotation.from_euler(
#     #         "xyz",
#     #         np.array(
#     #             [
#     #                 ac_junc_rotations[junc_rotation_name],
#     #                 0,
#     #                 -ac_junc_angles[junc_angle_name],
#     #             ]
#     #         ),
#     #     ).as_matrix()
#     # elif junc_rotation_name == "r1":
#     #     TB[:3, :3] = Rotation.from_euler(
#     #         "xyz",
#     #         np.array(
#     #             [
#     #                 ac_junc_rotations[junc_rotation_name],
#     #                 0,
#     #                 -ac_junc_angles[junc_angle_name],
#     #             ]
#     #         ),
#     #     ).as_matrix()  # Flip junc_angle sign to keep (choose shape to be longer)
#     # else:
#     #     raise NotImplementedError

#     # # Align to r sphere of L1
#     # TC = L1.get_T(1.0)
#     # TC[:3, 3] = L1.r_sphere_origin
#     # T = TC @ (TB @ TA)

# mesh_list = [L1, L2]
# T_list = [L1_T, L2_T]
# description = "-".join([L1_name, junc_rotation_name, junc_angle_name, L2_name])

# label = "A" + str(count).zfill(3)

# # Transform final shape to align correctly with interface
# T_final = np.eye(4)

# boolean_list = ["union" for _ in mesh_list]
# s  = Shape(mesh_list, T_list, boolean_list, label, description, SAVE_DIR, T_final=T_final, mesh_fairing_distance=mesh_fairing_distance, post_z_shift=0)
# s.mesh.show(smooth=False)
# s.mesh.color = [255, 0, 0, 255]
# # if "_K1" not in L1_name:
# #     T_final[:3, :3] = Rotation.from_euler(
# #         "xyz", np.array([-np.pi / 2, 0, 0])
# #     ).as_matrix()

# # # Rotate shapes with a curved Limb1 (endpoint of L1 will be at midpoint in Z-dimension which is most efficient for storage on shelves)
# # elif "th1" in L1_name:
# #     T_final[:3, :3] = Rotation.from_euler(
# #         "xyz", np.array([-np.pi / 2, -np.pi / 4, 0])
# #     ).as_matrix()
# # else:
# #     raise NotImplementedError

# # Shift up shapes for all limb1s
# # if "s_th1" == description[:5]:
# # all_z_shift = -2
# # T_final[2, 3] = all_z_shift

# # # Shift down post for curved limb1s
# # if "s_th1" == description[:5]:
# #     post_z_shift = +X_WIDTH * 0.6 + all_z_shift
# # else:
# #     post_z_shift = all_z_shift

# # boolean_list = ["union" for _ in mesh_list]
# # s = Shape(
# #     mesh_list,
# #     T_list,
# #     boolean_list,
# #     label,
# #     description,
# #     SAVE_DIR,
# #     T_final=T_final,
# #     mesh_fairing_distance=mesh_fairing_distance,
# #     post_z_shift=post_z_shift,
# # )





# for i, comb in enumerate(combs[:]):
#     construct_shapes(comb)

# # a = construct_shapes(combs[1])

# #########################################
# ### Shafts for Limb1 and Limb2 Shapes ###
# #########################################

# # Construct dict containing the different shafts that serve as Limb1 and Limb2
# mesh_dict = {}
# for theta_name in ac_theta_dict.keys():
#     for r1_name in ["1"]:
#         for r2_name in ["1", "2"]:
#             for r3_name in ["0", "1", "2"]:

#                 s_name = "_".join(["s", theta_name, r1_name, r2_name, r3_name])
#                 r1 = ac_radii[int(r1_name)]
#                 r2 = ac_radii[int(r2_name)]
#                 r3 = ac_radii[int(r3_name)]
#                 theta = ac_theta_dict[theta_name]

#                 mesh_dict[s_name] = Shaft(
#                     SEGMENT_LENGTH,
#                     r1,
#                     r2,
#                     r3,
#                     theta,
#                     lengthtype="two_hemi",  # Length takes into account both spherical ends
#                     num_cs=NUM_CS,
#                     num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
#                 )
