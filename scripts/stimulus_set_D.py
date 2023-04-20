import copy
import numpy as np
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shaft import Shaft
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
import scipy
import pyrender
import cv2

### Parameters ###
NUM_CP_PER_BACKBONE = 5
SEGMENT_LENGTH = 40
NUM_CS = 11
X_WIDTH = 4.25  # base radius off which other features are derived.
NUM_CP_PER_CROSS_SECTION = 50

# Limb sizes
ac_radii = np.array([0.1 * X_WIDTH, 1 * X_WIDTH, 2 * X_WIDTH])
ac_theta_dict = {"th0": 0, "th1": np.pi / 2}
ac_junc_angles = {"ja0": 0, "ja1": np.pi / 4, "ja2": np.pi / 2}
ac_junc_rotations = {"r0": 0, "r1": np.pi}
SPHERE_ALIGNMENT_OFFSET = np.array(
    [0.25, 0.1, 0.1]
)  # Instead of perfectly aligning the left and right spheres of limb1 and limb2, shift limb2 slightly, which is necessary for the boolean union to work. Because these shifts are small (<0.25mm), this should be OK.

# Post and interface
POST_RADIUS = ac_radii[1] * 0.99
POST_OFFSET = 2  # Post and interface

# Saving png and stl
SAVE_DIR = Path("./sample_shapes/stimulus_set_D/")

#################################
### Axial Components / Shafts ###
#################################

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

##############
### Shapes ###
##############


class Shape:
    """Shape is the combination of multiple meshes into one watertight mesh. Moreover, an interface and post can be added."""

    def __init__(
        self,
        mesh_list,
        T_list,
        label,
        description,
        save_dir,
        T_final=np.eye(4),
        fairing_distance=0,
        post_z_shift=0,
    ):
        self.mesh_list = mesh_list
        self.T_list = T_list
        self.label = label
        self.description = description
        self.T_final = T_final
        self.save_dir = save_dir
        self.fairing_distance = fairing_distance
        self.post_z_shift = post_z_shift

        self.combine_meshes()
        self.attach_interface()
        self.export_stl()

        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", np.array([-np.pi / 2, 0, 0])).as_matrix()
        self.save_mesh_as_png(
            rotation=T,
            interface=True,
        )

    def combine_meshes(self):

        # Transform meshes
        new_mesh_list = []
        for i, mesh in enumerate(self.mesh_list):
            mesh = mesh.copy()
            new_mesh_list.append(mesh.apply_transform(self.T_list[i]))

        # Fuse meshes
        meshA = new_mesh_list[0]
        for meshB in new_mesh_list[1:]:
            meshA = fuse_meshes(meshA, meshB, self.fairing_distance, "union")

        # Apply T_final
        self.mesh = meshA.apply_transform(self.T_final)

    def attach_interface(self):

        # Attach post
        post_backbone_cp = np.hstack(
            [
                np.linspace(
                    POST_OFFSET, INTERFACE_SHIFT - POST_OFFSET, NUM_CP_PER_BACKBONE
                ).reshape(-1, 1),
                np.zeros((NUM_CP_PER_BACKBONE, 1)),
                np.zeros((NUM_CP_PER_BACKBONE, 1)),
            ]
        )
        post_backbone = Backbone(post_backbone_cp, reparameterize=True)
        post_th = np.linspace(
            0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False
        ).reshape(-1, 1)
        post_cp = np.hstack(
            (POST_RADIUS * np.cos(post_th), POST_RADIUS * np.sin(post_th))
        )
        post_cs_list = [
            CrossSection(controlpoints=post_cp, position=pos)
            for pos in [0.0, 0.01, 0.99, 1.0]
        ]
        post_ac = AxialComponent(post_backbone, post_cs_list, smooth_with_post=False)

        # Shift post in z direction (to improve alignment with shape)
        post_ac.mesh = post_ac.mesh.apply_translation([0, 0, self.post_z_shift])

        meshA = self.mesh.copy()
        meshA = fuse_meshes(meshA, post_ac.mesh, 2, "union")

        # Attach interface
        label = str(self.label).zfill(4)
        interface = load_interface(INTERFACE_PATH, label)
        mesh_with_interface = fuse_meshes(meshA, interface, 0, "union")

        self.mesh_with_interface = mesh_with_interface

    def export_stl(self):

        SAVE_DIR = Path(self.save_dir, "stl")

        # Construct save_dir
        if SAVE_DIR.is_dir() is False:
            SAVE_DIR.mkdir(parents=True)

        # Export
        filename = Path(SAVE_DIR, self.label).with_suffix(".stl")
        self.mesh_with_interface.export(filename)

    def save_mesh_as_png(
        self,
        return_img=False,
        rotation=None,
        resolution=(1920, 1080),
        interface=False,
    ):
        """
        Saves the mesh as a png.
        """
        # Convert to Path class
        save_dir = Path(self.save_dir, "png")

        if type(save_dir) == str:
            save_dir = Path(save_dir)

        # Construct save_dir
        if return_img is False:
            if save_dir.exists() is False:
                save_dir.mkdir(parents=True)

        filename = str(Path(save_dir, self.label).with_suffix(".png"))

        # Compose scene
        scene = pyrender.Scene(ambient_light=[0.1, 0.5, 0.3], bg_color=[1, 1, 1])

        # Add mesh to scene
        if rotation is not None:
            mesh_pose = rotation
        else:
            mesh_pose = np.eye(4)
        if interface == True:
            mesh = pyrender.Mesh.from_trimesh(self.mesh_with_interface, smooth=False)
        else:
            mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
        scene.add(mesh, pose=mesh_pose)

        # Camera pose explained:
        # +X axis is towards the right of the screen
        # +Y axis is towards the top of the screen
        # -Z axis points into the screen (camera looks into the screen)
        yfov = np.pi / 4.0
        ywidth = 120  # mm
        camera_pose = np.eye(4)
        camera_pose[2, 3] = (
            ywidth / 2 / np.tan(yfov / 2)
        )  # Calculate correct distance for camera to have ywidth and yfov
        camera = pyrender.PerspectiveCamera(yfov=yfov)
        scene.add(camera, pose=camera_pose)

        # Add directional light
        light_euler = np.array([-np.pi / 4, 0, 0])
        R = scipy.spatial.transform.Rotation.from_euler("xyz", light_euler).as_matrix()
        light_pose = np.eye(4)
        light_pose[:3, :3] = R
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5e3)
        scene.add(light, pose=light_pose)

        r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, _ = r.render(scene)

        if return_img is True:
            return color
        else:
            cv2.imwrite(filename, color)


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
        label = str(count)
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
        label = str(count)
        description = "-".join([L1_name, junc_rotation_name, junc_angle_name, L2_name])

    # Transform final shape to align correctly with interface
    T_final = np.eye(4)

    if "th0" in L1_name:
        T_final[:3, :3] = Rotation.from_euler(
            "xyz", np.array([np.pi / 2, 0, 0])
        ).as_matrix()

    # Rotate shapes with a curved Limb1 (endpoint of L1 will be at midpoint in Z-dimension which is most efficient for storage on shelves)
    elif "th1" in L1_name:
        T_final[:3, :3] = Rotation.from_euler(
            "xyz", np.array([np.pi / 2, np.pi / 4, 0])
        ).as_matrix()
    else:
        raise NotImplementedError

    # Shift down post for curved limb1s
    if "s_th1" == description[:5]:
        post_z_shift = -X_WIDTH * 0.7
    else:
        post_z_shift = 0

    s = Shape(
        mesh_list,
        T_list,
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


from multiprocessing import Pool
from tqdm import tqdm

# for i, comb in enumerate(combs[:]):
#     construct_shapes(comb)

with Pool() as pool:
    mapped_values = list(
        tqdm(pool.imap_unordered(construct_shapes, combs), total=len(combs))
    )
