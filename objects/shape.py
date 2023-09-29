import copy
import numpy as np
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.utilities import (
    fuse_meshes,
    fair_mesh,
)
from objects.parameters import INTERFACE_PATH, INTERFACE_SHIFT
from scripts.stimulus_set_params import (
    POST_OFFSET,
    NUM_CP_PER_CROSS_SECTION,
    NUM_CP_PER_BACKBONE,
    POST_RADIUS,
)
from objects.interface import load_interface
from scipy.spatial.transform.rotation import Rotation
from pathlib import Path
import scipy
import pyrender
import cv2
from multiprocessing import Pool


class Shape:
    """Shape is the combination of multiple meshes into one watertight mesh. Moreover, an interface and post can be added to make a robot-graspable 3D-object."""

    def __init__(
        self,
        mesh_list,
        T_list,
        boolean_list,
        label,
        description,
        save_dir,
        T_final=np.eye(4),
        fairing_distance=0,
        post_z_shift=0,
        fair_box=None,
    ):
        self.mesh_list = mesh_list
        self.T_list = T_list
        self.boolean_list = boolean_list
        self.label = label
        self.description = description
        self.T_final = T_final
        self.save_dir = save_dir
        self.fairing_distance = fairing_distance
        self.post_z_shift = post_z_shift
        self.fair_box = fair_box

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
        """Combine triangular meshes by fusing them together."""

        # Transform meshes
        new_mesh_list = []
        for i, mesh in enumerate(self.mesh_list):
            mesh = mesh.copy()
            new_mesh_list.append(mesh.apply_transform(self.T_list[i]))

        # Fuse meshes
        meshA = new_mesh_list[0]
        i = 1
        for meshB in new_mesh_list[1:]:
            meshA = fuse_meshes(meshA, meshB, self.fairing_distance, self.boolean_list[i])
            i += 1

        # Fair region contained in box if needed
        if self.fair_box != None:
            neighbors = np.arange(meshA.vertices.shape[0])[self.fair_box.contains(meshA.vertices)]
            meshA = fair_mesh(meshA.copy(), neighbors, harmonic_power=2)

        # Apply T_final
        self.mesh = meshA.apply_transform(self.T_final)

    def attach_interface(self):
        """Attach interface and post to the shape."""

        # Attach post
        post_backbone_cp = np.hstack(
            [
                np.linspace(POST_OFFSET, INTERFACE_SHIFT - POST_OFFSET, NUM_CP_PER_BACKBONE).reshape(-1, 1),
                np.zeros((NUM_CP_PER_BACKBONE, 1)),
                np.zeros((NUM_CP_PER_BACKBONE, 1)),
            ]
        )
        post_backbone = Backbone(post_backbone_cp, reparameterize=True)
        post_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
        post_cp = np.hstack((POST_RADIUS * np.cos(post_th), POST_RADIUS * np.sin(post_th)))
        post_cs_list = [CrossSection(controlpoints=post_cp, position=pos) for pos in [0.0, 0.01, 0.99, 1.0]]
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
        """Export mesh as stl."""

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
