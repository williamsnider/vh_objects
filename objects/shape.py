import trimesh
import numpy as np
import matplotlib.pyplot as plt
from objects.utilities import fuse_meshes, calc_R_euler_angles
from objects.parameters import (
    HARMONIC_POWER,
    FAIRING_DISTANCE,
    INTERFACE_PATH,
    POST_SECTIONS,
    POST_OFFSET,
    POST_RADIUS,
    POST_LENGTH,
    POST_FAIRING_DISTANCE,
)
from objects.interface import Interface
from pathlib import Path
import pyrender
import copy
import cv2


class Shape:
    def __init__(self, ac_list, align_OBB=False, fuse_to_interface=False, label="test", save_dir=None):

        self.ac_list = ac_list
        self.label = label
        self.fuse_to_interface = fuse_to_interface
        self.mesh = None

        # Fuse axial components meshes
        self.combine_meshes([ac.mesh for ac in ac_list], operation="union")

        # # Find oriented bounding box
        # if align_OBB is True:
        #     self.align_mesh()

        # # Smoothly fuse to interface
        # if fuse_to_interface is True:
        #     self.fuse_mesh_to_interface()

    def check_inputs(self):

        assert type(self.ac_list) is list, "ac_list must be a list, even if it has just 1 ac."

    def combine_meshes(self, meshes_to_fuse, operation="union"):
        """Joins meshes together by iterating through the list of meshes."""
        mesh_list = copy.copy(meshes_to_fuse)
        mesh1 = mesh_list.pop(0)

        while len(mesh_list) != 0:

            mesh2 = mesh_list.pop(0)
            mesh1 = fuse_meshes(mesh1, mesh2, FAIRING_DISTANCE, operation=operation)
            # mesh1, mesh2 = check_and_move_identical_verts(mesh1, mesh2)  # Overlapping vertices cause boolean problems
            # union_mesh, edge_verts_indices = calc_mesh_boolean_and_edges(mesh1, mesh2, operation)

            # # # Debug
            # # broken = trimesh.repair.broken_faces(union_mesh, color=[255, 0, 0, 255])
            # # union_mesh.show()
            # # union_mesh.fill_holes()
            # # union_mesh.show()

            # neighbors = find_neighbors(union_mesh, edge_verts_indices, distance=FAIRING_DISTANCE)
            # if union_mesh.is_watertight is False:
            #     import warnings

            #     warnings.warn(
            #         "union_mesh is not watertight. This is probably because there is a complex joining of two axial components. This could be solves by slightly shifting the vertices near the broken faces and attempting the union again."
            #     )
            #     print("Mesh will not be faired, as it is not watertight.")
            # else:
            #     mesh1 = fair_mesh(union_mesh, neighbors, HARMONIC_POWER)  # Rename the joint mesh to mesh1 so other ACs can be added

        self.mesh = mesh1

        # ac_list = copy.copy(self.ac_list)
        # mesh1 = ac_list.pop(0).mesh  # Take first AC

        # while len(ac_list) != 0:

        #     mesh2 = ac_list.pop(0).mesh

        #     mesh1, mesh2 = check_and_move_identical_verts(mesh1, mesh2)  # Overlapping vertices cause boolean problems
        #     union_mesh, edge_verts_indices = calc_mesh_boolean_and_edges(mesh1, mesh2, operation)

        #     # TODO: Implement a method that slightly shifts the vertices on mesh1 and mesh2 that are near the union_mesh vertices that have broken faces, and retry boolean union. This is likely to work, as evidenced by small changes in the euler angles on the joining of the two meshes usually works.

        #     # # Debug
        #     # broken = trimesh.repair.broken_faces(union_mesh, color=[255, 0, 0, 255])
        #     # union_mesh.show()
        #     # union_mesh.fill_holes()
        #     # union_mesh.show()

        #     neighbors = find_neighbors(union_mesh, edge_verts_indices, distance=FAIRING_DISTANCE)
        #     if union_mesh.is_watertight is False:
        #         import warnings

        #         warnings.warn(
        #             "union_mesh is not watertight. This is probably because there is a complex joining of two axial components. This could be solves by slightly shifting the vertices near the broken faces and attempting the union again."
        #         )
        #         print("Mesh will not be faired, as it is not watertight.")
        #     else:
        #         mesh1 = fair_mesh(union_mesh, neighbors)  # Rename the joint mesh to mesh1 so other ACs can be added

        # self.mesh = mesh1

    def align_mesh(self):
        """Rotate the mesh so that it's bbox is optimal for haptic shelving."""

        # Find arbitrarily aligned bounding box
        origin = np.array([0, 0, 0, 0])
        T, extents = trimesh.bounds.oriented_bounds(self.mesh, angle_digits=0)
        self.mesh.apply_transform(T)  # Aligns bbox to axes and centers bbox at origin

        # Shift mesh so that bounding box corner is at origin
        self.mesh.apply_translation(self.mesh.bounds[1])

        # Find plane that is closest to point onf mesh
        fusion_idx = 0  # TODO: will fail for joint meshes at the base
        point_on_mesh = self.mesh.vertices[fusion_idx]
        corners = trimesh.bounds.corners(self.mesh.bounds)
        sides = np.array(
            [
                [0, 1, 2, 3],
                [0, 3, 4, 7],
                [0, 1, 4, 5],
                [1, 2, 5, 6],
                [2, 3, 6, 7],
                [4, 5, 6, 7],
            ]
        )
        corner_neighbors = {
            0: [1, 3, 4],
            1: [0, 2, 5],
            2: [1, 3, 6],
            3: [0, 2, 7],
            4: [0, 5, 7],
            5: [1, 4, 6],
            6: [2, 5, 6],
            7: [3, 4, 6],
        }
        plane_verts = corners[sides[:, :3]]
        plane_normals = np.cross(
            plane_verts[:, 0, :] - plane_verts[:, 1, :],
            plane_verts[:, 0, :] - plane_verts[:, 2, :],
        )
        plane_normals = plane_normals / np.linalg.norm(plane_normals, axis=1, keepdims=True)
        vec_from_plane_to_point = point_on_mesh - corners[sides[:, 0]]
        distance_to_planes = np.abs(np.dot(vec_from_plane_to_point, plane_normals.T)).diagonal()
        closest_side = np.argmin(distance_to_planes)

        # Find 2 corners on plane that are closest to point on mesh
        corner_verts = corners[sides[closest_side]]
        distance_to_corners = np.linalg.norm(corner_verts - point_on_mesh, axis=1)
        closest_corners = np.argsort(distance_to_corners)  # ON THE CLOSEST PLANE

        # Use these corners to assign a coordinate system
        short_edge = corner_verts[closest_corners[1]] - corner_verts[closest_corners[0]]
        middle_edge = corner_verts[closest_corners[2]] - corner_verts[closest_corners[0]]
        corners_in_plane = set(sides[closest_side])
        corners_neighboring_closest_corner = set(corner_neighbors[sides[closest_side][closest_corners[0]]])
        other_neighbor = (corners_neighboring_closest_corner - corners_in_plane).pop()
        long_edge = corners[other_neighbor] - corner_verts[closest_corners[0]]
        curr = np.stack(
            [
                short_edge,
                middle_edge,
                long_edge,
            ],
            axis=0,
        )
        curr = curr / np.linalg.norm(curr, axis=0)

        goal = np.array(
            [
                [0, 0, 1],  # Short edge should point in +Z direction
                [0, 1, 0],
                [1, 0, 0],  # Long edge should point in +X direction
            ]
        )

        R = np.linalg.inv(curr) @ goal  # Rotation matrix

        # Perform transformations
        self.mesh.vertices -= corner_verts[closest_corners[0]]  # Translate corner to origin
        self.mesh.vertices = self.mesh.vertices @ R  # Rotate
        assert np.all(self.mesh.bounds[0, :] == 0), "Corner not aligned at 0."

        # Center short edge on origin
        shift_to_Z = self.mesh.bounds[1, 2] / 2
        self.mesh.vertices[:, 2] -= shift_to_Z

        # Fix normals (make sure they're pointing out)
        trimesh.repair.fix_inversion(self.mesh)

    def create_interface(self):
        """Creates a mesh of the interface."""

        self.interface = Interface(INTERFACE_PATH, self.label)

    def fuse_mesh_to_interface(self):

        # Create a post that connects the interface and shape
        post = trimesh.creation.cylinder(POST_RADIUS, POST_LENGTH + POST_OFFSET, sections=POST_SECTIONS)

        # Subdivide post to improve fairing
        post = post.subdivide()

        # Transform post so that it is intersecting both shape and interface
        R = calc_R_euler_angles([np.pi / 2, 0, 0])
        goal_position = np.array([0, -(POST_LENGTH + POST_OFFSET / 2) / 2, 0])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = goal_position
        post = post.apply_transform(T)
        self.post = post

        # Fuse interface, post, and shape
        interface_and_post = fuse_meshes(self.interface.mesh, self.post, fairing_distance=0, operation="union")
        interface_post_and_shape = fuse_meshes(
            interface_and_post, self.mesh, fairing_distance=POST_FAIRING_DISTANCE, operation="union"
        )
        self.mesh = interface_post_and_shape

    def construct_scene(self):

        # Bounds
        bounds = self.mesh.bounds
        bounds_pts = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
        bounds = trimesh.points.PointCloud(bounds_pts)

        # Axes
        axis = trimesh.creation.axis(origin_size=1)

        if self.fuse_to_interface is True:
            self.scene = trimesh.Scene([self.interface, self.mesh, bounds, axis])
        else:
            self.scene = trimesh.Scene([self.mesh, bounds, axis])

    def export_stl(self, save_dir):

        # Convert to Path class
        if type(save_dir) == str:
            save_dir = Path(save_dir)

        # Construct save_dir
        if save_dir.is_dir() is False:
            save_dir.mkdir(parents=True)

        # Export
        filename = Path(save_dir, self.label).with_suffix(".stl")
        self.mesh.export(filename)

    def export_png(self, save_dir):

        self.construct_scene()

        # Convert to Path class
        if type(save_dir) == str:
            save_dir = Path(save_dir)

        # Construct save_dir
        if save_dir.is_dir() is False:
            save_dir.mkdir()

        # Export
        filename = Path(save_dir, self.label).with_suffix(".png")
        resolution = (1920, 1080)
        png = self.scene.save_image(resolution=resolution)  # bytes
        with open(filename, "wb") as f:
            f.write(png)

    def save_mesh_as_png(self, save_dir, return_img=False, rotation=None):
        """
        Saves the mesh as a png.
        """
        # Convert to Path class
        if type(save_dir) == str:
            save_dir = Path(save_dir)

        # Construct save_dir
        if return_img is False:
            if save_dir.is_dir() is False:
                save_dir.mkdir(parents=True)

        filename = str(Path(save_dir, self.label).with_suffix(".png"))

        # Compose scene
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.3], bg_color=[1, 1, 1])

        # Add mesh and interface
        mesh_pose = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        if rotation is not None:
            mesh_pose = mesh_pose @ rotation

        # Add mesh to scene
        mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)
        scene.add(mesh, pose=mesh_pose)

        # Add interface to scene
        if self.fuse_to_interface is True:
            interface = pyrender.Mesh.from_trimesh(self.interface, smooth=False)
            scene.add(interface, pose=mesh_pose)

        # Add directional light
        light_pose = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5e3)
        scene.add(light, pose=light_pose)

        # Set camera pose parameters
        # With these settings, the resulting image lines up with our shape coordinate system:
        # Image left = shape +X
        # Image up = shape +Y
        # Out of image (Towards viewer) = shape +Z  (check this one)
        u = np.array([-1, 0, 0])
        v = np.array([0, 1, 0])
        n = np.cross(u, v)
        # e = np.array([0, 0, -150])  #  eye: camera position in world coordinates
        e = np.array([35, 0, -80])  #  eye: camera position in world coordinates
        camera_pose = np.array(
            [
                [u[0], u[1], u[2], e[0]],
                [v[0], v[1], v[2], e[1]],
                [n[0], n[1], n[2], e[2]],
                [0, 0, 0, 1],
            ]
        )
        # # Set camera pose parameters such that camera lies on positive z-axis looking towards the origin
        # u = np.array([1, 0, 0])  # up vector
        # n = np.array([0, 1, 0])  # view direction; opposite vector of camera's "view"
        # v = np.cross(u, n)
        # e = np.array([0, 30, 0])  #  eye: camera position in world coordinates
        # camera_pose = np.array(
        #
        #         [u[0], v[0], -n[0], -e[0]],
        #         [u[1], v[1], -n[1], -e[1]],
        #         [u[2], v[2], -n[2], -e[2]],
        #         [0, 0, 0, 1],
        #     ]
        # )

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)

        # TODO: This is not 16 bit depth
        r = pyrender.OffscreenRenderer(7680, 4320, bitdepth="16bit")
        color, _ = r.render(scene)

        if return_img is True:
            return color
        else:
            # Save png - at 16bit depth
            cv2.imwrite(filename, color)

    def plot_meshes(self):

        trimesh.Scene([ac.mesh for ac in self.ac_list]).show()
