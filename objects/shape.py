import trimesh
import numpy as np
import scipy
import matplotlib.pyplot as plt
from compas_cgal.booleans import boolean_union
import igl
from objects.utilities import plot_mesh_and_specific_indices
from objects.parameters import HARMONIC_POWER, FAIRING_DISTANCE
from pathlib import Path
import pyrender
import numpngw
import copy


class Shape:
    def __init__(self, ac_list, align_OBB=False, fuse_to_interface=False, label="test", save_dir=None):

        self.ac_list = ac_list
        self.label = label
        self.fuse_to_interface = fuse_to_interface

        # Fuse meshes
        self.fuse_meshes()

        # Find oriented bounding box
        if align_OBB is True:
            self.align_mesh()

        # Smoothly fuse to interface
        if fuse_to_interface is True:
            self.fuse_mesh_to_interface()

    def check_inputs(self):

        assert type(self.ac_list) is list, "ac_list must be a list, even if it has just 1 ac."

    def fuse_meshes(self):
        def check_and_move_identical_verts(mesh1, mesh2):
            """Mesh boolean fails often when two vertices on two meshes are identical, so we need to shift one vertex very slightly."""

            # Calculate distance between two sets of vertices
            tree = scipy.spatial.KDTree(mesh1.vertices)
            dd, ii = tree.query(mesh2.vertices, k=1)
            identical_verts = np.isclose(dd, 0)

            # Shift these verts
            mesh2.vertices[identical_verts] += 1e-3  #

            # Check that this shift was successful
            dd, ii = tree.query(mesh2.vertices, k=1)
            identical_verts = np.isclose(dd, 0)
            assert np.any(identical_verts) == False, "Shifting the identical vertex did not work."

            return mesh1, mesh2

        def calc_mesh_boolean_and_edges(mesh1, mesh2):

            # Use compas/CGAL to calculate boolean union
            mesh_A = [mesh1.vertices.tolist(), mesh1.faces.tolist()]
            mesh_B = [mesh2.vertices.tolist(), mesh2.faces.tolist()]
            mesh_C = boolean_union(mesh_A, mesh_B)

            # Get edges - vertices that were in neither initial mesh
            set_A = set([tuple(l) for l in mesh_A[0]])
            set_B = set([tuple(l) for l in mesh_B[0]])
            set_C = set([tuple(l) for l in mesh_C[0]])
            new_verts = (set_C - set_A) - set_B
            edge_verts_pts = np.zeros((len(new_verts), 3))
            for i, v in enumerate(new_verts):
                edge_verts_pts[i] = list(v)

            # Return as trimesh - easier to work with
            mesh = trimesh.Trimesh(
                vertices=mesh_C[0],
                faces=mesh_C[1],
            )

            # Get indices of edge_verts
            tree = scipy.spatial.KDTree(mesh.vertices)
            _, edge_verts_indices = tree.query(edge_verts_pts, k=1)

            if mesh.is_watertight:
                print("Mesh is watertight")
            else:
                print("Mesh is NOT watertight")
            return mesh, edge_verts_indices

        def find_neighbors(mesh, group, distance):

            mesh_pts = mesh.vertices.__array__()
            edge_pts = mesh.vertices[group].__array__()

            tree = scipy.spatial.KDTree(mesh_pts)
            neighbors_list = tree.query_ball_point(edge_pts, r=distance)
            neighbors = set()
            for n in neighbors_list:
                neighbors.update(set(n))

            return list(neighbors)

        def fair_mesh(union_mesh, neighbors):

            v = union_mesh.vertices.__array__()
            f = union_mesh.faces.__array__().astype("int64")
            num_verts = v.shape[0]
            b = np.array(list(set(range(num_verts)) - set(neighbors))).astype(
                "int64"
            )  # Bounday indices - NOT to be faired
            bc = v[b]  # XYZ coordinates of the boundary indices
            z = igl.harmonic_weights(v, f, b, bc, HARMONIC_POWER)  # Smooths indices at creases

            union_mesh.vertices = z
            faired_mesh = union_mesh

            return faired_mesh

        ac_list = copy.copy(self.ac_list)
        mesh1 = ac_list.pop(0).mesh  # Take first AC

        while len(ac_list) != 0:

            mesh2 = ac_list.pop(0).mesh

            mesh1, mesh2 = check_and_move_identical_verts(mesh1, mesh2)  # Overlapping vertices cause boolean problems
            union_mesh, edge_verts_indices = calc_mesh_boolean_and_edges(mesh1, mesh2)
            neighbors = find_neighbors(union_mesh, edge_verts_indices, distance=FAIRING_DISTANCE)
            mesh1 = fair_mesh(union_mesh, neighbors)  # Rename the joint mesh to mesh1 so other ACs can be added

        self.mesh = mesh1

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

    def fuse_mesh_to_interface(self):

        interface = trimesh.load_mesh("base_interface.stl")
        self.interface = interface

        # trimesh.Scene([interface, self.mesh, bounds, axis]).show()
        # fig = plt.figure()
        # ax = plt.axes(projection="3d")
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # ax.view_init(elev=-90, azim=90)

        # x, y, z = self.mesh.vertices.T
        # ax.plot(x, y, z, "b.")

        # for x in bounds[:, 0]:
        #     for y in bounds[:, 1]:
        #         for z in bounds[:, 2]:
        #             ax.plot(x, y, z, "g.")

        # x, y, z = self.mesh.vertices[0].T
        # ax.plot(x, y, z, "r*")
        # plt.show()

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
            save_dir.mkdir()

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

    def save_mesh_as_png(self, save_dir):
        """
        Saves the mesh as a png.
        """
        # Convert to Path class
        if type(save_dir) == str:
            save_dir = Path(save_dir)

        # Construct save_dir
        if save_dir.is_dir() is False:
            save_dir.mkdir()

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
        e = np.array([0, 0, -100])  #  eye: camera position in world coordinates
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
        #     [
        #         [u[0], v[0], -n[0], -e[0]],
        #         [u[1], v[1], -n[1], -e[1]],
        #         [u[2], v[2], -n[2], -e[2]],
        #         [0, 0, 0, 1],
        #     ]
        # )

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)

        r = pyrender.OffscreenRenderer(7680, 4320, bitdepth="16bit")
        color, _ = r.render(scene)

        # Save png - at 16bit depth
        numpngw.write_png(filename, color)
        print(filename)

    def plot_meshes(self):

        trimesh.Scene([ac.mesh for ac in self.ac_list]).show()
