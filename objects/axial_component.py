import numpy as np
from splipy import BSplineBasis, Surface
import trimesh
from objects.parameters import (
    ORDER,
    SLIDE_FACTOR,
    SHRINK_FACTOR,
    NUM_ENDPOINTS,
    NUM_ENDPOINTS_SLOPE,
    NUM_CROSS_SECTION_SLOPE,
    SAMPLING_DENSITY_U,
    SAMPLING_DENSITY_V,
)
from objects.utilities import open_uniform_knot_vector, calc_face_normals
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import copy
import open3d as o3d


class AxialComponent:
    def __init__(
        self,
        length,
        curvature,
        cross_sections,
        euler_angles=np.array([0, 0, 0]),
        parent_axial_component=None,
        position_along_parent=1.0,
        position_along_self=0.0,
    ):
        self.length = length
        self.curvature = curvature
        if self.curvature == 0:
            self.radius = None
        else:
            self.radius = 1.0 / self.curvature
        self.cross_sections = cross_sections
        # self.elevation = elevation
        # self.azimuth = azimuth
        # self.rotation = rotation
        self.euler_angles = euler_angles
        self.parent_axial_component = parent_axial_component
        self.position_along_parent = position_along_parent
        self.position_along_self = position_along_self

        # Do calculations
        self.calc_points()

    def check_inputs(self):

        assert self.length > 0, "axial_component length must be greater than 0."
        assert self.curvature >= 0, "Curvature cannot be negative."
        assert self.length * self.curvature <= 2 * np.pi, "Axial component is too curved and loops back into itself."
        assert type(self.cross_sections) is list, "cross_sections must be input as a list."

        current_position = -1
        for cs in self.cross_sections:
            position = cs.position
            assert (
                position > current_position
            ), "cross_sections must be ordered by increasing position, and these positions cannot repeat"

        assert type(self.euler_angles) is np.ndarry, "Euler angles must be input as a numpy array"

        assert self.euler_angles.shape == (
            1,
            3,
        ), "Euler angles must be in a 1x3 numpy array"

        assert np.all(self.euler_angles <= np.pi) and np.all(
            self.euler_angles >= -np.pi, "Euler angles must be between -pi and pi"
        )

    def calc_points(self):
        self.calc_transformation_matrices()
        self.calc_R_euler_angles()
        # self.calc_R_elevation_azimuth_rotation()
        self.get_controlpoints()
        self.make_surface()
        self.make_mesh()

    def calc_transformation_matrices(self):

        if self.parent_axial_component is None:

            # Initialize translation and rotationmatrices, which will be updated later
            self.translation = np.array([0, 0, 0])  # TODO: delete this?
            self.R_align_with_parent = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
            self.child_join_point = np.array([0, 0, 0])
            self.parent_join_point = np.array([0, 0, 0])
            return

        else:
            # Translation matrix
            parent_join_point = self.parent_axial_component.r(self.position_along_parent)
            child_join_point = self.r(self.position_along_self, local=True)
            self.child_join_point = child_join_point
            self.parent_join_point = parent_join_point

            # Rotation matrix
            pos = self.position_along_self
            child_basis = np.stack(
                [
                    self.T(pos, local=True)[0],
                    self.N(pos, local=True)[0],
                    self.B(pos, local=True)[0],
                ],
                axis=0,
            )

            pos = self.position_along_parent
            parent_basis = np.stack(
                [
                    self.parent_axial_component.T(pos)[0],
                    self.parent_axial_component.N(pos)[0],
                    self.parent_axial_component.B(pos)[0],
                ],
                axis=0,
            )
            self.R_align_with_parent = np.linalg.inv(child_basis) @ parent_basis

    def calc_R_euler_angles(self):
        s = np.sin
        c = np.cos

        a1, a2, a3 = self.euler_angles
        R_x = np.array(
            [
                [1, 0, 0],
                [0, c(a1), -s(a1)],
                [0, s(a1), c(a1)],
            ]
        )

        R_y = np.array(
            [
                [c(a2), 0, s(a2)],
                [0, 1, 0],
                [-s(a2), 0, c(a2)],
            ]
        )

        R_z = np.array(
            [
                [c(a3), -s(a3), 0],
                [s(a3), c(a3), 0],
                [0, 0, 1],
            ]
        )

        self.R_euler_angles = R_z @ (R_y @ R_x)

    # def calc_R_elevation_azimuth_rotation(self):
    #     def find_rot_mat_about_vector(theta, u):
    #         """
    #         Calculate the rotation matrix needed to rotate a point about a unit vector (u) by amount theta. Importantly, this only works if the unit vector passes through the origin, which is a safe assumption in this project.

    #         Args:
    #             theta (float): Angle (radians) for the rotation.
    #             u (array): Vector (shape 3x1) which is the axis about which the point should be rotated.

    #         Returns:
    #             array: 3x3 rotation matrix
    #         """
    #         assert np.isclose(np.linalg.norm(u), 1), "Must be a unit vector"
    #         return [
    #             [
    #                 np.cos(theta) + u[0] ** 2 * (1 - np.cos(theta)),
    #                 u[0] * u[1] * (1 - np.cos(theta)) - u[2] * np.sin(theta),
    #                 u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(theta),
    #             ],
    #             [
    #                 u[0] * u[1] * (1 - np.cos(theta)) + u[2] * np.sin(theta),
    #                 np.cos(theta) + u[1] ** 2 * (1 - np.cos(theta)),
    #                 u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(theta),
    #             ],
    #             [
    #                 u[0] * u[2] * (1 - np.cos(theta)) - u[1] * np.sin(theta),
    #                 u[1] * u[2] * (1 - np.cos(theta)) + u[0] * np.sin(theta),
    #                 np.cos(theta) + u[2] ** 2 * (1 - np.cos(theta)),
    #             ],
    #         ]

    #     s = np.sin
    #     c = np.cos

    #     # Rotation matrix for elevation
    #     e = self.elevation
    #     # self.R_elevation = np.array(
    #     #     [
    #     #         [s(e), c(e), 0],
    #     #         [-c(e), s(e), 0],
    #     #         [0, 0, 1],
    #     #     ]
    #     # )
    #     # self.R_elevation = np.array(
    #     #     [
    #     #         [1, 0, 0],
    #     #         [0, s(e), c(e)],
    #     #         [0, -c(e), s(e)],
    #     #     ]
    #     # )
    #     self.R_elevation = np.array(
    #         [
    #             [s(e), 0, c(e)],
    #             [0, 1, 0],
    #             [-c(e), 0, s(e)],
    #         ]
    #     )
    #     # Rotation matrix for azimuth
    #     a = self.azimuth
    #     # self.R_azimuth = np.array(
    #     #     [
    #     #         [c(a), 0, s(a)],
    #     #         [0, 1, 0],
    #     #         [-s(a), 0, c(a)],
    #     #     ]
    #     # )
    #     self.R_azimuth = np.array(
    #         [
    #             [1, 0, 0],
    #             [0, c(a), s(a)],
    #             [0, -s(a), c(a)],
    #         ]
    #     )

    #     # Rotation matrix for rotation (about axis determined by elevation and azimuth)
    #     # Calculate tangent vector after rotating from elevation and azimuth
    #     T = self.T(self.position_along_self, local=True)
    #     T = T @ self.R_elevation
    #     T = T @ self.R_azimuth

    #     # Get rotation matrix that rotates about this vector by amount theta
    #     r = self.rotation
    #     self.R_rotation = find_rot_mat_about_vector(r, T[0])
    #     self.R_elevation_azimuth_rotation = (
    #         self.R_elevation @ self.R_azimuth @ self.R_rotation
    #     )

    def r(self, t, local=False):

        # Coordinate system explanation. Imagine you are the monkey, and the robot has brought the shape to your hand. The robot gripper is on the right and parallel to the ground, and the shape is pointing to the left.
        # Origin is at position 0.0 of the parent axial component.
        # +x is the direction along which the axial component points (if the axial component has curvature==0, it goes solely along the +x axis)
        # +y is the direction along which the object curves upward
        # +z is the direction pointing roughly in the same direction as the monkey's view.

        # Convert to numpy array
        if type(t) is not np.ndarray:
            t = np.array([t])

        # Assert 0<=t<=1 since this is the ratio along the full length
        if np.any(t < 0) or np.any(t > 1):
            raise Exception("Input t must be within closed interval [0,1].")

        num_vals = len(t)

        # For straight lines, have the full extent be in the x-axis
        if self.curvature == 0:
            x = self.length * t
            y = np.zeros(num_vals)
            z = np.zeros(num_vals)
        # For arcs, use this equation of a parametric curve
        else:
            theta = self.length * t / self.radius  # Convert arc length to radians
            x = self.radius * np.sin(theta)
            y = self.radius * -np.cos(theta) + self.radius
            z = np.zeros(num_vals)

        r = np.stack((x, y, z), axis=-1)

        if local is True:
            return r

        else:
            # Transform to align with parent
            r = r - self.child_join_point  # Move child_join_point to origin
            r = r @ self.R_align_with_parent  # Rotate to align TNB
            r = r @ self.R_euler_angles  # Rotate for elevation, azimuth, rotation
            r = r + self.parent_join_point  # Move child_join_point to parent_join_point

            return r

    def T(self, t, local=False):

        # Convert to numpy array
        if type(t) is not np.ndarray:
            t = np.array([t])

        # Assert 0<=t<=1 since this is the ratio along the full length
        if np.any(t < 0) or np.any(t > 1):
            raise Exception("Input t must be within closed interval [0,1].")

        num_vals = len(t)

        # For straight lines, have the tangent vector be along the x-axis ()
        if self.curvature == 0:
            x = np.ones(num_vals)
            y = np.zeros(num_vals)
            z = np.zeros(num_vals)
        # For arcs, calculate the tangent vector using first derivative of r_t equation
        else:
            theta = self.length * t / self.radius  # Convert arc length to radians
            x = np.cos(theta)
            y = np.sin(theta)
            z = np.zeros(num_vals)

        T = np.stack((x, y, z), axis=-1)

        if local is True:
            return T
        else:
            # Transform to align with parent
            T = T @ self.R_align_with_parent
            T = T @ self.R_euler_angles
            return T

    def N(self, t, local=False):

        # Convert to numpy array
        if type(t) is not np.ndarray:
            t = np.array([t])

        # Assert 0<=t<=1 since this is the ratio along the full length
        if np.any(t < 0) or np.any(t > 1):
            raise Exception("Input t must be within closed interval [0,1].")

        num_vals = len(t)

        # For straight lines, have the normal vector be along the Y-axis ()
        if self.curvature == 0:
            x = np.zeros(num_vals)
            y = np.ones(num_vals)
            z = np.zeros(num_vals)
        # For arcs, calculate the tangent vector using first derivative of r_t equation
        else:
            theta = self.length * t / self.radius  # Convert arc length to radians
            x = -np.sin(theta)
            y = np.cos(theta)
            z = np.zeros(num_vals)

        N = np.stack((x, y, z), axis=-1)

        if local is True:
            return N
        else:
            # Transform to align with parent
            N = N @ self.R_align_with_parent
            N = N @ self.R_euler_angles
            return N

    def B(self, t, local=False):

        # Convert to numpy array
        if type(t) is not np.ndarray:
            t = np.array([t])

        # Assert 0<=t<=1 since this is the ratio along the full length
        if np.any(t < 0) or np.any(t > 1):
            raise Exception("Input t must be within closed interval [0,1].")

        if local is True:
            # Take cross product of T and N
            T = self.T(t, local=True)
            N = self.N(t, local=True)
        else:
            # Take cross product of T and N
            T = self.T(t)
            N = self.N(t)
        B = np.cross(T, N)
        return B

    def get_controlpoints(self):

        # Determine size of controlpoint array
        num_cross_sections = len(self.cross_sections)
        num_rows = NUM_ENDPOINTS + NUM_ENDPOINTS_SLOPE + NUM_CROSS_SECTION_SLOPE + num_cross_sections
        self.num_rows = num_rows
        num_cp_per_cross_section = self.cross_sections[0].controlpoints.shape[0]

        # Controlpoint array structure
        # (0, :, :) - controlpoints at endpoint 0.0
        # (1, :, :) - controlpoints controlling slope at endpoint 0.0
        # (2, :, :) - controlpoints controlling slope at bottom-most cross section
        # (3, :, :) - controlpoints of cross section 0
        # (4, :, :) - controlpoints of cross section 1
        # ...
        # (-4, :, :) - controlpoints of cross section -1
        # (-3, :, :) - controlpoints controlling slope at top-most cross section
        # (-2, :, :) - controlpoints controlling slope at endpoint 1.0
        # (-1, :, :) - controlpoints at endpoint 1.0
        # Assign controlpoints - endpoints
        controlpoints = np.zeros([num_rows, num_cp_per_cross_section, 3])
        controlpoints[0, :, :] = np.repeat(self.r(0.0), num_cp_per_cross_section, axis=0)  # 0.0 endpoint
        controlpoints[-1, :, :] = np.repeat(self.r(1.0), num_cp_per_cross_section, axis=0)  # 1.0 endpoint

        # Assign controlpoints - cross sections
        idx = 3
        for cs in self.cross_sections:

            cp = cs.controlpoints
            pos = cs.position

            # Rotate and translate cross section
            cp = self.align_cross_section_to_position(cp, pos)

            # Assign to controlpoint array
            controlpoints[idx, :, :] = cp
            idx += 1

        # Assign controlpoints - extra to determine slope at endpoint
        for idx in [1, -2]:

            # Get the cross section we will use.
            if idx == 1:
                cs = self.cross_sections[0]
                pos = 0.0
            if idx == -2:
                cs = self.cross_sections[-1]
                pos = 1.0

            # Grab the cp we are going to shrink
            cp = cs.controlpoints

            # Shrink the cross section
            cp = cp * SHRINK_FACTOR

            # Rotate and translate cross section
            cp = self.align_cross_section_to_position(cp, pos)

            # Assign to controlpoint array
            controlpoints[idx, :, :] = cp

        # Assign controlpoints - extra to determine slope at edge cross sections
        for idx in [2, -3]:

            # Get the cross section we will use.
            if idx == 2:
                cs = self.cross_sections[0]
                pos = cs.position
            if idx == -3:
                cs = self.cross_sections[-1]
                pos = cs.position

            # Grab the cp we are going to slide
            cp = cs.controlpoints

            # Rotate and translate cross section
            cp = self.align_cross_section_to_position(cp, pos)

            # Get the tangent vector at this position
            if idx == 2:
                vec = -self.T(pos)  # Negate to go in opposite direction (toward 0.0)
            if idx == -3:
                vec = self.T(pos)

            # Scale vec
            vec = vec * SLIDE_FACTOR

            # Slide cp
            cp = cp + vec

            # Assign to controlpoint array
            controlpoints[idx, :, :] = cp

        self.controlpoints = controlpoints

    def align_cross_section_to_position(self, cp, pos):

        # Rotate so that cross tangent, normal, and binormal vectors are aligned
        current = np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        target = np.stack([self.T(pos)[0], self.N(pos)[0], self.B(pos)[0]], axis=0)
        rot = np.linalg.inv(current) @ target
        cp = cp @ rot

        # Translate to point on backbone
        # point = self.r(pos)
        cp = cp + self.r(pos)

        return cp

    def make_surface(self):

        # Inputs
        degree = ORDER - 1

        # Basis 1 - cross section
        num_cp_per_cross_section = self.cross_sections[0].controlpoints.shape[0]
        num_knots = num_cp_per_cross_section + ORDER + degree
        knot = np.linspace(0, 1, num_knots)
        basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)

        # Basis 2 - along the major axis of the axial component
        num_rows = self.num_rows
        knot = open_uniform_knot_vector(num_rows, ORDER)
        basis2 = BSplineBasis(order=ORDER, knots=knot, periodic=-1)

        # Controlpoints
        cp = self.controlpoints
        cp = cp.reshape(num_rows * num_cp_per_cross_section, cp.shape[2])

        # Surface
        surface = Surface(basis1, basis2, cp, rational=False)
        self.surface = surface

    def make_mesh(self):

        uu = SAMPLING_DENSITY_U
        vv = SAMPLING_DENSITY_V

        ####################
        # Vertices
        (us, vs) = self.surface.start()
        (ue, ve) = self.surface.end()
        u = np.linspace(us, ue, uu, endpoint=False)
        v = np.linspace(vs, ve, vv)
        verts_array = self.surface(u, v)
        verts = np.zeros(((uu) * (vv - 2) + NUM_ENDPOINTS, 3))
        verts[:-2, :] = verts_array[:, 1:-1, :].reshape(-1, 3, order="F")  # Skip endpoints
        verts[-2, :] = self.surface(0, 0)  # Add 0.0 endpoint
        verts[-1, :] = self.surface(1, 1)  # Add 1.0 endpoint
        self.verts = verts

        ####################
        # Faces - CCW Winding (for consistent normals)
        faces = np.zeros((uu * (vv - 2) * 2, 3), dtype="int")
        # faces_array = np.zeros((SD * 2, SD - 2, 3), dtype="int")
        base_column = np.zeros((uu * 2, 3), dtype="int")
        base_column[::2, 0] = np.arange(0, uu)
        base_column[1::2, 0] = np.arange(0, uu)
        base_column[::2, 1] = np.arange(uu, uu * 2)
        base_column[1::2, 1] = np.arange(uu + 1, uu * 2 + 1)
        base_column[::2, 2] = np.arange(uu + 1, uu * 2 + 1)
        base_column[1:-1:2, 2] = np.arange(1, uu)
        base_column[-2, 2] = uu  # Fix wrapping
        base_column[-1, 1] = uu  # Fix wrapping
        base_column[:, 1:] = base_column[:, :-3:-1]  # Reverse for CCW winding

        # Grid faces
        for i in range(vv - 3):
            add_to_column = i * uu
            column = base_column + add_to_column
            start = uu * i * 2
            stop = uu * (i + 1) * 2
            faces[start:stop, :] = column
            # faces_array[:, i, :] = column

        # Endpoint faces
        num_verts = verts.shape[0]
        endpoint_idx = num_verts - 2  # 0.0 endpoint
        column = np.zeros((uu, 3), dtype="int")
        column[:, 0] = np.arange(0, uu)
        column[:-1, 1] = np.arange(1, uu)
        column[:, 2] = endpoint_idx
        column[:, 1:] = column[:, :-3:-1]  # Reverse for CCW winding
        faces[-uu * 2 : -uu] = column

        # Endpoint faces
        endpoint_idx = num_verts - 1  # 1.0 endpoint
        column = np.zeros((uu, 3), dtype="int")
        column[:, 0] = np.arange(
            uu * (vv - 3),
            uu * (vv - 2),
        )
        column[:, 1] = endpoint_idx
        column[:-1, 2] = np.arange(
            uu * (vv - 3) + 1,
            uu * (vv - 2),
        )
        column[-1, 2] = uu * (vv - 3)
        column[:, 1:] = column[:, :-3:-1]  # Reverse for CCW winding
        faces[-uu:] = column
        assert ~np.any(faces > num_verts), "Faces include vertices that don't exist."
        self.faces = faces

        ####################
        # Face normals
        self.face_norms = calc_face_normals(verts, faces)

        ####################
        # Vertex normals
        self.vert_norms = trimesh.geometry.mean_vertex_normals(num_verts, self.faces, self.face_norms)

        # vert_norms = np.zeros(verts.shape)

        # # Vertices with 6 neighbors
        # base_column = np.zeros((uu, 6), dtype="int")
        # base_column[0, :] = np.array(
        #     [uu * 2 - 2, uu * 2 - 1, 0, uu * 2, uu * 2 + 1, uu * 3 - 1]
        # )
        # base_column[-1, :] = np.array(
        #     [[uu * 2 - 4, uu * 2 - 3, uu * 2 - 2, uu * 4 - 3, uu * 4 - 2, uu * 4 - 1]]
        # )
        # base_column[1:-1, 0] = np.arange(0, 0 + 2 * (uu - 2), 2)
        # base_column[1:-1, 1] = np.arange(1, 1 + 2 * (uu - 2), 2)
        # base_column[1:-1, 2] = np.arange(2, 2 + 2 * (uu - 2), 2)
        # base_column[1:-1, 3] = np.arange(2 * uu + 1, 2 * uu + 1 + 2 * (uu - 2), 2)
        # base_column[1:-1, 4] = np.arange(2 * uu + 2, 2 * uu + 2 + 2 * (uu - 2), 2)
        # base_column[1:-1, 5] = np.arange(2 * uu + 3, 2 * uu + 3 + 2 * (uu - 2), 2)
        # array_neighbor_6 = np.zeros((uu, vv - 4, 6), dtype="int")
        # for i in range(0, vv - 4):
        #     column = base_column + 2 * uu * i
        #     array_neighbor_6[:, i, :] = column
        # normals = face_norms[array_neighbor_6]
        # normals = np.mean(normals, axis=2)
        # normals /= np.linalg.norm(normals, axis=2, keepdims=True)
        # vert_norms[1 * uu : -1 * uu - 2] = normals.reshape(-1, 3)

        # # Vertices with 5 neighbors (those bordering endpoint 0.0)
        # column = np.zeros((uu, 5), dtype="int")
        # column[:, 0] = np.arange(-1, 2 * uu - 1, 2)
        # column[:, 1] = np.arange(0, 2 * uu, 2)
        # column[:, 2] = np.arange(1, 2 * uu + 1, 2)
        # column[:, 3] = np.arange(2 * uu * (vv - 3), 2 * uu * (vv - 3) + 2 * uu, 2)
        # column[:, 4] = np.arange(
        #     2 * uu * (vv - 3) + 1, 2 * uu * (vv - 3) + 2 * uu + 1, 2
        # )
        # column[0, :] = np.array(
        #     [2 * uu - 1, 0, 1, 2 * uu * (vv - 3), 2 * uu * (vv - 3) + 1]
        # )
        # normals = face_norms[column]
        # normals = np.mean(normals, axis=1)
        # normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        # vert_norms[:uu] = normals.reshape(-1, 3)

        # # Vertices with 5 neighbors (those bordering endpoint 1.0) #TODO: fix this
        # column = np.zeros((uu, 5), dtype="int")
        # s = 2 * uu * (vv - 4)  # shift amount - for brevity
        # column[:, 0] = np.arange(s - 2, s + 2 * uu - 2, 2)
        # column[:, 1] = np.arange(s - 1, s + 2 * uu - 1, 2)
        # column[:, 2] = np.arange(s, s + 2 * uu, 2)
        # column[:, 3] = np.arange(s + 3 * uu, s + 4 * uu, 1)
        # column[:, 4] = np.arange(s + 3 * uu + 1, s + 4 * uu + 1, 1)
        # column[-1, 4] = s + 3 * uu
        # normals = face_norms[column]
        # normals = np.mean(normals, axis=1)
        # normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        # vert_norms[-uu - 2 : -2] = normals.reshape(-1, 3)

        # # Vertices with uu neighbors (endpoints)
        # endpoint_idx = num_verts - 2  # 0.0 endpoint
        # start = 2 * uu * (vv - 3) + 0 * uu
        # stop = start + uu
        # neighbors = np.arange(start, stop)
        # normal = face_norms[neighbors]
        # normal = np.mean(normal, axis=0)
        # normal /= np.linalg.norm(normal)
        # vert_norms[endpoint_idx] = normal

        # # Vertices with SD neighbors (endpoints)
        # endpoint_idx = num_verts - 1  # 0.0 endpoint
        # start = 2 * uu * (vv - 3) + 1 * uu
        # stop = start + uu
        # neighbors = np.arange(start, stop)
        # normal = face_norms[neighbors]
        # normal = np.mean(normal, axis=0)
        # normal /= np.linalg.norm(normal)
        # vert_norms[endpoint_idx] = normal

        ####################
        # Construct trimesh
        self.mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            face_normals=self.face_norms,
            vertex_normals=self.vert_norms,
            process=False,
        )

    def plot_o3d_mesh(self):
        """
        Plots the mesh using the custom triangles, vertices, and normals.
        """
        self.o3d_mesh = self.mesh.as_open3d

        # Unsure why, but need to copy for the vector3dvector function to work
        vert_norms = copy.copy(self.vert_norms)
        face_norms = copy.copy(self.face_norms)
        self.o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vert_norms)
        self.o3d_mesh.triangle_normals = o3d.utility.Vector3dVector(face_norms)
        o3d.visualization.draw_geometries([self.o3d_mesh])
