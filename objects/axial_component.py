import numpy as np
from splipy import BSplineBasis, Surface
import trimesh
from objects.parameters import ORDER, SHRINK_FACTOR, SAMPLING_DENSITY_U, SAMPLING_DENSITY_V
from objects.utilities import open_uniform_knot_vector, calc_cp_hemisphere, fair_mesh
from objects.backbone import Backbone
import scipy.interpolate
import scipy.spatial

# Fixed variables - used to built controlpoints array with the correct size.
NUM_ENDPOINTS = 2  # Position 0.0 and 1.0 on the axial component's backbone
NUM_ENDPOINTS_SLOPE = 2  # Controlpoints used to control surface slope near endpoints
# NUM_CROSS_SECTION_SLOPE = 2  # Controlpoints used to control surface slope near cross sections adjacent to endpoints


class AxialComponent:
    def __init__(
        self,
        backbone,
        cross_sections,
        euler_angles=np.array([0, 0, 0]),
        parent_axial_component=None,
        position_along_parent=1.0,
        position_along_self=0.0,
        smooth_with_post=False,
        hemisphere_ends=False,
        endpoint_offsets=np.array([0, 0]),
    ):
        self.backbone = backbone
        self.cross_sections = cross_sections
        self.euler_angles = euler_angles
        self.parent_axial_component = parent_axial_component
        self.position_along_parent = position_along_parent
        self.position_along_self = position_along_self
        self.smooth_with_post = smooth_with_post
        self.length = self.backbone.length()
        self.hemisphere_ends = hemisphere_ends
        self.endpoint_offsets = endpoint_offsets

        # Do calculations
        self.calc_points()

    def check_inputs(self):

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
        self.get_controlpoints()
        # self.adjust_controlpoints()
        self.make_surface()
        self.make_mesh()

    def calc_transformation_matrices(self):

        if self.parent_axial_component is None:

            # Initialize translation and rotation matrices, which will be updated later
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

    def r(self, t, local=False):

        # TODO: This is out of date
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

        r = self.backbone.r(t)

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

        T = self.backbone.T(t)

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

        N = self.backbone.N(t)

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

        B = self.backbone.B(t)

        if local is True:
            return B
        else:
            # Transform to align with parent
            B = B @ self.R_align_with_parent
            B = B @ self.R_euler_angles
            return B
        return B

    def get_controlpoints(self):
        """
        # controlpoints array structure
        # (0, :, :) - controlpoints at endpoint 0.0
        # (1, :, :) - controlpoints controlling slope at endpoint 0.0
        # (2, :, :) - controlpoints of cross section 0
        # (3, :, :) - controlpoints of cross section 1
        # ...
        # (-4, :, :) - controlpoints of cross section -2
        # (-3, :, :) - controlpoints of cross section -1
        # (-2, :, :) - controlpoints controlling slope at endpoint 1.0
        # (-1, :, :) - controlpoints at endpoint 1.0
        """

        # Construct empty controlpoint array
        num_cross_sections = len(self.cross_sections)
        num_rows = NUM_ENDPOINTS + NUM_ENDPOINTS_SLOPE + num_cross_sections
        self.num_rows = num_rows
        num_cp_per_cross_section = self.cross_sections[0].controlpoints.shape[0]
        controlpoints = np.zeros([num_rows, num_cp_per_cross_section, 3])

        # Assign left side controlpoints
        SHRINK_FACTOR = 0.75
        pos = 0.0
        cs_idx = 0
        R = self.calc_R_cs_to_pos(pos)
        cs_rot = self.cross_sections[cs_idx].controlpoints @ R
        controlpoints[0] = cs_rot * 0 + self.r(pos) - self.T(pos) * self.endpoint_offsets[0]  # Endpoint
        controlpoints[1] = (
            cs_rot * SHRINK_FACTOR + self.r(pos) - self.T(pos) * self.endpoint_offsets[0]
        )  # Determines slope at endpoint

        # Assign right side controlpoints
        pos = 1.0
        cs_idx = -1
        R = self.calc_R_cs_to_pos(pos)
        cs_rot = self.cross_sections[cs_idx].controlpoints @ R
        controlpoints[-1] = cs_rot * 0 + self.r(pos) + self.T(pos) * self.endpoint_offsets[1]  # Endpoint
        controlpoints[-2] = (
            cs_rot * SHRINK_FACTOR + self.r(pos) + self.T(pos) * self.endpoint_offsets[1]
        )  # Determines slope at endpoint

        # Assign interior controlpoints
        idx = 2
        for cs in self.cross_sections:

            cp = cs.controlpoints
            pos = cs.position

            # Rotate and translate cross section
            R = self.calc_R_cs_to_pos(pos)
            cs_rot = cp @ R
            cs_T = cs_rot + self.r(pos)

            # Assign to controlpoint array
            controlpoints[idx] = cs_T
            idx += 1
        self.controlpoints = controlpoints
        return

    def calc_R_cs_to_pos(self, pos):

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
        return rot
        # cp = cp @ rot

        # # Translate to point on backbone
        # # point = self.r(pos)
        # cp = cp + self.r(pos)

        # return cp

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
        # Skip calculations for face and vertex normals since that should be done after fusing all axial components

        ####################
        # Construct trimesh
        self.mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False,
        )

        # Fair mesh at intersection of hemisphere and axial component
        if self.hemisphere_ends == True:

            mesh = self.mesh.copy()

            # Determine the indices in the vv dimension that correspond to the intersection
            # lower = 2.75
            upper = 5
            # l_start = round((vv - 2) * lower / self.num_rows)
            l_start = 1
            r_end = (vv - 2) - l_start
            l_end = round((vv - 2) * upper / self.num_rows)
            l_indices = np.arange(uu * l_start, uu * l_end)
            r_start = round((vv - 2) * (self.num_rows - upper) / self.num_rows)
            # r_end = round((vv - 2) * (self.num_rows - lower) / self.num_rows)
            r_indices = np.arange(uu * r_start, uu * r_end)
            indices = np.concatenate([l_indices, r_indices])
            mesh.visual.vertex_colors[indices] = [255, 255, 0, 255]
            # mesh.show(smooth=False)
            faired_mesh = fair_mesh(mesh, indices, harmonic_power=2)
            self.mesh = faired_mesh
