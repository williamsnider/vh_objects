import numpy as np
from splipy import BSplineBasis, Surface
from objects.parameters import SAMPLING_DENSITY_U, SAMPLING_DENSITY_V
from objects.utilities import calc_hemisphere_controlpoints, make_surface, make_mesh


class AxialComponent:
    """AxialComponents are BSpline surfaces constructed from a backbone and a list of cross sections. The surface is sampled to form a watertight, triangular mesh."""

    def __init__(
        self,
        backbone,
        cross_sections,
        euler_angles=np.array([0, 0, 0]),
        parent_axial_component=None,
        position_along_parent=1.0,
        position_along_self=0.0,
        smooth_with_post=False,
        hemispherical_ends=False,
        hemispherical_polynomial=None,
        hemisphere_x=[None, None],
    ):
        self.backbone = backbone
        self.cross_sections = cross_sections
        self.euler_angles = euler_angles
        self.parent_axial_component = parent_axial_component
        self.position_along_parent = position_along_parent
        self.position_along_self = position_along_self
        self.smooth_with_post = smooth_with_post
        self.length = self.backbone.length()
        self.hemispherical_ends = hemispherical_ends
        self.hemispherical_polynomial = hemispherical_polynomial
        self.hemisphere_x = hemisphere_x

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
        """Executes the steps of axial component construction."""

        self.calc_transformation_matrices()
        self.calc_R_euler_angles()
        self.get_controlpoints()
        if self.hemispherical_ends == True:
            self.make_hemispherical_ends()
        self.surface = make_surface(self.controlpoints)
        self.mesh = make_mesh(self.surface, SAMPLING_DENSITY_U, SAMPLING_DENSITY_V)

    def calc_transformation_matrices(self):
        """Calculates the transformation matrices for axial components that are not the parent axial component."""
        if self.parent_axial_component is None:
            # Initialize translation and rotation matrices, which will be updated later
            self.translation = np.array([0, 0, 0])
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
        # Coordinate system explanation
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
        """Calculate tangent vector at point along backbone of axial component."""
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
        """Calculate normal vector at point along backbone of axial component."""
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
        """Calculate binormal vector at point along backbone of axial component."""

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
        Gets control points from cross sections, which are used to make the axial component surface.

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
        NUM_ENDPOINTS = 2
        NUM_ENDPOINTS_SLOPE = 2

        # Construct empty controlpoint array
        num_cross_sections = len(self.cross_sections)
        num_rows = NUM_ENDPOINTS + NUM_ENDPOINTS_SLOPE + num_cross_sections
        self.num_rows = num_rows
        num_cp_per_cross_section = self.cross_sections[0].controlpoints.shape[0]
        controlpoints = np.zeros([num_rows, num_cp_per_cross_section, 3])

        # Assign left side controlpoints
        SHRINK_FACTOR = 0.5
        pos = 0.0
        cs_idx = 0
        R = self.calc_R_cs_to_pos(pos)
        cs_rot = self.cross_sections[cs_idx].controlpoints @ R
        controlpoints[0] = cs_rot * 0 + self.r(pos)  # Endpoint
        controlpoints[1] = cs_rot * SHRINK_FACTOR + self.r(pos)  # Determines slope at endpoint

        # Assign right side controlpoints
        pos = 1.0
        cs_idx = -1
        R = self.calc_R_cs_to_pos(pos)
        cs_rot = self.cross_sections[cs_idx].controlpoints @ R
        controlpoints[-1] = cs_rot * 0 + self.r(pos)  # Endpoint
        controlpoints[-2] = cs_rot * SHRINK_FACTOR + self.r(pos)  # Determines slope at endpoint

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

    def make_hemispherical_ends(self):
        """Adjusts the controlpoints on the ends so that the axial component has hemispherical ends that smoothly fit with a quadratic scaleprofile."""

        # Left side
        t = 0.0
        x = self.hemisphere_x[0]
        base_cp = self.controlpoints[2]
        tan_vec = self.T(t)
        endpoint = self.r(t)
        result_left, a = calc_hemisphere_controlpoints(
            base_cp,
            tan_vec,
            endpoint,
            self.hemispherical_polynomial,
            x,
            morph_to_ellipse=False,
        )

        # Right side
        t = 1.0
        x = self.hemisphere_x[1]
        base_cp = self.controlpoints[-3]
        tan_vec = self.T(t)
        endpoint = self.r(t)
        result_right, a = calc_hemisphere_controlpoints(
            base_cp,
            tan_vec,
            endpoint,
            self.hemispherical_polynomial,
            x,
            morph_to_ellipse=False,
        )

        mid_cp = self.controlpoints[2:-2]
        new_cp = np.vstack([result_left[:-1], mid_cp, result_right[-2::-1]])

        self.controlpoints = new_cp
        self.num_rows = self.controlpoints.shape[0]
        self.a = a

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
