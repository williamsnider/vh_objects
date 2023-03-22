import numpy as np
from splipy import BSplineBasis, Surface
import trimesh
from objects.parameters import ORDER, SHRINK_FACTOR, SAMPLING_DENSITY_U, SAMPLING_DENSITY_V
from objects.utilities import open_uniform_knot_vector, calc_cp_hemisphere
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

        if self.smooth_with_post == False:

            # Construct empty controlpoint array
            num_cross_sections = len(self.cross_sections)
            num_rows = NUM_ENDPOINTS + NUM_ENDPOINTS_SLOPE + num_cross_sections
            self.num_rows = num_rows
            num_cp_per_cross_section = self.cross_sections[0].controlpoints.shape[0]
            controlpoints = np.zeros([num_rows, num_cp_per_cross_section, 3])

            # Assign controlpoints - endpoints
            controlpoints[0, :, :] = np.repeat(self.r(0.0), num_cp_per_cross_section, axis=0)  # 0.0 endpoint
            controlpoints[-1, :, :] = np.repeat(self.r(1.0), num_cp_per_cross_section, axis=0)  # 1.0 endpoint

            # Assign controlpoints - cross sections
            idx = 2
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

            self.controlpoints = controlpoints
        else:

            """
            # controlpoints array structure
            # (0, :, :) - controlpoints at center of post/interface intersection
            # (1, :, :) - controlpoints at outer radius of post/interface intersection
            # (2, :, :) - controlpoints at outer radius of post/interface intersection slightly shifted
            # (3, :, :) - controlpoints at outer radius of post/interface intersection at midpoint
            # (4, :, :) - controlpoints at outer radius of post/interface intersection at 0.0 point
            # (5, :, :) - controlpoints of cross section 0
            # (6, :, :) - controlpoints of cross section 1
            # ...
            # (-4, :, :) - controlpoints of cross section -2
            # (-3, :, :) - controlpoints of cross section -1
            # (-2, :, :) - controlpoints controlling slope at endpoint 1.0
            # (-1, :, :) - controlpoints at endpoint 1.0
            """

            # Construct empty controlpoint array
            num_cross_sections = len(self.cross_sections)
            NUM_POST_POINTS = 5
            NUM_SINGLE_ENDCAP_POINT = 2
            num_rows = NUM_POST_POINTS + NUM_SINGLE_ENDCAP_POINT + num_cross_sections
            self.num_rows = num_rows
            num_cp_per_cross_section = self.cross_sections[0].controlpoints.shape[0]
            controlpoints = np.zeros([num_rows, num_cp_per_cross_section, 3])

            # Assign controlpoints - post
            POST_LENGTH = 15
            POST_INTERFACE_POINT = np.array([-POST_LENGTH, 0, 0]) + self.r(0.0)
            controlpoints[0, :, :] = np.repeat(POST_INTERFACE_POINT, num_cp_per_cross_section, axis=0)  # 1.0 endpoint

            # TODO: Make this just linear
            POST_RADIUS = 5.404  # This gives a B-spline radius of 5.00mm with 8 controlpoints.
            c = np.cos
            s = np.sin
            th = np.linspace(0, 2 * np.pi, num_cp_per_cross_section, endpoint=False).reshape(-1, 1)
            cp = np.hstack((np.zeros((num_cp_per_cross_section, 1)), POST_RADIUS * c(th), POST_RADIUS * s(th)))
            controlpoints[1, :, :] = cp + self.r(0.0) + np.array([-POST_LENGTH, 0, 0])
            controlpoints[2, :, :] = cp + self.r(0.0) + np.array([-POST_LENGTH, 0, 0]) * 0.8
            controlpoints[3, :, :] = cp + self.r(0.0) + np.array([-POST_LENGTH, 0, 0]) * 0.5
            controlpoints[4, :, :] = cp + self.r(0.0) + np.array([-POST_LENGTH, 0, 0]) * 0.2

            # Assign controlpoints - endpoints
            controlpoints[-1, :, :] = np.repeat(self.r(1.0), num_cp_per_cross_section, axis=0)  # 1.0 endpoint

            # Assign controlpoints - cross sections
            idx = 5
            for cs in self.cross_sections:

                cp = cs.controlpoints
                pos = cs.position

                # Rotate and translate cross section
                cp = self.align_cross_section_to_position(cp, pos)

                # Assign to controlpoint array
                controlpoints[idx, :, :] = cp
                idx += 1

            # Assign controlpoints - extra to determine slope at endpoint
            for idx in [-2]:

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
            self.controlpoints = controlpoints

        if self.hemisphere_ends == True:
            """Adjust controlpoints so that ends are hemispherical"""
            orig_cp = self.controlpoints
            center_cp = orig_cp[2:-2]

            cp_hemisphere_right = calc_cp_hemisphere(center_cp[-1], self.r(1.0), self.T(1.0))
            cp_hemisphere_left = calc_cp_hemisphere(center_cp[0], self.r(0.0), -self.T(0.0))

            new_cp = np.vstack([cp_hemisphere_left[:0:-1], center_cp, cp_hemisphere_right[1:]])
            self.controlpoints = new_cp
            self.num_rows = new_cp.shape[0]

    def adjust_controlpoints(self):
        """Adjusts controlpoints to prevent self intersections.

        Makes the assumption that the largest extent of shapes are in the xy plane."""
        cp = self.controlpoints
        adjusted_cp = cp.copy()
        for i in range(cp.shape[1]):
            pts = cp[:, i, :]

            # Project into plane connecting endpoints and midpoint of controlpoint vector
            pA = pts[0]
            pB = pts[cp.shape[0] // 2]
            pC = pts[-1]
            vecAB = (pB - pA) / np.linalg.norm(pB - pA)
            vecAC = (pC - pA) / np.linalg.norm(pC - pA)
            N = (np.cross(vecAB, vecAC) / np.linalg.norm(np.cross(vecAB, vecAC))).reshape(-1, 3)
            T = vecAB.reshape(-1, 3)
            B = np.cross(N, vecAB).reshape(-1, 3)
            curr = np.vstack([T, B, N])
            goal = np.eye(3)
            R = goal @ np.linalg.inv(curr.T).T
            pts_flat = pts @ R

            # # Project into xy plane for purposes of finding loop
            # pts_xy = pts.copy()
            # pts_xy[:, 2] = 0

            t = np.arange(pts.shape[0])
            spline = scipy.interpolate.make_interp_spline(t, pts_flat, k=1)

            # Identify regions of overlap
            NUM_SAMPLES = 100
            tt = np.linspace(t[0], t[-1], NUM_SAMPLES)
            spl = spline(tt)
            tree = scipy.spatial.KDTree(spl)
            d, indices = tree.query(spl, 2)

            # Overlaps occur when the index of the 1st (self) and 2nd nearest neighbors are far apart
            THRESHOLD = int(0.1 * NUM_SAMPLES)
            diff = np.abs(indices[:, 0] - indices[:, 1])
            overlap_indices = np.where(diff > THRESHOLD)[0]

            if len(overlap_indices) == 0:
                continue

            # # Test if the angle between line segments is >90
            # segments_as_vectors = np.diff(spl, axis=0)
            # ANGLE_THRESHOLD = np.pi
            # large_angles = (
            #     np.where(angle_between(segments_as_vectors[1:], segments_as_vectors[:-1]) > ANGLE_THRESHOLD)[0] + 1
            # )

            start = np.ceil(overlap_indices.min() / NUM_SAMPLES * t[-1]).astype("int")
            stop = np.floor(overlap_indices.max() / NUM_SAMPLES * t[-1]).astype("int")

            # Cubic hermite spline
            x = np.array([0, 1])
            y = pts[[start - 1, stop + 1]]
            dydx = np.array(
                [
                    pts[start - 1] - pts[start - 2],
                    pts[stop + 2] - pts[stop + 1],
                ]
            )
            chs = scipy.interpolate.CubicHermiteSpline(x, y, dydx)

            # New cp
            inner_t = np.linspace(0, 1, stop - start + 3)[1:-1]
            new_cp = np.vstack(
                [
                    pts[:start],
                    chs(inner_t),
                    pts[stop + 1 :],
                ]
            )
            adjusted_cp[:, i, :] = new_cp

            # import matplotlib.pyplot as plt

            # ax = plt.figure().add_subplot(projection="3d")
            # # ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-r*")
            # ax.plot(pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2], "-b*")
            # # ax.plot(new_cp[:, 0], new_cp[:, 1], new_cp[:, 2], "-k*")

            # ax.plot(spl[overlap_indices, 0], spl[overlap_indices, 1], spl[overlap_indices, 2], "og")
            # # ax.plot(new_spl[:, 0], new_spl[:, 1], new_spl[:, 2], "-b")
            # # xlim = ax.get_xlim3d()
            # # ylim = ax.get_ylim3d()
            # # zlim = ax.get_zlim3d()
            # # ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
            # plt.show()

        self.controlpoints = adjusted_cp

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
        # Skip calculations for face and vertex normals since that should be done after fusing all axial components

        ####################
        # Construct trimesh
        self.mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=False,
        )
