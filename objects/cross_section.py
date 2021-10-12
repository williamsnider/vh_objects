import numpy as np
from objects.utilities import angle_between


class CrossSection:
    def __init__(self, controlpoints, position, rotation=0, tilt=0):

        # Assign object attributes
        self.controlpoints = controlpoints
        self.position = position
        self.rotation = rotation
        self.tilt = tilt

        # Check that inputs are valid
        self.check_inputs()

        # Carry out rotation, roll, and tilt
        self.calc_points()

    def check_inputs(self):

        assert type(self.controlpoints) is np.ndarray, "controlpoints must be input as a numpy array."

        assert self.controlpoints.shape[1] == 2, "controlpoints must have 2 dimensions."

        assert np.all(np.sum(self.controlpoints ** 2, axis=1) > 0), "Controlpoints cannot be at origin (0, 0)."

        assert self.position >= 0 and self.position <= 1, "position must be within closed interval [0,1]."

        assert self.rotation >= 0 and self.rotation <= 2 * np.pi, "rotation must be within closed interval [0,1]."

        assert self.tilt <= np.pi / 2 and self.tilt >= -np.pi / 2, "tilt must be within closed interval [-pi/2, pi/2]."

    def calc_points(self):
        self.calc_rotation()
        self.align_controlpoints()
        self.calc_tilt()

    def calc_rotation(self):

        # Rotate controlpoints
        cp = self.controlpoints
        t = self.rotation
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        self.controlpoints = cp @ R

    def align_controlpoints(self):

        # Find angles between vectors to controlpoints and <1,0> (vector to theta=0)
        vec = self.controlpoints / np.linalg.norm(self.controlpoints, axis=1, keepdims=True)
        # angles = np.arccos(vec[:, 0])
        angles = angle_between(vec, np.array([1, 0]))

        # Roll so that the vector between the origin and the 0th controlpoint is closest to having a minimum able between it and <1,0>
        shift = np.where(angles == angles.min())[0]
        cp = np.roll(self.controlpoints, -shift, axis=0)

        # Check the roll worked
        angles = angle_between(cp, np.array([1, 0]))
        assert np.where(angles == angles.min())[0] == 0

        self.controlpoints = cp

    def calc_tilt(self):

        # Add z-axis
        num_cp = self.controlpoints.shape[0]
        cp = np.zeros((num_cp, 3))
        cp[:, :2] = self.controlpoints

        # Rotate about x-axis
        t = self.tilt
        c = np.cos
        s = np.sin
        R = np.array(
            [
                [1, 0, 0],
                [0, c(t), -s(t)],
                [0, s(t), c(t)],
            ]
        )
        cp = cp @ R

        self.controlpoints = cp
