from splipy import Curve, BSplineBasis
from objects.parameters import NUM_SAMPLES_FOR_REPARAMETERIZATION, ORDER, NUM_INTERPOLATION_POINTS, EPSILON
from objects.utilities import open_uniform_knot_vector, calc_R_euler_angles
import numpy as np
from objects.backbone import Backbone


class BackboneFromDigits:
    """Takes a list of digit segments and angles and returns the controlpoint array corresponding to the backbone.

    This is basically just a helper class since for the current experiment we are building the backbone thinking about digits."""

    def __init__(self, digit_segments, angles_between_segments):

        self.digit_segments = digit_segments.copy()  # Solve bug
        self.angles_between_segments = angles_between_segments.copy()

        self.num_digit_segments = len(self.digit_segments)
        self.num_angles_between_segments = len(angles_between_segments)

        self.check_inputs()
        self.align_segments()
        self.link_segments()

    def check_inputs(self):
        assert (
            self.num_digit_segments - 1 == self.num_angles_between_segments
        ), "num_digit_segments is not 1 larger than num_angles_between_segments"

        assert type(
            self.digit_segments[0] == type(Backbone)
        ), "digit_segment is not an instance of the Backbone class"  # Use Backbone class because T,N,B are calculated more predictably than splipy.curve

        for segment in self.digit_segments:

            assert np.all(
                np.isclose(segment.controlpoints[0], 0)
            ), "First controlpoint for each digit segment must be at origin (0,0,0)."

    def align_segments(self):
        """Connect the digits end-to-end, also preserving their angles."""

        # # Create backbone_cp_array
        # num_cp_per_segment = (
        #     self.digit_segments[0].controlpoints.shape[0] - 1
        # )  # Segment1's last cp overlaps with Segments2's first cp
        # num_cp_per_backbone = self.num_segments * num_cp_per_segment + 1  # Add in cp for Segment[-1]'s last cp
        # backbone_cp = np.zeros((num_cp_per_backbone, 3))

        for i in range(self.num_digit_segments):

            if i == 0:
                continue

            prev_segment = self.digit_segments[i - 1]
            curr_segment = self.digit_segments[i]

            # Translation matrix to align end of previous and beginning of current
            T = prev_segment.controlpoints[-1] - curr_segment.controlpoints[0]
            print(T)
            # Rotation matrix to align end of previous and beginning of current
            prev_TNB = np.stack(
                [
                    prev_segment.T(1)[0],
                    prev_segment.N(1)[0],
                    prev_segment.B(1)[0],
                ]
            )
            curr_TNB = np.stack(
                [
                    curr_segment.T(0)[0],
                    curr_segment.N(0)[0],
                    curr_segment.B(0)[0],
                ]
            )
            R_align = np.linalg.inv(curr_TNB) @ prev_TNB

            # Rotation matrix to adjust angle of current
            R_euler = calc_R_euler_angles(self.angles_between_segments[i - 1])

            # Carry out transformations
            cp = curr_segment.controlpoints.copy()
            cp = cp @ R_align
            cp = cp @ R_euler
            cp = cp + T

            # Update list
            self.digit_segments[i] = Backbone(
                cp, reparameterize=False
            )  # Create new segment with the aligned controlpoints

            # Begining and end of cps must be the same
            # TODO: FIX THIS
            assert np.all(
                np.isclose(self.digit_segments[i].controlpoints[0], self.digit_segments[i - 1].controlpoints[-1])
            )
            print(cp)

    def link_segments(self):

        # Create backbone_cp_array
        num_cp_per_segment = (
            self.digit_segments[0].controlpoints.shape[0] - 1
        )  # Segment1's last cp overlaps with Segments2's first cp
        num_cp_per_backbone = (
            self.num_digit_segments * num_cp_per_segment + 1
        )  # Add in 1 extra cp for Segment[-1]'s last cp
        backbone_cp = np.zeros((num_cp_per_backbone, 3))

        for i, segment in enumerate(self.digit_segments):

            # Include all cps for the last segment\
            if i == self.num_digit_segments - 1:
                start = i * num_cp_per_segment
                backbone_cp[start:, :] = segment.controlpoints  # Include last cp
            else:
                start = i * num_cp_per_segment
                stop = (i + 1) * num_cp_per_segment
                backbone_cp[start:stop, :] = segment.controlpoints[:num_cp_per_segment]  # Skip last cp

        self.controlpoints = backbone_cp
