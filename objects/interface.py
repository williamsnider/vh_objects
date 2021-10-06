import trimesh
from objects.parameters import CUBE_SIDE_LENGTH, POST_RADIUS, POST_HEIGHT, POST_SECTIONS, FINGERTIP_SLOT_SIDE_LENGTH
from compas_cgal.booleans import boolean_union, boolean_difference
import numpy as np


class Interface:
    """
    What the robot will grab onto
    """

    def __init__(
        self,
        label="test123",
    ):
        self.label = label

        # Make the interface
        self.make_base_interface()

    def make_base_interface(self):

        # TODO: Rewrite using compas.geometry primitives and see if that allows the interface to be watertight with POST_SECTIONS=8.

        # Make cube
        cube = trimesh.primitives.Box(extents=(CUBE_SIDE_LENGTH, CUBE_SIDE_LENGTH, CUBE_SIDE_LENGTH))
        cube_VF = [cube.vertices, cube.faces]

        # Make post
        T = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, POST_HEIGHT / 2 + CUBE_SIDE_LENGTH / 2],
                [0, 0, 0, 1],
            ]
        )
        post = trimesh.primitives.Cylinder(
            radius=POST_RADIUS,
            height=POST_HEIGHT,
            sections=POST_SECTIONS,
            transform=T,
        )
        post_VF = [post.vertices, post.faces]
        base_interface_VF = boolean_union(cube_VF, post_VF)

        # Make fingertip slot - +Y side
        T = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, CUBE_SIDE_LENGTH / 2 - FINGERTIP_SLOT_SIDE_LENGTH / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        fingertip_slot_A = trimesh.primitives.Box(
            extents=(FINGERTIP_SLOT_SIDE_LENGTH, FINGERTIP_SLOT_SIDE_LENGTH, FINGERTIP_SLOT_SIDE_LENGTH),
            transform=T,
        )
        fingertip_slot_A_VF = [fingertip_slot_A.vertices, fingertip_slot_A.faces]
        base_interface_VF = boolean_difference(base_interface_VF, fingertip_slot_A_VF)

        # Make fingertip slot - -Y side
        fingertip_slot_B = trimesh.primitives.Box(
            extents=(FINGERTIP_SLOT_SIDE_LENGTH, FINGERTIP_SLOT_SIDE_LENGTH, FINGERTIP_SLOT_SIDE_LENGTH),
            transform=-T,
        )
        fingertip_slot_B_VF = [fingertip_slot_B.vertices, fingertip_slot_B.faces]
        base_interface_VF = boolean_difference(base_interface_VF, fingertip_slot_B_VF)

        # Make peg

        mesh = trimesh.Trimesh(
            vertices=base_interface_VF[0],
            faces=base_interface_VF[1],
        )
        mesh.show()
        assert mesh.is_watertight, "Mesh is not watertight. Try adjusting the number of POST_SECTIONS."

        pass

    def add_label(self):
        pass

    def extrude_post_to_shape(self):
        pass


# Outline
# Generate shape
# Determine bounding box of shape (use this to exclude shapes that are too large)
# Orient interface to this bounding box (bottommost portion of shape should be along same plane as bottom of peg)
# Use pre-built interface (containing cube, fingertip, peg, and post) (missing label and extrusion to shape)
# Extrude the end of the post to the base of the shape
