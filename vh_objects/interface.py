# Add label to interface
import trimesh
import numpy as np
from freetype import Face
import trimesh
from shapely import geometry
from vh_objects.utilities import fuse_meshes
from scipy.spatial.transform import Rotation
from vh_objects.parameters import (
    FONT_PATH,
    INTERFACE_WIDTH,
    LABEL_DEPTH,
    INTERFACE_DEPTH_FROM_ORIGIN,
    INTERFACE_PATH,
    INTERFACE_SHIFT,
)

########################
### Helper functions ###
########################


def get_character_outline(char, font_height, shift=(0, 0)):
    """Get the outline of a text character. Returns a list of contours, where each contour is a list of points."""
    char_size_pts = int(font_height * 1000 * 3 // 2)

    assert len(char) == 1
    face = Face(str(FONT_PATH))
    face.set_char_size(char_size_pts)  # If changing, must also change HEIGHT and WIDTH in calc_offsets
    face.load_char(char)
    slot = face.glyph
    outline = slot.outline
    contours = outline.contours

    # Iterate through contours to get groups of points
    start = 0
    contours_list = []
    outline_points = np.array(outline.points)
    outline_points += shift  # Shift x and y
    for c in contours:
        end = c + 1
        contour_points = outline_points[start:end]
        contour_points = np.vstack([contour_points, contour_points[0]])  # Add first element to end to complete loop
        contour_points = contour_points.astype("float")

        start = end

        # Append only if there are at least 4 contour points (including wrapping the first)
        if contour_points.shape[0] >= 4:
            contours_list.append(contour_points)

    # Scale from pts to mm
    scaled_list = []
    for c in contours_list:
        scaled_list.append(c / 1000)

    return scaled_list


def calc_offsets(text, font_height):
    """Calculate offsets for centering lines/characters."""

    lines = text.split("_")
    num_lines = len(lines)

    FONT_OFFSET_HEIGHT = int(font_height * 1000)
    FONT_OFFSET_WIDTH = int(font_height * 1000 * 9 // 10)

    # Calculate height offsets based on number of lines
    height_offsets = np.arange(FONT_OFFSET_HEIGHT * num_lines, 0, -FONT_OFFSET_HEIGHT)
    if num_lines % 2 == 1:
        height_offsets -= height_offsets[(num_lines - 1) // 2] + FONT_OFFSET_HEIGHT // 2
    else:
        height_offsets -= height_offsets[num_lines // 2] + FONT_OFFSET_HEIGHT

    # Calculate width offsets based on number of charactes in each line
    offsets = []
    for line_num, line in enumerate(lines):
        num_chars = len(line)

        width_offsets = np.arange(0, FONT_OFFSET_WIDTH * num_chars, FONT_OFFSET_WIDTH)
        if num_chars % 2 == 1:
            width_offsets -= width_offsets[(num_chars) // 2] + FONT_OFFSET_WIDTH // 2
        else:
            width_offsets -= width_offsets[(num_chars) // 2]

        line_offset = [(w, height_offsets[line_num]) for w in width_offsets]
        offsets.append(line_offset)

    # Traverse characters to output a nested list corresponding to offsets
    chars = []
    for line_num, line in enumerate(lines):
        char = [c for c in line]
        chars.append(char)

    return chars, offsets


def text_to_mesh(text, extrusion_height, font_height):
    """Converts text to a trimesh."""

    # Extract characters and calculate offsets for centering lines/characters
    chars, offsets = calc_offsets(text, font_height)

    # Gather outlines of characters
    text_list = []
    for line_num, line in enumerate(chars):
        assert len(line) <= 4, "No more than 4 characters allowed per line."
        for char_num, char in enumerate(line):
            offset = offsets[line_num][char_num]
            contours_list = get_character_outline(char, font_height, shift=offset)
            exterior = contours_list[0]
            num_contours = len(contours_list)

            if num_contours > 1:
                interior = contours_list[1:]
            else:
                interior = []

            text_list.append([exterior, interior])

    polygons = []
    for exterior, interior in text_list:
        polygons.append(geometry.Polygon(shell=exterior, holes=interior))

    meshes = []
    for poly in polygons:
        meshes.append(
            trimesh.creation.extrude_polygon(poly, extrusion_height * 2)
        )  # Double extrusion since half will not be intersecting the interface (i.e. need to double or else the actual depth of the label will not equal the requested depth).

    return meshes


def load_interface(stl_path, label=None):
    """Loads the interface and adds the label."""
    interface = trimesh.load_mesh(stl_path)
    Z_shift = -17.5  # mm

    if label != None:
        # Generate mesh of label
        label_meshes = text_to_mesh(label, LABEL_DEPTH)

        # # Top label
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("zyx", np.array([-np.pi / 2, 0, 0])).as_matrix()
        T[:3, 3] = np.array(
            [
                0,
                -1 * INTERFACE_DEPTH_FROM_ORIGIN,
                22.5 + 25.4 / 2 - LABEL_DEPTH,
            ]
        )
        top_label = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # Back label
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("zyx", np.array([-np.pi / 2, 0, np.pi / 2])).as_matrix()
        T[:3, 3] = np.array(
            [
                0,
                -2 * INTERFACE_DEPTH_FROM_ORIGIN + LABEL_DEPTH,
                15,
            ]
        )
        back_label = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # Left Label down
        T = np.eye(4)
        R = Rotation.from_euler("zyx", np.array([-np.pi / 2, -np.pi / 2, 0])).as_matrix()
        T[:3, :3] = R
        T[:3, 3] = np.array(
            [
                -INTERFACE_WIDTH / 2 + LABEL_DEPTH,
                -INTERFACE_DEPTH_FROM_ORIGIN,
                Z_shift,
            ]
        )
        left_label_down = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # Left Label up
        T = np.eye(4)
        R = Rotation.from_euler("zyx", np.array([-np.pi / 2, -np.pi / 2, 0])).as_matrix()
        T[:3, :3] = R
        T[:3, 3] = np.array(
            [
                -INTERFACE_WIDTH / 2 + LABEL_DEPTH,
                -INTERFACE_DEPTH_FROM_ORIGIN,
                -Z_shift + 12.5,
            ]
        )
        left_label_up = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # Right Label down
        T = np.eye(4)
        R = Rotation.from_euler("zyx", np.array([np.pi / 2, np.pi / 2, 0])).as_matrix()
        T[:3, :3] = R
        T[:3, 3] = np.array(
            [
                INTERFACE_WIDTH / 2 - LABEL_DEPTH,
                -INTERFACE_DEPTH_FROM_ORIGIN,
                Z_shift,
            ]
        )
        right_label_down = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # Right Label up
        T = np.eye(4)
        R = Rotation.from_euler("zyx", np.array([np.pi / 2, np.pi / 2, 0])).as_matrix()
        T[:3, :3] = R
        T[:3, 3] = np.array(
            [
                INTERFACE_WIDTH / 2 - LABEL_DEPTH,
                -INTERFACE_DEPTH_FROM_ORIGIN,
                -Z_shift + 12.5,
            ]
        )
        right_label_up = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # All labels
        all_labels = [
            *left_label_down,
            *left_label_up,
            *right_label_down,
            *right_label_up,
            *top_label,
            *back_label,
        ]

        interface_with_label = interface.copy()
        for mesh2 in all_labels:
            interface_with_label = fuse_meshes(interface_with_label, mesh2, fairing_distance=0, operation="difference")
        interface = interface_with_label

    # Align with shape
    T = np.eye(4)
    R = Rotation.from_euler("xyz", np.array([0, 0, -np.pi / 2])).as_matrix()
    T[:3, :3] = R
    T[:3, 3] = np.array([INTERFACE_SHIFT, 0, 0])
    interface.apply_transform(T)

    return interface


if __name__ == "__main__":
    # Test label
    label = "1001"
    interface = load_interface(INTERFACE_PATH, label)
    interface.show()
