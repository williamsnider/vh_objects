# Add label to interface
import trimesh
import numpy as np
from freetype import Face
import trimesh
from shapely import geometry
from objects.utilities import calc_R_euler_angles, fuse_meshes
from pathlib import Path
from scipy.spatial.transform import Rotation
from objects.parameters import (
    FONT_PATH,
    INTERFACE_WIDTH,
    FONT_OFFSET_WIDTH,
    FONT_OFFSET_HEIGHT,
    LABEL_DEPTH,
    INTERFACE_DEPTH_FROM_ORIGIN,
    INTERFACE_HEIGHT_ABOVE_ORIGIN,
    INTERFACE_PATH,
    INTERFACE_SHIFT,
)

########################
### Helper functions ###
########################


def get_character_outline(char, shift=(0, 0)):
    assert len(char) == 1
    face = Face(str(FONT_PATH))
    face.set_char_size(24 * 64)  # If changing, must also change HEIGHT and WIDTH in calc_offsets
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

        # Scale points to fit onto interface. Interface is 25.4mm in width. We partition that as if we have 6 characters per line (max is 5 in reality).
        NUM_CHARS_PER_LINE_WITH_ADDITIONAL = 5
        scale = INTERFACE_WIDTH / (FONT_OFFSET_WIDTH * NUM_CHARS_PER_LINE_WITH_ADDITIONAL)
        contour_points *= scale

        start = end

        # Append only if there are at least 4 contour points (including wrapping the first)
        if contour_points.shape[0] >= 4:
            contours_list.append(contour_points)

    return contours_list


def calc_offsets(text):

    lines = text.split("_")
    num_lines = len(lines)

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


def text_to_mesh(text, extrusion_height):

    # Extract characters and calculate offsets for centering lines/characters
    chars, offsets = calc_offsets(text)

    # Gather outlines of characters
    text_list = []
    for line_num, line in enumerate(chars):

        assert len(line) <= 4, "No more than 4 characters allowed per line."
        for char_num, char in enumerate(line):

            offset = offsets[line_num][char_num]
            contours_list = get_character_outline(char, shift=offset)
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

    import trimesh

    meshes = []
    for poly in polygons:
        meshes.append(
            trimesh.creation.extrude_polygon(poly, extrusion_height * 2)
        )  # Double extrusion since half will not be intersecting the interface (i.e. need to double or else the actual depth of the label will not equal the requested depth).

    return meshes


def load_interface(stl_path, label=None):

    """Label uses underscores to split into different lines"""
    interface = trimesh.load_mesh(stl_path)

    if label != None:
        # Generate mesh of label
        label_meshes = text_to_mesh(label, LABEL_DEPTH)

        # Top label
        T = np.eye(4)
        T[:3, 3] = np.array([0, -INTERFACE_DEPTH_FROM_ORIGIN, INTERFACE_HEIGHT_ABOVE_ORIGIN - LABEL_DEPTH])
        top_label = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # Left Label
        T = np.eye(4)
        R = Rotation.from_euler("zyx", np.array([-np.pi / 2, -np.pi / 2, 0])).as_matrix()
        T[:3, :3] = R
        T[:3, 3] = np.array([-INTERFACE_WIDTH / 2 + LABEL_DEPTH, -INTERFACE_DEPTH_FROM_ORIGIN * 3 / 2, -26 / 2])
        left_label = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # Right Label
        T = np.eye(4)
        R = Rotation.from_euler("zyx", np.array([np.pi / 2, np.pi / 2, 0])).as_matrix()
        T[:3, :3] = R
        T[:3, 3] = np.array([INTERFACE_WIDTH / 2 - LABEL_DEPTH, -INTERFACE_DEPTH_FROM_ORIGIN * 3 / 2, -26 / 2])
        right_label = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # All labels
        all_labels = [*top_label, *left_label, *right_label]

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
    label = "0_0_0_1"  # Splits lines based on underscores
    interface = load_interface(INTERFACE_PATH, label)
    interface.show()
