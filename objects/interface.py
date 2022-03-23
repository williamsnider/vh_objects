import numpy as np
from freetype import Face
import trimesh
from shapely import geometry
from objects.parameters import FONT_PATH, INTERFACE_WIDTH, LABEL_DEPTH, POST_LENGTH, POST_OFFSET, POST_RADIUS
from objects.utilities import calc_R_euler_angles, fuse_meshes

# Parameters for font face glyph
HEIGHT = 2200
WIDTH = 1600

########################
### Helper functions ###
########################


def get_character_outline(char, shift=(0, 0)):
    assert len(char) == 1
    face = Face(str(FONT_PATH))
    face.set_char_size(48 * 64)  # If changing, must also change HEIGHT and WIDTH in calc_offsets
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
        scale = INTERFACE_WIDTH / (WIDTH * NUM_CHARS_PER_LINE_WITH_ADDITIONAL)
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
    height_offsets = np.arange(HEIGHT * num_lines, 0, -HEIGHT)
    if num_lines % 2 == 1:
        height_offsets -= height_offsets[(num_lines - 1) // 2] + HEIGHT // 2
    else:
        height_offsets -= height_offsets[num_lines // 2] + HEIGHT

    # Calculate width offsets based on number of charactes in each line
    offsets = []
    for line_num, line in enumerate(lines):
        num_chars = len(line)

        width_offsets = np.arange(0, WIDTH * num_chars, WIDTH)
        if num_chars % 2 == 1:
            width_offsets -= width_offsets[(num_chars) // 2] + WIDTH // 2
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


class Interface:
    """
    What the robot will grab onto
    """

    def __init__(self, stl_path, label="test_123"):

        self.mesh = trimesh.load_mesh(stl_path)
        self.label = label
        self.apply_label()
        self.transform_to_correct_pose()

    def apply_label(self):
        """Extrudes the interface's label onto the mesh.

        Splits label lines by underscores."""

        # Generate mesh of label
        label_meshes = text_to_mesh(self.label, LABEL_DEPTH)

        # Transform to be on correct side of interface
        goal_center = np.array([-INTERFACE_WIDTH / 2 + LABEL_DEPTH, 0, -INTERFACE_WIDTH / 2])
        T = np.eye(4)
        T[:3, :3] = calc_R_euler_angles([np.pi / 2, 0, 3 * np.pi / 2])
        T[:3, 3] = goal_center
        label_meshes_transformed = [mesh.copy().apply_transform(T) for mesh in label_meshes]

        # Compute difference between interface mesh and each character mesh
        difference_mesh = self.mesh.copy()
        for mesh2 in label_meshes_transformed:
            difference_mesh = fuse_meshes(difference_mesh, mesh2, fairing_distance=0, operation="difference")
        self.mesh = difference_mesh

    def transform_to_correct_pose(self):
        """Transforms the interface so that it is away from the shape (with endpoint (0,0,0))."""

        new_interface = self.mesh.copy()
        goal_position = np.array([0, -POST_LENGTH, 0])
        T = np.eye(4)
        T[:3, :3] = calc_R_euler_angles([3 * np.pi / 2, 0, 0])
        T[:3, 3] = goal_position
        new_interface = new_interface.apply_transform(T)
        self.mesh = new_interface
