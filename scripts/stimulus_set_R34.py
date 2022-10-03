### Stimulus set of objects for the R34 application ###
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.stats


BACKBONE_LENGTH = 40
CS_SCALE = BACKBONE_LENGTH / 4


def apply_gaussian_deformation(mesh, idx, height, sigma):

    gaussian = scipy.stats.norm(0, sigma)

    vert = mesh.vertices[idx]
    norm = mesh.vertex_normals[idx]

    # Get distance of all points from that bump
    dists = np.linalg.norm(mesh.vertices - vert, axis=1).reshape(-1, 1)

    # Apply weighted height based on how the distance to each point
    weights = gaussian.pdf(dists) / gaussian.pdf(0)
    bump = height * norm * weights
    new_verts = mesh.vertices + bump

    mesh.vertices = new_verts

    return mesh


#################
### Backbones ###
#################

# straight
NUM_CP = 3
cp = np.array(
    [
        np.linspace(0, BACKBONE_LENGTH, NUM_CP),
        np.zeros(NUM_CP),
        np.zeros(NUM_CP),
    ]
).T
backbone_straight = Backbone(controlpoints=cp, reparameterize=True, name="straight")

# curve_weak
NUM_CP = 3
CURVE_FACTOR = 4
curve_height = BACKBONE_LENGTH / CURVE_FACTOR
cp = np.array(
    [
        np.linspace(0, BACKBONE_LENGTH, NUM_CP),
        [0, curve_height, 0],
        np.zeros(NUM_CP),
    ]
).T
backbone_curve_weak = Backbone(controlpoints=cp, reparameterize=True, name="curve_weak")

# curve_strong
NUM_CP = 3
CURVE_FACTOR = 2
curve_height = BACKBONE_LENGTH / CURVE_FACTOR
cp = np.array(
    [
        np.linspace(0, BACKBONE_LENGTH, NUM_CP),
        [0, curve_height, 0],
        np.zeros(NUM_CP),
    ]
).T
backbone_curve_strong = Backbone(controlpoints=cp, reparameterize=True, name="curve_strong")

######################
### Cross Sections ###
######################
c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 8 * 2 * np.pi), s(0 / 8 * 2 * np.pi)],
        [c(1 / 8 * 2 * np.pi), s(1 / 8 * 2 * np.pi)],
        [c(2 / 8 * 2 * np.pi), s(2 / 8 * 2 * np.pi)],
        [c(3 / 8 * 2 * np.pi), s(3 / 8 * 2 * np.pi)],
        [c(4 / 8 * 2 * np.pi), s(4 / 8 * 2 * np.pi)],
        [c(5 / 8 * 2 * np.pi), s(5 / 8 * 2 * np.pi)],
        [c(6 / 8 * 2 * np.pi), s(6 / 8 * 2 * np.pi)],
        [c(7 / 8 * 2 * np.pi), s(7 / 8 * 2 * np.pi)],
    ]
)

# For clarity, define variables used in the following transformations
cp_x = base_cp[0, 0]  # x coordinate of point we will manipulate
cp_x_prev_next = base_cp[1, 0]  # x coordinate of points on either side of the point we will manipulate
concave_convex_shift = cp_x_prev_next - 0.001  # How much to shift the point for concave/convex cross sections

# concave_high
cp_concave_high = base_cp.copy()
cp_concave_high[0, 0] = cp_x_prev_next - concave_convex_shift
cp_concave_high *= CS_SCALE

# concave_low
# We want the concave low's curvature to be the opposite of the round_high. To do this, we will reflect the controlpoint across the line connecting the current and next controlpoints. In other words, the x-value of this controlpoint will be the same distance away from the previous/next controlpoint's x values, however, it will be closer to the origin.
cp_concave_low = base_cp.copy()
cp_x_flipped = cp_x_prev_next - (cp_x - cp_x_prev_next)  # shift to left of the line
cp_concave_low[0, 0] = cp_x_flipped
cp_concave_low *= CS_SCALE

# elliptical
cp_elliptical = base_cp.copy()
cp_elliptical[:, 0] *= 4 / 5
cp_elliptical[:, 1] *= 6 / 5
cp_elliptical *= CS_SCALE

# round
cp_round = base_cp.copy()
cp_round *= CS_SCALE

# convex - inverse of concave_high
cp_convex = base_cp.copy()
cp_convex[0, 0] = cp_x_prev_next + concave_convex_shift
cp_convex *= CS_SCALE

# plane
cp_plane = base_cp.copy()
cp_plane[0, :] = cp_plane[[1, -1], :].mean(axis=0)
cp_plane *= CS_SCALE


##############
### Shapes ###
##############
NUM_CS = 10
shapes = []

# Cylinder
cs_list = [CrossSection(cp_round, i) for i in np.linspace(0.05, 0.95, 10)]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cylinder_straight")
shapes.append(s)

# Curved cylinder
cs_list = [CrossSection(cp_round, i) for i in np.linspace(0.05, 0.95, 10)]
ac = AxialComponent(backbone=backbone_curve_strong, cross_sections=cs_list)
s = Shape([ac], label="cylinder_curved")
shapes.append(s)

# Cone
cs_list = [
    CrossSection(cp_round * 1 / 3, 0.05),
    CrossSection(cp_round * 2 / 3, 0.5),
    CrossSection(cp_round * 3 / 3, 0.95),
]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cone")
shapes.append(s)

# Bugle
scale_geomspace = np.geomspace(1 / 3, 4 / 3, NUM_CS)
location_linspace = np.linspace(0.05, 0.95, NUM_CS)
cs_list = [CrossSection(cp_round * scale_geomspace[i], location_linspace[i]) for i in range(NUM_CS)]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="bugle")
shapes.append(s)

# Hourglass
scale_geomspace = np.geomspace(1 / 3, 4 / 3, NUM_CS)
scale_geomspace[: NUM_CS // 2] = scale_geomspace[: NUM_CS // 2 - 1 : -1]
location_linspace = np.linspace(0.05, 0.95, NUM_CS)
cs_list = [CrossSection(cp_round * scale_geomspace[i], location_linspace[i]) for i in range(NUM_CS)]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="hourglass")
shapes.append(s)

# Football
cs_list = [
    CrossSection(cp_round * 1 / 2, 0.05),
    CrossSection(cp_round * 3 / 2, 0.5),
    CrossSection(cp_round * 1 / 2, 0.95),
]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="football")
shapes.append(s)

# Cylinder concave_low
cs_list = [
    CrossSection(cp_concave_low, 0.05),
    CrossSection(cp_concave_low, 0.5),
    CrossSection(cp_concave_low, 0.95),
]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cylinder_concave_low")
shapes.append(s)

# Cylinder concave_high
cs_list = [
    CrossSection(cp_concave_high, 0.05),
    CrossSection(cp_concave_high, 0.5),
    CrossSection(cp_concave_high, 0.95),
]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cylinder_concave_high")
shapes.append(s)

# Cylinder convex
cs_list = [
    CrossSection(cp_convex, 0.05),
    CrossSection(cp_convex, 0.5),
    CrossSection(cp_convex, 0.95),
]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cylinder_convex")
shapes.append(s)

# Cylinder elliptical
cs_list = [
    CrossSection(cp_elliptical, 0.05),
    CrossSection(cp_elliptical, 0.5),
    CrossSection(cp_elliptical, 0.95),
]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cylinder_elliptical")
shapes.append(s)

# Cylinder local concavity (low)
num_cs = 9
cs_list = [CrossSection(cp_round, i) for i in np.linspace(0.05, 0.95, num_cs)]
cs_list[(num_cs - 1) // 2] = CrossSection(cp_concave_low, 0.5)
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cylinder_local_concave_low")
shapes.append(s)

# Cylinder local concavity (low) x2
num_cs = 9
location_linspace = np.linspace(0.05, 0.95, num_cs)
cs_list = [CrossSection(cp_round, i) for i in location_linspace]
cs_list[2] = CrossSection(cp_concave_low, location_linspace[2])
cs_list[6] = CrossSection(cp_concave_low, location_linspace[6])
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cylinder_local_concave_low_x2")
shapes.append(s)

# Cylinder local concavity (low) x2 rotated
num_cs = 9
location_linspace = np.linspace(0.05, 0.95, num_cs)
cs_list = [CrossSection(cp_round, i) for i in location_linspace]
cs_list[2] = CrossSection(cp_concave_low, location_linspace[2])
cs_list[6] = CrossSection(cp_concave_low, location_linspace[6], rotation=np.pi / 2)
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
s = Shape([ac], label="cylinder_local_concave_low_x2_rotated")
shapes.append(s)


for s in shapes:

    s.create_interface()
    s.fuse_mesh_to_interface()

    if "local" in s.label:
        eul = -np.pi / 4
    else:
        eul = -np.pi / 2
    r = R.from_euler("xyz", [0, np.pi / 8, 0]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    s.save_mesh_as_png("./samples", rotation=T, interface=True)
