### Stimulus set of objects for the R34 application ###
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.utilities import approximate_arc, get_deformation_vertex, get_deformation_points_along_plane
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy
from pathlib import Path
from objects.utilities import approximate_arc

BACKBONE_LENGTH = 40
CS_SCALE = BACKBONE_LENGTH / 4


def align_backbone_center(backbone):
    # Rotate to reach T(0.5) == +X axis
    original = np.vstack([backbone.T(0.5), backbone.N(0.5), backbone.B(0.5)])
    goal = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    rot = goal @ np.linalg.inv(original)
    cp = backbone.controlpoints @ rot
    return Backbone(controlpoints=cp, reparameterize=True)


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
shapes = []
NUM_CS = 10

thickness_list = [6 / 8, 7 / 8, 8 / 8, 9 / 8, 10 / 8]
curvature_list = np.linspace(np.pi / 2, 0, 5, endpoint=False)[::-1]
# # Varying cylinder thickness
# for i, scale in enumerate(thickness_list):
#     cs_list = [CrossSection(cp_round * scale, i) for i in np.linspace(0.05, 0.95, NUM_CS)]
#     ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
#     s = Shape([ac], label="cylinder_straight_{}".format(i))
#     s.save_dir = "varying_thickness_cylinder"
#     shapes.append(s)


# # Varying curvature of constant curvature medial-axis
# scale = 0.75
# for i, angle in enumerate(curvature_list):
#     arc_array = approximate_arc(angle, BACKBONE_LENGTH)
#     backbone = Backbone(controlpoints=arc_array, reparameterize=True)

#     # Rotate so middle is parallel to +x axis
#     backbone = align_backbone_center(backbone)

#     # Shift down 3mm, just for plotting
#     cp = backbone.controlpoints
#     cp[:, 1] -= 2
#     backbone = Backbone(controlpoints=cp, reparameterize=True)

#     # cp[:,]

#     cs_list = [CrossSection(cp_round * scale, i) for i in np.linspace(0.05, 0.95, NUM_CS)]
#     ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
#     s = Shape([ac], label="cylinder_curvature_{}".format(i))
#     s.save_dir = "varying_curvature"

#     shapes.append(s)

# # # Varying degree of bend for sharp-bend medial-axis
# # scale = 0.75
# # for bend in [np.pi / 9, 2 * np.pi / 9, 3 * np.pi / 9]:
# #     cp = np.array(
# #         [
# #             np.array(BACKBONE_LENGTH) * np.array([0, 0.40, 0.60, 1.0]),
# #             np.zeros(4),
# #             np.zeros(4),
# #         ]
# #     ).T
# #     midpoint = np.array([BACKBONE_LENGTH / 2, 0, 0])
# #     T = R.from_euler("z", -bend)
# #     cp_rot = (cp - midpoint) @ T.as_matrix() + midpoint
# #     cp[-2:] = cp_rot[-2:]

# #     backbone = Backbone(controlpoints=cp, reparameterize=True)
# #     cs_list = [CrossSection(cp_round * scale, i) for i in [0.05, 0.10, 0.40, 0.60, 0.90, 0.95]]
# #     ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
# #     s = Shape([ac], label="sharp_bend_{}".format(str(round(bend, 2)).replace(".", "-")))
# #     shapes.append(s)
# #     s.mesh.show()

# # Cross section across shape
# shift_list = np.linspace(0.5, 1.5, 6)
# for i, shift in enumerate(shift_list):
#     cs_cp = base_cp.copy()
#     cs_cp[0, 0] *= shift
#     cs_cp *= CS_SCALE
#     cs_list = [CrossSection(cs_cp, i, rotation=np.pi / 3) for i in np.linspace(0.05, 0.95, NUM_CS)]
#     ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
#     s = Shape([ac], label="concave_convex_{}".format(i))
#     s.save_dir = "cross_section_across_shape"
#     shapes.append(s)

# # Cone
# cs_list = [
#     CrossSection(cp_round * 1 / 3, 0.05),
#     CrossSection(cp_round * 2 / 3, 0.5),
#     CrossSection(cp_round * 3 / 3, 0.95),
# ]
# ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
# s = Shape([ac], label="cone")
# s.save_dir = "varying_round_cross_section_size"

# shapes.append(s)

# # Bugle
# scale_geomspace = np.geomspace(2 / 5, 4 / 3, NUM_CS)
# location_linspace = np.linspace(0.05, 0.95, NUM_CS)
# cs_list = [CrossSection(cp_round * scale_geomspace[i], location_linspace[i]) for i in range(NUM_CS)]
# ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
# s = Shape([ac], label="bugle")
# s.save_dir = "varying_round_cross_section_size"
# shapes.append(s)

# # Hourglass
# scale_geomspace = np.geomspace(2 / 5, 4 / 3, NUM_CS)
# scale_geomspace[: NUM_CS // 2] = scale_geomspace[: NUM_CS // 2 - 1 : -1]
# location_linspace = np.linspace(0.05, 0.95, NUM_CS)
# cs_list = [CrossSection(cp_round * scale_geomspace[i], location_linspace[i]) for i in range(NUM_CS)]
# ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
# s = Shape([ac], label="hourglass")
# s.save_dir = "varying_round_cross_section_size"
# shapes.append(s)

# # Football
# cs_list = [
#     CrossSection(cp_round * 2 / 5, 0.05),
#     CrossSection(cp_round * 3 / 2, 0.5),
#     CrossSection(cp_round * 2 / 5, 0.95),
# ]
# ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
# s = Shape([ac], label="football")
# s.save_dir = "varying_round_cross_section_size"
# shapes.append(s)

# Surface deformations

cs_list = [CrossSection(cp_round, i) for i in np.linspace(0.05, 0.95, NUM_CS)]
ac = AxialComponent(backbone=backbone_straight, cross_sections=cs_list)
base_cylinder = Shape([ac], label="cylinder")


# return mesht:
# loc_dict[t] = {}
# for ang in ang_list:
#     pts, normals = get_deformation_vertex(base_cylinder.mesh, base_cylinder.ac_list[0], t, N_rotation=ang)
#     loc_dict[t][ang] = {"pts": pts, "normals": normals}


class Deformation:
    def __init__(self, pts, normals, magnitude, sigma, deformation_type, t, ang):
        self.pts = pts
        self.normals = normals
        self.magnitude = magnitude
        self.sigma = sigma
        self.deformation_type = deformation_type
        self.t = t
        self.ang = ang

    def __eq__(self, other):

        if type(other) != type(self):
            raise TypeError
        else:

            return all(
                [
                    self.magnitude == other.magnitude,
                    self.sigma == other.sigma,
                    self.deformation_type == other.deformation_type,
                    self.t == other.t,
                    self.ang == other.ang,
                ]
            )


t_list = [1 / 3, 2 / 3]
ang_list = [np.pi / 4, 3 * np.pi / 4]
magnitude_list = [-3, 3]
sigma_list = [2]

# List of deformation types
deformation_types = []
deformation_groups = []

for i in range(0, 3):

    if i == 0:
        base = base_cylinder.copy()
        base_dir = "surface_deformations/cylinder"
    elif i == 1:

        # Varying curvature of constant curvature medial-axis
        scale = 1
        angle = curvature_list[1]
        arc_array = approximate_arc(angle, BACKBONE_LENGTH)
        backbone = Backbone(controlpoints=arc_array, reparameterize=True)

        # Rotate so middle is parallel to +x axis
        backbone = align_backbone_center(backbone)

        # Shift down 3mm, just for plotting
        cp = backbone.controlpoints
        cp[:, 1] -= 2
        backbone = Backbone(controlpoints=cp, reparameterize=True)

        # cp[:,]

        cs_list = [CrossSection(cp_round * scale, i) for i in np.linspace(0.05, 0.95, NUM_CS)]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac], label="cylinder_curvature_{}".format(i))

        base = s.copy()
        base_dir = "surface_deformations/curved_cylinder_0"

    elif i == 2:

        # Varying curvature of constant curvature medial-axis
        scale = 1
        angle = curvature_list[3]
        arc_array = approximate_arc(angle, BACKBONE_LENGTH)
        backbone = Backbone(controlpoints=arc_array, reparameterize=True)

        # Rotate so middle is parallel to +x axis
        backbone = align_backbone_center(backbone)

        # Shift down 3mm, just for plotting
        cp = backbone.controlpoints
        cp[:, 1] -= 2
        backbone = Backbone(controlpoints=cp, reparameterize=True)

        # cp[:,]

        cs_list = [CrossSection(cp_round * scale, i) for i in np.linspace(0.05, 0.95, NUM_CS)]
        ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
        s = Shape([ac], label="cylinder_curvature_{}".format(i))
        base = s.copy()
        base_dir = "surface_deformations/curved_cylinder_1"
    else:
        raise NotImplementedError

    # Single vertex
    count = 0
    for magnitude in magnitude_list:
        for sigma in sigma_list:
            for t in t_list:
                for ang in ang_list:

                    # Apply to base shape
                    s = base.copy()
                    # s.label = "magnitude-{}_sigma-{}_t-{}_ang-{}".format(
                    #     *[str(round(x, 1)).replace(".", "p") for x in [magnitude, sigma, t, ang]]
                    # )
                    s.label = "point_single_{}".format(count)
                    count += 1
                    pts, normals = get_deformation_vertex(base.mesh, base.ac_list[0], t, N_rotation=ang)

                    # Add to list
                    d = Deformation(pts, normals, magnitude, sigma, "bump", t, ang)
                    deformation_types.append(d)

                    if ang != 0.0:
                        continue

                    s.apply_gaussian_deformation(pts, normals, magnitude, sigma)
                    s.save_dir = base_dir
                    shapes.append(s)

    # Double vertex
    deformation_groups = []
    for d1 in deformation_types:
        for d2 in deformation_types:

            if (d1.t == d2.t) & (d1.ang == d2.ang):
                continue

            # Omit pairs in same location
            if np.all(d1.pts == d2.pts):
                continue

            # Ensure deformation not already found (i.e. switching d1 and d2 but the result is the same)
            already_in = False
            for g in deformation_groups:
                if (d1 in g) & (d2 in g):
                    already_in = True

            if already_in:
                continue

            # Omit pairs at angle!=0 (redundant on base cylinder)
            if (d1.ang != ang_list[0]) & (d2.ang != ang_list[0]):
                continue
            else:
                deformation_groups.append([d1, d2])

    for count, group in enumerate(deformation_groups):
        s = base.copy()
        s.label = "point_double_{}".format(count)
        s.save_dir = base_dir

        for d in group:
            s.apply_gaussian_deformation(d.pts, d.normals, d.magnitude, d.sigma)
        shapes.append(s)

    # Ridge
    count = 0
    ridge_list = []
    for magnitude in magnitude_list:
        for sigma in sigma_list:
            for t in t_list:
                s = base.copy()
                s.label = "ridge_single_{}".format(count)
                count += 1
                s.save_dir = base_dir
                mesh = s.mesh
                N = s.ac_list[0].T(t)
                point = s.ac_list[0].r(t)
                pts, normals = get_deformation_points_along_plane(mesh, N, point)
                s.apply_gaussian_deformation(pts, normals, magnitude, sigma)
                shapes.append(s)

                ridge_list.append(Deformation(pts, normals, magnitude, sigma, "Ridge", t, ang=0.0))

    # Double ridge
    count = 0
    for r1 in ridge_list:
        for r2 in ridge_list:

            if r1.t == r2.t:
                continue

            if r1.t != t_list[0]:
                continue

            s = base.copy()

            s.label = "ridge_double_{}".format(count)
            count += 1
            s.save_dir = base_dir
            for r in [r1, r2]:
                s.apply_gaussian_deformation(r.pts, r.normals, r.magnitude, r.sigma)
            shapes.append(s)

    # for group in deformation_groups:
    #     s = base.copy()
    #     s.label = "NeedsALabel"
    #     for d in group:
    #         s.apply_gaussian_deformation(d.pts, d.normals, d.magnitude, d.sigma)
    #     s.mesh.show(smooth=False)

    # # Ridge and vertex
    # ridge_list = []
    # for magnitude in magnitude_list:
    #     for sigma in sigma_list:
    #         for t in t_list:
    #             s = base.copy()
    #             mesh = s.mesh
    #             N = s.ac_list[0].T(t)
    #             point = s.ac_list[0].r(t)
    #             pts, normals = get_deformation_points_along_plane(mesh, N, point)
    #             ridge_list.append(Deformation(pts, normals, magnitude, sigma, "Ridge", t, 0.0))

    groups = []
    count = 0
    for ridge in ridge_list:
        for d1 in deformation_types:

            if ridge.t == d1.t:
                continue

            if d1.ang != ang_list[0]:
                continue

            g = [ridge, d1]

            s = base.copy()
            s.label = "ridge_and_point_{}".format(count)
            s.save_dir = base_dir
            count += 1
            for d in g:
                s.apply_gaussian_deformation(d.pts, d.normals, d.magnitude, d.sigma)
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
    save_dir = Path("./samples", s.save_dir)
    s.save_mesh_as_png(save_dir, rotation=T, interface=True)
