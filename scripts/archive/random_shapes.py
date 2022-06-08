# Make several random shapes

from objects.axial_component import AxialComponent
from objects.cross_section import CrossSection
from objects.shape import Shape
import numpy as np

# Set random seed for debugging
# np.random.seed(1)

CP_GAUSSIAN_SIGMA = 10
CS_RADIUS_RANGE = [20, 40]
CS_NUM_RANGE = [2, 4]
CS_TILT_RANGE = [-np.pi / 8, np.pi / 8]
AC_LENGTH_RANGE = [25, 100]
AC_CURVATURE_RANGE = [0, 10]
AC_EULER_ANGLES_RANGE = [0, np.pi]
AC_POSITION_RANGE = [0, 1]
AC_NUM_RANGE = [1, 1]
ALIGN_OBB = False

c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
    ]
)


def make_random_cs(base_cp, position, CP_GAUSSIAN_SIGMA, CS_RADIUS_RANGE):

    # Scale base_cp
    scale = np.random.uniform(low=CS_RADIUS_RANGE[0], high=CS_RADIUS_RANGE[1])
    cp = base_cp * scale

    # Shift individual cp
    shift = np.random.normal(scale=CP_GAUSSIAN_SIGMA, size=cp.shape)
    cp += shift

    # Rotation -implement later (more relevant for GA)

    # Tilt angle
    tilt = np.random.uniform(low=CS_TILT_RANGE[0], high=CS_TILT_RANGE[1])

    return CrossSection(cp, position, tilt=tilt)


def make_random_ac(AC_LENGTH_RANGE, AC_CURVATURE_RANGE, CS_NUM_RANGE, AC_EULER_ANGLES_RANGE, parent=None):

    # Length
    length = np.random.uniform(low=AC_LENGTH_RANGE[0], high=AC_LENGTH_RANGE[1])

    # Curvature - not implemented
    curvature = 0

    # Euler angles
    euler_angles = np.random.uniform(low=AC_EULER_ANGLES_RANGE[0], high=AC_EULER_ANGLES_RANGE[1], size=3).T

    # Generate cross sections
    num_cs = np.random.randint(low=CS_NUM_RANGE[0], high=CS_NUM_RANGE[1] + 1)
    cs_positions = np.sort(np.random.uniform(size=num_cs))
    cross_sections = []
    for i in range(num_cs):
        position = cs_positions[i]
        cs = make_random_cs(base_cp, position, CP_GAUSSIAN_SIGMA, CS_RADIUS_RANGE)
        cross_sections.append(cs)

    # Position along self

    # Position along parent

    return AxialComponent(length, curvature, cross_sections, euler_angles, parent_axial_component=parent)


def make_random_shape(AC_NUM_RANGE, label="test", save_dir=None):

    num_ac = np.random.randint(low=AC_NUM_RANGE[0], high=AC_NUM_RANGE[1] + 1)
    ac_list = []
    for i in range(num_ac):

        # Assign parent
        if i == 0:
            parent = None
        else:
            parent = np.random.choice(ac_list)

        # Create Axial Component
        ac = make_random_ac(AC_LENGTH_RANGE, AC_CURVATURE_RANGE, CS_NUM_RANGE, AC_EULER_ANGLES_RANGE, parent=parent)
        ac_list.append(ac)

    return Shape(ac_list, align_OBB=ALIGN_OBB, fuse_to_interface=True, label=label, save_dir=save_dir)


s = make_random_shape(AC_NUM_RANGE)
s.mesh.show()
# cs0 = CrossSection(base_cp * 0.5, 0.3)
# cs1 = CrossSection(base_cp * 0.5, 0.7)
# ac1 = AxialComponent(2 * np.pi * 1 * 0.25, curvature=0, cross_sections=[cs0, cs1])
# ac2 = AxialComponent(
#     2 * np.pi * 1 * 0.25,
#     curvature=1 / 1,
#     cross_sections=[cs0, cs1],
#     parent_axial_component=ac1,
#     position_along_parent=0.75,
#     position_along_self=0.0,
#     euler_angles=np.array([0, np.pi / 3, 0]),
# )
# s = Shape([ac1, ac2])
# s.fuse_meshes(ac1.mesh, ac2.mesh)
