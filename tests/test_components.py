# Test that the segments listed in objects.components produce correct shapes

from objects.components import sd_cylinder, sd_curved_cylinder, sd_sphere, sd_ellipsoid, sd_cone_forward, sd_curved_elliptical_cylinder, sd_elliptical_cylinder
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.backbone import Backbone
from objects.utilities import transform_sd_mesh
from objects.parameters import cs_scale_backbone

import numpy as np

# Create base cross section
c = np.cos
s = np.sin
base_cp = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [0.1, 0.1],
    ]
)

base_cp_round = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
    ]
)

# Create base backbone
cp = np.array(
    [
        [0, 0, 0],
        [0, 10, 0],
        [0, 20, 0],
        [0, 30, 0],
        [10, 40, 0],
        [20, 50, 0],
        [30, 60, 0],
    ]
)
backbone1 = Backbone(cp, reparameterize=False)


# def test_fuse_surface_deformation():

#     for sd_mesh, origin in [sd_cylinder, sd_curved_cylinder, sd_sphere, sd_ellipsoid, sd_cone_forward, sd_curved_elliptical_cylinder, sd_elliptical_cylinder,]:
        
#         ac = AxialComponent(backbone1, [CrossSection(base_cp_round*cs_scale_backbone, i) for i in np.linspace(0.1, 0.9, 5)])
#         s = Shape([ac])
#         theta_backbone = np.pi
#         theta_linear_segment = 3*np.pi/2
#         pos=0.25
#         sd_mesh_transformed = transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment)
#         # sd_mesh_rotations = [transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment) for theta_linear_segment in np.linspace(0, 2*np.pi, 4, endpoint=False)]

#         s.combine_meshes([s.mesh, sd_mesh_transformed])
#         s.mesh.show()

def test_fuse_surface_deformation_concave():

    for sd_mesh, origin in [sd_cylinder]:
        
        ac = AxialComponent(backbone1, [CrossSection(base_cp_round*cs_scale_backbone, i) for i in np.linspace(0.1, 0.9, 5)])
        s = Shape([ac])
        theta_backbone = np.pi
        theta_linear_segment = 3*np.pi/2
        pos=0.25
        sd_mesh_transformed = transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment)
        # sd_mesh_rotations = [transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment) for theta_linear_segment in np.linspace(0, 2*np.pi, 4, endpoint=False)]

        s.combine_meshes([s.mesh, sd_mesh_transformed], operation='difference')
        s.mesh.show()

if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_components.py"])
