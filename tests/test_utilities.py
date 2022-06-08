import numpy as np
from objects.utilities import fuse_meshes, transform_sd_mesh
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.cross_section import CrossSection
import copy


# def test_fuse_meshes():
#     """Test that two meshes are fused (and faired) properly."""

#     # Generate mesh1 (shape)
#     backbone = backbone_flat
#     cp = cp_round
#     rotation = 0
#     cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 9)]
#     ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
#     s = Shape([ac])

#     # Transform and fuse with surface deformation
#     sd_mesh, origin = sd_sphere
#     origin = np.array(origin)
#     pos = 0.33
#     theta_backbone = 0
#     theta_linear_segment = 0
#     operation = "union"
#     sd_mesh_transformed = transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment)
#     # sd_mesh_rotations = [transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment) for theta_linear_segment in np.linspace(0, 2*np.pi, 4, endpoint=False)]
#     s.combine_meshes([s.mesh, sd_mesh_transformed], operation=operation)
#     s.mesh.show()


# def test_transform_sd_mesh():
#     """Test that the surface deformation mesh is aligned properly."""

#     # Varying surface deformations

#     for backbone in [backbone_weak_curve]:
#         for cp in [cp_round]:

#             # Generate base shape
#             rotation = 0
#             cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 9)]
#             ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
#             base_shape = Shape([ac])

#             # Apply surface deformations
#             for sd in [sd_curved_cylinder]:
#                 sd_mesh, origin = sd
#                 for theta_backbone in [3 * np.pi / 2]:

#                     s = copy.deepcopy(base_shape)
#                     for theta_linear_segment in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:

#                         for operation in ["union"]:

#                             for pos_list in [[0.33]]:
#                                 for pos in pos_list:

#                                     sd_mesh_transformed = transform_sd_mesh(
#                                         sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment
#                                     )
#                                     # sd_mesh_rotations = [transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment) for theta_linear_segment in np.linspace(0, 2*np.pi, 4, endpoint=False)]
#                                     s.combine_meshes([s.mesh, sd_mesh_transformed], operation=operation)
#                     s.mesh.show()


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_utilities.py"])
