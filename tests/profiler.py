### How to profile
# python -m cProfile -o program.prof tests/profiler.py
# snakeviz program.prof

import numpy as np
from objects.utilities import fuse_meshes, transform_sd_mesh
from objects.axial_component import AxialComponent
from objects.shape import Shape
from objects.cross_section import CrossSection
from objects.components import backbone_flat, cp_round, sd_sphere


# Generate mesh1 (shape)
backbone = backbone_flat
cp = cp_round
rotation = 0
cs_list = [CrossSection(cp, i, rotation=rotation) for i in np.linspace(0.1, 0.9, 9)]
ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
s = Shape([ac])

# Transform and fuse with surface deformation
sd_mesh, origin = sd_sphere
origin = np.array(origin)
pos = 0.33
theta_backbone = 0
theta_linear_segment = 0
operation = "union"
sd_mesh_transformed = transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment)
# sd_mesh_rotations = [transform_sd_mesh(sd_mesh, origin, ac, pos, theta_backbone, theta_linear_segment) for theta_linear_segment in np.linspace(0, 2*np.pi, 4, endpoint=False)]
s.combine_meshes([s.mesh, sd_mesh_transformed], operation=operation)
