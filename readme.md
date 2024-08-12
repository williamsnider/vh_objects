# vh_objects

Creating 3D stimuli for visuohaptic experiments.

![Sample shape with interface](assets/shape_rotation.gif)

The above 3D stimulus contains a graspable region as well as a robot interface region, which allows a robotic arm to deliver it to subjects during experiments.

## Motivation

To study the neural basis of 3D shape perception, we need to generate 3D stimuli that are

- parameterized along a wide range of shape feature dimensions (curvature, medial-axis structure, etc.)
- generated using a small number of parameters, making them amenable to a genetic algorithm approach to explore shape space
- renderable (for visual presentations) and watertight / 3D-printable (for haptic presentations)

By using B-Spline curves and surfaces, _vh_objects_ generates stimuli that fulfill these requirements.

## Stimulus structure

In brief, stimuli are generated using a backbone and cross sections to form axial components, which can be fused together to create a final shape.

- _backbones_ are B-Spline curves that form the medial-axis structure of an axial component
- _cross_sections_ are closed B-Spline curves located along a backbone, controlling how the surface curves along an axial component
- _axial_components_ are B-Spline surfaces formed using the controlpoints of cross_sections located along a backbone. The surface is sampled to form a watertight, triangular mesh that can be rendered and 3D-printed.
- _shapes_ are one or more axial_components which are fused together to create complex stimuli. They are also fused to a post and a robot interface region, which allows a robotic arm to pick/place it.

## Usage

```python
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
import numpy as np

# Generate a backbone from controlpoints (quadratic b-spline)
backbone_cp = np.array(
    [
        [ 0.,  0.,  0.],
        [20.,  0.,  0.],
        [40.,  0.,  0.],
        [60.,  0.,  0.],
        [80.,  0.,  0.]
    ]
)
backbone = Backbone(controlpoints = backbone_cp, reparameterize=True)

# Generate cross sections of varying shape
c = np.cos
s = np.sin
round_cp = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
    ]
)
concave_cp = round_cp.copy()
concave_cp[-1] = [0.5, 0.5]
cs0 = CrossSection(controlpoints = round_cp * 20, position = 0.1)
cs1 = CrossSection(controlpoints = concave_cp * 10, position = 0.5)
cs2 = CrossSection(controlpoints = concave_cp*20, position = 0.9)

# Generate an axial component, which uses the cross sections positioned along the backbone to form a quadratic b-spline surface
ac = AxialComponent(backbone=backbone, cross_sections=[cs0, cs1, cs2])

# Generate a shape (which can combine multiple axial components)
s = Shape([ac], label='Test_1234')
s.mesh.show()

# Fuse the shape to an interface (that the robotic arm can grasp)
s.create_interface()
s.fuse_mesh_to_interface()
s.mesh_with_interface.show()
```

## Installation

```bash
# Set up directory
git clone https://github.com/williamsnider/vh_objects.git
cd vh_objects

# # Install vh_objects
# conda create --name vh_objects_venv python=3.7
# conda activate vh_objects_venv
# pip install -e .

# # Install other packages from conda (compas easily installable via conda)
# # conda install -c conda-forge matplotlib trimesh rtree compas compas_cgal igl shapely opencv ipython ipykernel ipympl pytest --yes

# conda install -c conda-forge compas compas_cgal igl shapely trimesh matplotlib  ipython pytest

# conda install ipython pytest  scipy 
# pip install matplotlib opencv-python




# conda create --name vh_objects_venv
# conda activate vh_objects
# conda install -c conda-forge compas compas_cgal igl
# pip install -e .
# conda install -c conda-forge shapely trimesh matplotlib ipython pytest


conda create --name vh_objects_venv -c conda-forge python=3.9 compas compas_cgal igl shapely trimesh matplotlib ipython pytest
conda activate vh_objects_venv
pip install -e .
# Received ImportError: libscip.so.9.0: cannot open shared object file: No such file or directory, so installed SCIP from source https://scipopt.org/index.php#download

# Added location of libscip.so.9.0 to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Downgraded pyglet to be compatible with trimesh viewer
pip install pyglet==1.5

```


