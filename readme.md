# Objects

Stimuli for visuo-haptic experiments.

## Installation

```bash
# Set up directory
git clone https://github.com/williamsnider/objects.git
cd objects

# Install objects
conda create --name objects_venv python=3.8
pip install -e .

# Compas contains CGAL bindings
conda install -c conda-forge matplotlib trimesh rtree compas compas_cgal igl shapely ipython ipykernel black pytest --yes
```

## TODO: Usage

```python
```

## License
Libigl (and therefore also objects) is licensed under MPL-2.