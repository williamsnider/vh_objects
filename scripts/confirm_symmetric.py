from pathlib import Path
import trimesh
from stimulus_set_params import SEGMENT_LENGTH, X_WIDTH


# Load file
fname = Path("sample_shapes/stimulus_set_C/stl/C657.stl")
mesh = trimesh.load_mesh(fname)

# Create trimesh scene
scene = trimesh.Scene()


# Insert plane into scene
plane = trimesh.creation.box((30,30,0.1))
plane.apply_translation([20,0,20])

# Show scene
scene.add_geometry([mesh, plane])
scene.show()

