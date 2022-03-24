
import trimesh
import pickle
from pathlib import Path

pickle_path = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set\Z\stimulus_set_ 24_03_22__10_31_41.pickle")
with open(pickle_path, "rb") as f:
    shapes = pickle.load(f)

# Shapes Rendered #

# Z_0000
# b_flat-cs_round

shapes[0].mesh.show()

# Z_0001
# b_weak_curve-cs_round

shapes[1].mesh.show()

# Z_0002
# b_strong_curve-cs_round

shapes[2].mesh.show()

# Z_0003
# b_sharp_bend-cs_round

shapes[3].mesh.show()

# Z_0004
# b_hook_f-cs_round

shapes[4].mesh.show()

# Z_0005
# b_hook_r-cs_round

shapes[5].mesh.show()

