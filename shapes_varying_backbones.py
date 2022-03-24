
import trimesh
import pickle
from pathlib import Path

pickle_path = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set\Z\stimulus_set_ 24_03_22__12_49_50.pickle")
with open(pickle_path, "rb") as f:
    (shapes, transitions_dict) = pickle.load(f)

# Shapes Rendered

shapes[0].mesh_with_interface.show()

# Above shape is: Z_0000
# b_flat-cs_round

shapes[1].mesh_with_interface.show()

# Above shape is: Z_0001
# b_weak_curve-cs_round

shapes[2].mesh_with_interface.show()

# Above shape is: Z_0002
# b_strong_curve-cs_round

shapes[3].mesh_with_interface.show()

# Above shape is: Z_0003
# b_sharp_bend-cs_round

shapes[4].mesh_with_interface.show()

# Above shape is: Z_0004
# b_hook_f-cs_round

shapes[5].mesh_with_interface.show()

# Above shape is: Z_0005
# b_hook_r-cs_round

