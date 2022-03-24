
import trimesh
import pickle
from pathlib import Path

pickle_path = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set\Z\stimulus_set_ 24_03_22__12_49_50.pickle")
with open(pickle_path, "rb") as f:
    (shapes, transitions_dict) = pickle.load(f)

# Shapes Rendered

shapes[29].mesh_with_interface.show()

# Above shape is: Z_0029
# b_flat-cs_round-50%-100%-150%

shapes[30].mesh_with_interface.show()

# Above shape is: Z_0030
# b_flat-cs_round-150%-100%-50%

shapes[31].mesh_with_interface.show()

# Above shape is: Z_0031
# b_flat-cs_round-50%-100%-200%

shapes[32].mesh_with_interface.show()

# Above shape is: Z_0032
# b_flat-cs_round-200%-100%-50%

shapes[33].mesh_with_interface.show()

# Above shape is: Z_0033
# b_flat-cs_round-50%-150%-50%

shapes[34].mesh_with_interface.show()

# Above shape is: Z_0034
# b_flat-cs_round-150%-50%-150%

