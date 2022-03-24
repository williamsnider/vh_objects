
import trimesh
import pickle
from pathlib import Path

pickle_path = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set\Z\stimulus_set_ 24_03_22__12_49_50.pickle")
with open(pickle_path, "rb") as f:
    (shapes, transitions_dict) = pickle.load(f)

# Shapes Rendered

shapes[6].mesh_with_interface.show()

# Above shape is: Z_0006
# b_flat-cs_concave

shapes[7].mesh_with_interface.show()

# Above shape is: Z_0007
# b_flat-cs_plane

shapes[8].mesh_with_interface.show()

# Above shape is: Z_0008
# b_flat-cs_convex

shapes[9].mesh_with_interface.show()

# Above shape is: Z_0009
# b_flat-cs_elliptical

shapes[10].mesh_with_interface.show()

# Above shape is: Z_0010
# b_weak_curve-cs_round

shapes[11].mesh_with_interface.show()

# Above shape is: Z_0011
# b_weak_curve-cs_concave

shapes[12].mesh_with_interface.show()

# Above shape is: Z_0012
# b_weak_curve-cs_plane

shapes[13].mesh_with_interface.show()

# Above shape is: Z_0013
# b_weak_curve-cs_convex

shapes[14].mesh_with_interface.show()

# Above shape is: Z_0014
# b_weak_curve-cs_elliptical

