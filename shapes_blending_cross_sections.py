
import trimesh
import pickle
from pathlib import Path

pickle_path = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set\Z\stimulus_set_ 24_03_22__12_49_50.pickle")
with open(pickle_path, "rb") as f:
    (shapes, transitions_dict) = pickle.load(f)

# Shapes Rendered

shapes[15].mesh_with_interface.show()

# Above shape is: Z_0015
# b_flat-cs_round-cs_concave

shapes[16].mesh_with_interface.show()

# Above shape is: Z_0016
# b_flat-cs_concave-cs_round

shapes[17].mesh_with_interface.show()

# Above shape is: Z_0017
# b_flat-cs_round-cs_plane

shapes[18].mesh_with_interface.show()

# Above shape is: Z_0018
# b_flat-cs_plane-cs_round

shapes[19].mesh_with_interface.show()

# Above shape is: Z_0019
# b_flat-cs_round-cs_convex

shapes[20].mesh_with_interface.show()

# Above shape is: Z_0020
# b_flat-cs_convex-cs_round

shapes[21].mesh_with_interface.show()

# Above shape is: Z_0021
# b_flat-cs_round-cs_elliptical

shapes[22].mesh_with_interface.show()

# Above shape is: Z_0022
# b_flat-cs_elliptical-cs_round

shapes[23].mesh_with_interface.show()

# Above shape is: Z_0023
# b_flat-cs_elliptical-cs_concave

shapes[24].mesh_with_interface.show()

# Above shape is: Z_0024
# b_flat-cs_concave-cs_elliptical

shapes[25].mesh_with_interface.show()

# Above shape is: Z_0025
# b_flat-cs_elliptical-cs_plane

shapes[26].mesh_with_interface.show()

# Above shape is: Z_0026
# b_flat-cs_plane-cs_elliptical

shapes[27].mesh_with_interface.show()

# Above shape is: Z_0027
# b_flat-cs_elliptical-cs_convex

shapes[28].mesh_with_interface.show()

# Above shape is: Z_0028
# b_flat-cs_convex-cs_elliptical

