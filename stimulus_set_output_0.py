
import trimesh
import pickle
from pathlib import Path

pickle_path = Path(r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set\Z\stimulus_set_ 24_03_22__12_08_50.pickle")
with open(pickle_path, "rb") as f:
    shapes = pickle.load(f)

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

shapes[35].mesh_with_interface.show()

# Above shape is: Z_0035
# b_flat-cs_elliptical-sd_sphere-rot_b_0-rot_ls_0-pos_0.33_convex

shapes[36].mesh_with_interface.show()

# Above shape is: Z_0036
# b_flat-cs_elliptical-sd_sphere-rot_b_0-rot_ls_0-pos_0.66_convex

shapes[37].mesh_with_interface.show()

# Above shape is: Z_0037
# b_flat-cs_elliptical-sd_sphere-rot_b_0-rot_ls_0-pos_0.33_0.66_convex

shapes[38].mesh_with_interface.show()

# Above shape is: Z_0038
# b_flat-cs_elliptical-sd_sphere-rot_b_0-rot_ls_0-pos_0.33_concave

shapes[39].mesh_with_interface.show()

# Above shape is: Z_0039
# b_flat-cs_elliptical-sd_sphere-rot_b_0-rot_ls_0-pos_0.66_concave

shapes[40].mesh_with_interface.show()

# Above shape is: Z_0040
# b_flat-cs_elliptical-sd_sphere-rot_b_0-rot_ls_0-pos_0.33_0.66_concave

shapes[41].mesh_with_interface.show()

# Above shape is: Z_0041
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_0-pos_0.33_convex

shapes[42].mesh_with_interface.show()

# Above shape is: Z_0042
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_0-pos_0.66_convex

shapes[43].mesh_with_interface.show()

# Above shape is: Z_0043
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_0-pos_0.33_0.66_convex

shapes[44].mesh_with_interface.show()

# Above shape is: Z_0044
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_0-pos_0.33_concave

shapes[45].mesh_with_interface.show()

# Above shape is: Z_0045
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_0-pos_0.66_concave

shapes[46].mesh_with_interface.show()

# Above shape is: Z_0046
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_0-pos_0.33_0.66_concave

shapes[47].mesh_with_interface.show()

# Above shape is: Z_0047
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_90-pos_0.33_convex

shapes[48].mesh_with_interface.show()

# Above shape is: Z_0048
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_90-pos_0.66_convex

shapes[49].mesh_with_interface.show()

# Above shape is: Z_0049
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_90-pos_0.33_0.66_convex

shapes[50].mesh_with_interface.show()

# Above shape is: Z_0050
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_90-pos_0.33_concave

shapes[51].mesh_with_interface.show()

# Above shape is: Z_0051
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_90-pos_0.66_concave

shapes[52].mesh_with_interface.show()

# Above shape is: Z_0052
# b_flat-cs_elliptical-sd_ellipsoid-rot_b_0-rot_ls_90-pos_0.33_0.66_concave

shapes[53].mesh_with_interface.show()

# Above shape is: Z_0053
# b_flat-cs_elliptical-sd_cylinder-rot_b_0-rot_ls_0-pos_0.33_convex

shapes[54].mesh_with_interface.show()

# Above shape is: Z_0054
# b_flat-cs_elliptical-sd_cylinder-rot_b_0-rot_ls_0-pos_0.66_convex

shapes[55].mesh_with_interface.show()

# Above shape is: Z_0055
# b_flat-cs_elliptical-sd_cylinder-rot_b_0-rot_ls_0-pos_0.33_0.66_convex

shapes[56].mesh_with_interface.show()

# Above shape is: Z_0056
# b_flat-cs_elliptical-sd_cylinder_spherical_top-rot_b_0-rot_ls_0-pos_0.33_convex

shapes[57].mesh_with_interface.show()

# Above shape is: Z_0057
# b_flat-cs_elliptical-sd_cylinder_spherical_top-rot_b_0-rot_ls_0-pos_0.66_convex

shapes[58].mesh_with_interface.show()

# Above shape is: Z_0058
# b_flat-cs_elliptical-sd_cylinder_spherical_top-rot_b_0-rot_ls_0-pos_0.33_0.66_convex

shapes[59].mesh_with_interface.show()

# Above shape is: Z_0059
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_0-pos_0.33_convex

shapes[60].mesh_with_interface.show()

# Above shape is: Z_0060
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_0-pos_0.66_convex

shapes[61].mesh_with_interface.show()

# Above shape is: Z_0061
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_0-pos_0.33_0.66_convex

shapes[62].mesh_with_interface.show()

# Above shape is: Z_0062
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_90-pos_0.33_convex

shapes[63].mesh_with_interface.show()

# Above shape is: Z_0063
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_90-pos_0.66_convex

shapes[64].mesh_with_interface.show()

# Above shape is: Z_0064
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_90-pos_0.33_0.66_convex

shapes[65].mesh_with_interface.show()

# Above shape is: Z_0065
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_180-pos_0.33_convex

shapes[66].mesh_with_interface.show()

# Above shape is: Z_0066
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_180-pos_0.66_convex

shapes[67].mesh_with_interface.show()

# Above shape is: Z_0067
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_180-pos_0.33_0.66_convex

shapes[68].mesh_with_interface.show()

# Above shape is: Z_0068
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_270-pos_0.33_convex

shapes[69].mesh_with_interface.show()

# Above shape is: Z_0069
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_270-pos_0.66_convex

shapes[70].mesh_with_interface.show()

# Above shape is: Z_0070
# b_flat-cs_elliptical-sd_curved_cylinder-rot_b_0-rot_ls_270-pos_0.33_0.66_convex

shapes[71].mesh_with_interface.show()

# Above shape is: Z_0071
# b_flat-cs_elliptical-sd_cone-rot_b_0-rot_ls_0-pos_0.33_convex

shapes[72].mesh_with_interface.show()

# Above shape is: Z_0072
# b_flat-cs_elliptical-sd_cone-rot_b_0-rot_ls_0-pos_0.66_convex

shapes[73].mesh_with_interface.show()

# Above shape is: Z_0073
# b_flat-cs_elliptical-sd_cone-rot_b_0-rot_ls_0-pos_0.33_0.66_convex

