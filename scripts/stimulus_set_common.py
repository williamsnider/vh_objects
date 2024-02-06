# Script to generate stimulus set C (contains multi-joint stimuli, sheets)


# Linear segment
import copy
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from objects.backbone import Backbone
from objects.shape import Shape
from objects.utilities import (
    approximate_arc,
    make_mesh,
    make_surface,
    calc_hemisphere_controlpoints,
    angle_between,
    calc_mesh_boolean_and_edges,
    angle_between,
)
from scripts.sheets import construct_sheet, bend_sheet, make_base_cp, plot_arr
import trimesh
from scipy.spatial.transform.rotation import Rotation
from objects.shaft import Shaft
from scripts.stimulus_set_params import (
    NUM_CP_PER_BASE_SHEET,
    NUM_CP_PER_CROSS_SECTION,
    NUM_CS,
    NUM_CS_PER_SHEET,
    SEGMENT_LENGTH,
    X_WIDTH,
    VOLUMETRIC_RADII,
    SHEET_THICKNESS,
    POINT_RADII,
    POINT_ROUNDOVER_OFFSET,
    LEAF_RADII,
    APPENDAGE_LENGTH,
    SAVE_DIR,
    XYZ_OFFSET,
    ROUND_RADIUS,
    BOX_EXTENTS,
    BOX_TRANSLATION,
    TERMINATION_RADIUS,
    uu,
    vv,
)

######################################
### Base Components and Appendages ###
######################################


def slice_mesh(mesh, extent, T):
    mesh = mesh.copy()
    slicer = trimesh.primitives.Box(
        extents=np.array([extent, extent, extent]), transform=T
    )
    split_mesh, _ = calc_mesh_boolean_and_edges(mesh, slicer, "difference")

    return split_mesh


thin = Shaft(
    SEGMENT_LENGTH,
    VOLUMETRIC_RADII[0],
    VOLUMETRIC_RADII[0],
    VOLUMETRIC_RADII[0],
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)

volumetric = Shaft(
    SEGMENT_LENGTH,
    VOLUMETRIC_RADII[0],
    VOLUMETRIC_RADII[1],
    VOLUMETRIC_RADII[2],
    theta=0,
    lengthtype="two_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)

# Transformation matrix so that shafts are pointing towards +Z axis
T_point_z = np.eye(4)
T_point_z[:3, :3] = Rotation.from_euler("xyz", np.array([0, -np.pi / 2, 0])).as_matrix()



def plot_cp_and_backbone(cp, backbone):
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection="3d")
    arr = cp
    for i in range(arr.shape[0]):
        ax.plot(arr[i, :, 0], arr[i, :, 1], arr[i, :, 2], "b-*")

    # Plot backbone
    t = np.linspace(0, 1, 100)
    bx = backbone.r(t)[:, 0]
    by = backbone.r(t)[:, 1]
    bz = backbone.r(t)[:, 2]
    ax.plot(bx, by, bz, "r-")

    # Set scale
    xs = np.concatenate([arr[:, :, 0].ravel(), bx.ravel()])
    ys = np.concatenate([arr[:, :, 1].ravel(), by.ravel()])
    zs = np.concatenate([arr[:, :, 2].ravel(), bz.ravel()])
    ax.set_box_aspect(
        (np.ptp(xs), np.ptp(ys), np.ptp(zs))
    )  # aspect ratio is 1:1:1 in data space
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

APP_ANGLE = np.pi/2

def bend_app(app, app_length):
    app_K1_b_length = app_length - app.rdist
    app_K1_b_cp  = approximate_arc(APP_ANGLE, app_K1_b_length, 5)
    app_K1_b_cp = app_K1_b_cp[:, [1, 2, 0]]  # Reorder
    app_K1_b_cp[:, 0] *= -1  # Flip direction across yz axis
    app_K1_b = Backbone(app_K1_b_cp, reparameterize=True)

    cp = app.cp.copy()
    cp[:,:,2] *= -1
    bent_cp = bend_sheet(cp, app_K1_b, app_K1_b_length)
    surf = make_surface(bent_cp)
    app_K1 = make_mesh(surf, uu, vv)
    app_K1.faces = app_K1.faces[:, [0, 2, 1]]  # Flip faces to fix winding
    return app_K1


A_APPENDAGE_LENGTH = APPENDAGE_LENGTH * 1.75
A_X_WIDTH = X_WIDTH 
app0 = Shaft(A_APPENDAGE_LENGTH, 1.0 * A_X_WIDTH, 1.0 * A_X_WIDTH, 1.0*A_X_WIDTH, theta=0, lengthtype="one_hemi", num_cs=NUM_CS, num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,)
app0.apply_transform(T_point_z)
app0_K1 = bend_app(app0,A_APPENDAGE_LENGTH)

app1 = Shaft(
    A_APPENDAGE_LENGTH,
    1.0 * A_X_WIDTH,
    1.0 * A_X_WIDTH,
    1.7 * A_X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app1.apply_transform(T_point_z)

# Bend
app1_K1_b_length = A_APPENDAGE_LENGTH - app1.rdist
app1_K1_b_cp  = approximate_arc(APP_ANGLE, app1_K1_b_length, 5)
app1_K1_b_cp = app1_K1_b_cp[:, [1, 2, 0]]  # Reorder
app1_K1_b_cp[:, 0] *= -1  # Flip direction across yz axis
app1_K1_b = Backbone(app1_K1_b_cp, reparameterize=True)

cp = app1.cp
cp[:,:,2] *= -1
bent_cp = bend_sheet(cp, app1_K1_b, app1_K1_b_length)
surf = make_surface(bent_cp)
app1_K1 = make_mesh(surf, uu, vv)
app1_K1.faces = app1_K1.faces[:, [0, 2, 1]]  # Flip faces to fix winding


app2 = Shaft(
    A_APPENDAGE_LENGTH,
    1.0 * A_X_WIDTH,
    1.5 * A_X_WIDTH,
    TERMINATION_RADIUS,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app2.apply_transform(T_point_z)

# Bend
app2_K1_b_length = A_APPENDAGE_LENGTH - app2.rdist
app2_K1_b_cp  = approximate_arc(APP_ANGLE, app2_K1_b_length, 5)
app2_K1_b_cp = app2_K1_b_cp[:, [1, 2, 0]]  # Reorder
app2_K1_b_cp[:, 0] *= -1  # Flip direction across yz axis
app2_K1_b = Backbone(app2_K1_b_cp, reparameterize=True)

cp = app2.cp
cp[:,:,2] *= -1
bent_cp = bend_sheet(cp, app2_K1_b, app2_K1_b_length)
surf = make_surface(bent_cp)
app2_K1 = make_mesh(surf, uu, vv)
app2_K1.faces = app2_K1.faces[:, [0, 2, 1]]  # Flip faces to fix winding


app3 = Shaft(
    A_APPENDAGE_LENGTH,
    1.0 * A_X_WIDTH,
    1.7 * A_X_WIDTH,
    1.0 * A_X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app3.apply_transform(T_point_z)

# Bend
app3_K1_b_length = A_APPENDAGE_LENGTH - app3.rdist
app3_K1_b_cp  = approximate_arc(APP_ANGLE, app3_K1_b_length, 5)
app3_K1_b_cp = app3_K1_b_cp[:, [1, 2, 0]]  # Reorder
app3_K1_b_cp[:, 0] *= -1  # Flip direction across yz axis
app3_K1_b = Backbone(app3_K1_b_cp, reparameterize=True)

cp = app3.cp
cp[:,:,2] *= -1
bent_cp = bend_sheet(cp, app3_K1_b, app3_K1_b_length)
surf = make_surface(bent_cp)
app3_K1 = make_mesh(surf, uu, vv)
app3_K1.faces = app3_K1.faces[:, [0, 2, 1]]  # Flip faces to fix winding

app4 = Shaft(
    A_APPENDAGE_LENGTH,
    1.0 * A_X_WIDTH,
    1.0 * A_X_WIDTH,
    TERMINATION_RADIUS,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app4.apply_transform(T_point_z)

# Bend
app4_K1_b_length = A_APPENDAGE_LENGTH - app4.rdist
app4_K1_b_cp  = approximate_arc(APP_ANGLE, app4_K1_b_length, 5)
app4_K1_b_cp = app4_K1_b_cp[:, [1, 2, 0]]  # Reorder
app4_K1_b_cp[:, 0] *= -1  # Flip direction across yz axis
app4_K1_b = Backbone(app4_K1_b_cp, reparameterize=True)

cp = app4.cp
cp[:,:,2] *= -1
bent_cp = bend_sheet(cp, app4_K1_b, app4_K1_b_length)
surf = make_surface(bent_cp)
app4_K1 = make_mesh(surf, uu, vv)
app4_K1.faces = app4_K1.faces[:, [0, 2, 1]]  # Flip faces to fix winding

app5 = Shaft(
    A_APPENDAGE_LENGTH,
    1.0 * A_X_WIDTH,
    1.7 * A_X_WIDTH,
    1.7 * A_X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app5.apply_transform(T_point_z)
app5_K1 = bend_app(app5, A_APPENDAGE_LENGTH)

app6 = Shaft(APPENDAGE_LENGTH, 1.0 * X_WIDTH, 1.0 * X_WIDTH, 1.0*X_WIDTH, theta=0, lengthtype="one_hemi", num_cs=NUM_CS, num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,)
app6.apply_transform(T_point_z)

app7 = Shaft(
    APPENDAGE_LENGTH,
    1.0 * X_WIDTH,
    1.0 * X_WIDTH,
    1.35 * X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app7.apply_transform(T_point_z)

app8 = Shaft(
    APPENDAGE_LENGTH,
    1.0 * X_WIDTH,
    1.5 * X_WIDTH,
    TERMINATION_RADIUS,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app8.apply_transform(T_point_z)



app9 = Shaft(
    APPENDAGE_LENGTH,
    1.0 * X_WIDTH,
    1.5 * X_WIDTH,
    1.0 * X_WIDTH,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app9.apply_transform(T_point_z)

app10 = Shaft(
    APPENDAGE_LENGTH,
    1.0 * X_WIDTH,
    1.0 * X_WIDTH,
    TERMINATION_RADIUS,
    theta=0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app10.apply_transform(T_point_z)

# app11 = Shaft(
#     APPENDAGE_LENGTH,
#     1.0 * X_WIDTH,
#     1.5 * X_WIDTH,
#     1.5 * X_WIDTH,
#     theta=0,
#     lengthtype="one_hemi",
#     num_cs=NUM_CS,
#     num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
# )
# app11.apply_transform(T_point_z)


# Post
from scripts.stimulus_set_params import POST_RADIUS
post_th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1, 1)
post_cp = np.hstack((POST_RADIUS * np.sin(post_th), POST_RADIUS * np.cos(post_th), np.ones(post_th.shape)))
post_stack = np.zeros((NUM_CS, NUM_CP_PER_CROSS_SECTION, 3))
for i in range(1,NUM_CS):
    sub = post_cp.copy()
    sub[:,2] = np.linspace(-10,0,NUM_CS, endpoint=False)[i]
    post_stack[i] = sub
surf = make_surface(post_stack)
mesh = make_mesh(surf, uu, vv)


# Add post to app1
cp_app1_post = app1.cp
cp_app1_post[:NUM_CS] = post_stack
new_surf = make_surface(cp_app1_post)
app1_post = make_mesh(new_surf, uu, vv)
app1_post.faces = app1_post.faces[:, [0, 2, 1]]  # Flip faces to fix winding


b_cp = approximate_arc(np.pi / 2, X_WIDTH, 5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
backbone_x_width = Backbone(b_cp, reparameterize=True)

b_cp = approximate_arc(np.pi/2, APPENDAGE_LENGTH,5)
b_cp = b_cp[:, [1, 2, 0]]  # Reorder
b_cp[:, 0] *= -1  # Flip direction across yz axis
b_appendage_K1 = Backbone(b_cp, reparameterize=True)

app_point = Shaft(
    ROUND_RADIUS,
    1.0 * X_WIDTH,
    0.45 * X_WIDTH,
    TERMINATION_RADIUS,
    0,
    lengthtype="one_hemi",
    num_cs=NUM_CS,
    num_cp_per_cs=NUM_CP_PER_CROSS_SECTION,
)
app_point.apply_transform(T_point_z)
app_point_convex = app_point.mesh

# Flip across yz plane for concave
T = np.eye(4)
T[:3, :3] = Rotation.from_euler("xyz", np.array([np.pi, 0, 0])).as_matrix()
app_point_concave = app_point_convex.copy()
app_point_concave = app_point_concave.apply_transform(T)


app_round_concave = trimesh.primitives.creation.icosphere(3, radius=ROUND_RADIUS)
app_round_convex = app_round_concave.copy()


# For sheets, create a controlpoint grid, surface, and mesh from scratch (i.e. without the Shaft class).

# Rotation for K2
T_K2 = np.eye(4)
T_K2[:3, :3] = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()



# Round sheet
t = np.linspace(0, 2 * np.pi, NUM_CP_PER_BASE_SHEET, endpoint=False).reshape(-1, 1)
round_cs_cp = np.hstack([np.zeros(t.shape), np.cos(t), np.sin(t)])
base_sheet = round_cs_cp * ROUND_RADIUS
cp = construct_sheet(
    base_sheet, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET
)
surf = make_surface(cp)
sheet_round_K0 = make_mesh(surf, uu, vv)

# Bend round sheet
bent_cp = bend_sheet(cp, backbone_x_width, X_WIDTH)
surf = make_surface(bent_cp)
sheet_round_K1 = make_mesh(surf, uu, vv)
sheet_round_K2 = sheet_round_K1.copy()
sheet_round_K2.apply_transform(T_K2)


# Leaf sheet
num_edge_cp = 7
base_round_cp = 3
top_round_cp = 1
leaf_x = np.linspace(0, 1, 3) * APPENDAGE_LENGTH
leaf_y = LEAF_RADII
leaf_poly = np.polyfit(leaf_x, leaf_y, 2)
leaf_cp = make_base_cp(leaf_poly, leaf_x, num_edge_cp, base_round_cp, top_round_cp)
mean_xyz = leaf_cp.mean(axis=0)
leaf_cp = leaf_cp - mean_xyz  # Shift to origin for scaling
cp = construct_sheet(leaf_cp, sheet_thickness=SHEET_THICKNESS, num_cs=NUM_CS_PER_SHEET)
cp += mean_xyz.reshape(1, 1, 3)  # Shift back to original position
surf = make_surface(cp)
sheet_leaf_K0 = make_mesh(surf, uu, vv)

# Bent leaf sheet
bent_cp = bend_sheet(cp, b_appendage_K1, leaf_x[2] - leaf_x[0])
surf = make_surface(bent_cp)
sheet_leaf_K1 = make_mesh(surf, uu, vv)
sheet_leaf_K2 = sheet_leaf_K1.copy()
sheet_leaf_K2.apply_transform(T_K2)


# Point sheet
NUM_POINT_CS = NUM_CS * 2

# Calculate widths
point_x = np.linspace(0, APPENDAGE_LENGTH, 3)
point_y = POINT_RADII  # 3 radii determine polynomial form
point_poly = np.polyfit(point_x, point_y, 2)

# But sample along 4th position to ensure smooth transition into shape
xvals = np.linspace(-APPENDAGE_LENGTH * 2 / 3, APPENDAGE_LENGTH, NUM_POINT_CS)
widths = np.polyval(point_poly, xvals)

# Calculate z_levels
z_levels = np.linspace(-APPENDAGE_LENGTH * 2 / 3, APPENDAGE_LENGTH, NUM_POINT_CS)

# Assign controlpoints
cp = np.zeros((NUM_POINT_CS, 8, 3))
for i, width in enumerate(widths):

    if width == 0:
        inner = np.zeros((8, 2))
    else:
        inner = np.array(
            [
                [SHEET_THICKNESS / 2, 0],
                [SHEET_THICKNESS / 2, width - POINT_ROUNDOVER_OFFSET],
                [0, width],
                [-SHEET_THICKNESS / 2, width - POINT_ROUNDOVER_OFFSET],
                [-SHEET_THICKNESS / 2, 0],
                [-SHEET_THICKNESS / 2, -width + POINT_ROUNDOVER_OFFSET],
                [0, -width],
                [SHEET_THICKNESS / 2, -width + POINT_ROUNDOVER_OFFSET],
            ]
        )

    xyz = np.hstack([inner, z_levels[i] * np.ones((inner.shape[0], 1))])

    cp[i, :, :] = xyz

# Roundover edges

# side_y = POINT_RADII + POINT_ROUNDOVER_OFFSET
side_poly = point_poly  # np.polyfit(x, side_y, 2)
bot, _ = calc_hemisphere_controlpoints(
    cp[0],
    np.array([0, 0, 1]),
    cp[0].mean(axis=0),
    side_poly,
    -APPENDAGE_LENGTH * 2 / 3,
    morph_to_ellipse=True,
)
top, _ = calc_hemisphere_controlpoints(
    cp[-1],
    np.array([0, 0, 1]),
    cp[-1].mean(axis=0),
    side_poly,
    point_x[-1],
    morph_to_ellipse=True,
)
sheet_point_cp = np.vstack([bot, cp, top[-2::-1]])
surf = make_surface(sheet_point_cp)
sheet_point_K0 = make_mesh(surf, uu, vv)

bent_cp = bend_sheet(sheet_point_cp, b_appendage_K1, point_x[2] - point_x[0])
surf = make_surface(bent_cp)
sheet_point_K1 = make_mesh(
    surf, uu, vv
)  # TODO: Has artifact, fix after deciding on thickness/size
sheet_point_K2 = sheet_point_K1.copy()
sheet_point_K2.apply_transform(T_K2)

# Slice certain meshes to keep it from going through shape
extent = 100
T = np.eye(4)
T[2, 3] = -extent / 2 - 1 * X_WIDTH
app_point_convex = slice_mesh(
    app_point_convex,
    100,
    T,
)

sheet_point_K0 = slice_mesh(sheet_point_K0, extent, T)
sheet_point_K1 = slice_mesh(sheet_point_K1, extent, T)
sheet_point_K2 = slice_mesh(sheet_point_K2, extent, T)
sheet_leaf_K0 = slice_mesh(sheet_leaf_K0, extent, T)
sheet_leaf_K1 = slice_mesh(sheet_leaf_K1, extent, T)
sheet_leaf_K2 = slice_mesh(sheet_leaf_K2, extent, T)
sheet_round_K0 = slice_mesh(sheet_round_K0, extent, T)
sheet_round_K1 = slice_mesh(sheet_round_K1, extent, T)
sheet_round_K2 = slice_mesh(sheet_round_K2, extent, T)

mesh_dict = {
    "thin": thin.mesh,
    "volumetric": volumetric.mesh,
    "app0": app0.mesh, # 1 1 1
    "app0_K1": app0_K1,
    "app1": app1.mesh, # 1 1 2
    # "app1_post": app1_post,
    "app1_K1": app1_K1,
    "app2": app2.mesh, # 1 2 0
    "app2_K1": app2_K1,
    "app3": app3.mesh, # 1 2 1
    "app3_K1": app3_K1,
    "app4": app4.mesh, # 1 1 0
    "app4_K1": app4_K1,
    "app5": app5.mesh, # 1 2 2
    "app5_K1": app5_K1,
    "app6": app6.mesh, # 1 1 1
    "app7": app7.mesh, # 1 1 2
    "app8": app8.mesh, # 1 2 0
    "app9": app9.mesh, # 1 2 1 
    "app10": app10.mesh, # 1 1 0
    # "app11": app11.mesh, # 1 2 2
    "app_point_concave": app_point_concave,
    "app_point_convex": app_point_convex,
    "sheet_round_K0": sheet_round_K0,
    "sheet_round_K1": sheet_round_K1,
    "sheet_round_K2": sheet_round_K2,
    "app_round_concave": app_round_concave,
    "app_round_convex": app_round_convex,
    "sheet_leaf_K0": sheet_leaf_K0,
    "sheet_leaf_K1": sheet_leaf_K1,
    "sheet_leaf_K2": sheet_leaf_K2,
    "sheet_point_K0": sheet_point_K0,
    "sheet_point_K1": sheet_point_K1,
    "sheet_point_K2": sheet_point_K2,
}

if __name__ == "__main__":
    scene = trimesh.Scene()
    shift = np.array([0.0, 0.0, 0.0])
    for k, m in mesh_dict.items():

        mesh = copy.deepcopy(m)

        # Shift so bounds in lower left corner
        mesh = mesh.apply_translation(np.array([0, mesh.extents[1] / 2, 0]))
        mesh = mesh.apply_translation(shift)

        scene.add_geometry(mesh)

        extents = np.copy(mesh.extents)
        extents[0] = 0
        extents[2] = 0
        shift += extents * 1.1
    scene.show()