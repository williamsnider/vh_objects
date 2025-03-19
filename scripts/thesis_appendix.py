# Make appendix figures for thesis
from pathlib import Path
import pyrender
import trimesh
import numpy as np
import scipy
from trimesh.transformations import rotation_matrix as rotvec2T
import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_mesh_as_png(fname_stl, suffix="", z_rotation=None, resolution=1200):
    """
    Saves the mesh as a png.
    """

    fname_png = Path(fname_stl.parents[2], "png", fname_stl.parts[-2], fname_stl.name.replace(".stl", suffix + ".png"))

    if fname_png.parents[0].exists() is False:
        fname_png.parents[0].mkdir(parents=True)

    # Compose scene
    scene = pyrender.Scene(
        ambient_light=[0.2, 0.2, 0.2], bg_color=[255.0, 255.0, 255.0, 255.0]
    )  # White background - docs say float? but that gives black.

    # # Add mesh to scene
    # if rotation is not None:
    #     mesh_pose = rotation
    # else:
    #     mesh_pose = np.eye(4)
    # if interface == True:
    #     mesh = pyrender.Mesh.from_trimesh(self.mesh_with_interface, smooth=False)
    # else:
    #     mesh = pyrender.Mesh.from_trimesh(self.mesh, smooth=False)

    # Load mesh
    mesh = trimesh.load(fname_stl)
    mesh.visual.face_colors = [78, 165, 96, 255]
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    # Rotate mesh to get better view
    if z_rotation is None:
        mesh_pose = np.eye(4)
    else:
        mesh_pose = rotvec2T(z_rotation, [0, 0, 1])
    mesh_pose = (
        rotvec2T(-np.pi / 4, [0, 1, 0])
        @ rotvec2T(np.pi / 4, [1, 0, 0])
        @ rotvec2T(-6 * np.pi / 8, [0, 0, 1])
        @ mesh_pose
    )
    # mesh_pose = rotvec2T(np.pi / 2, [1, 1, 1])

    # TZ90 = rotvec2T(-np.pi / 2, [0, 0, 1])
    # TX90 = rotvec2T(-np.pi / 2.1, [1, 0, 0])
    # mesh_pose = TX90
    # mesh_pose[2, 3] -= 10

    scene.add(mesh, pose=mesh_pose)

    # Camera pose explained:
    # +X axis is towards the right of the screen
    # +Y axis is towards the top of the screen
    # -Z axis points into the screen (camera looks into the screen)
    yfov = np.pi / 4.0
    ywidth = 100  # mm
    camera_pose = np.eye(4)
    camera_pose[2, 3] = ywidth / 2 / np.tan(yfov / 2)  # Calculate correct distance for camera to have ywidth and yfov
    camera = pyrender.PerspectiveCamera(yfov=yfov)
    scene.add(camera, pose=camera_pose)

    # # Define the camera position and orientation
    # camera_position = np.array([0, 100, 100])
    # target_position = np.array([0, 0, 0])

    # # Calculate the forward direction (negative z-axis in camera space)
    # forward = target_position - camera_position
    # forward = forward / np.linalg.norm(forward)

    # # Calculate the right and up vectors for the camera
    # right = np.cross(np.array([0, 0, 1]), forward)
    # right = right / np.linalg.norm(right)
    # up = np.cross(forward, right)

    # # Set the camera pose
    # camera_pose = np.eye(4)
    # camera_pose[:3, 0] = right
    # camera_pose[:3, 1] = up
    # camera_pose[:3, 2] = -forward  # Forward is the negative Z direction in camera space
    # camera_pose[:3, 3] = camera_position

    # # Create the camera
    # yfov = np.pi / 4.0
    # camera = pyrender.PerspectiveCamera(yfov=yfov)

    # # Add the camera to the scene with the specified pose
    # scene.add(camera, pose=camera_pose)

    # # Add directional light
    # light_euler = np.array([-np.pi / 4, 0, 0])
    # R = scipy.spatial.transform.Rotation.from_euler("xyz", light_euler).as_matrix()
    # light_pose = np.eye(4)
    # light_pose[:3, :3] = R
    # light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5e3)
    # scene.add(light, pose=light_pose)

    # # Add primary directional light
    # light_euler = np.array([-np.pi / 4, 0, np.pi / 6])  # Adjusted angle
    # R = scipy.spatial.transform.Rotation.from_euler("xyz", light_euler).as_matrix()
    # light_pose = np.eye(4)
    # light_pose[:3, :3] = R
    # light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3000)
    # scene.add(light, pose=light_pose)

    # Add a second fill light (opposite direction)
    # fill_light_euler = np.array([np.pi / 6, np.pi, 0])
    fill_light_euler = np.array([-np.pi / 4, 0, np.pi / 6])  # Adjusted angle
    R2 = scipy.spatial.transform.Rotation.from_euler("xyz", fill_light_euler).as_matrix()
    fill_light_pose = np.eye(4)
    fill_light_pose[:3, :3] = R2
    fill_light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1500)
    scene.add(fill_light, pose=fill_light_pose)

    # Add a soft spot light for highlights
    spot_light = pyrender.SpotLight(
        color=[1, 1, 1], intensity=20000, innerConeAngle=np.pi / 8, outerConeAngle=np.pi / 3
    )
    spot_light_pose = np.eye(4)
    spot_light_pose[:3, 3] = [100, 100, 100]  # Position above the scene
    scene.add(spot_light, pose=spot_light_pose)

    r = pyrender.OffscreenRenderer(resolution, resolution)
    color, _ = r.render(scene)

    # # Display image with imshow
    # import matplotlib.pyplot as plt

    # plt.imshow(color)
    # plt.show()

    cv2.imwrite(str(fname_png), color)

    return fname_png


# Inputs
DPI = 600
PAGE_DIMS_IN = np.array([6, 8])  # inches
PAGE_DIMS_PX = (PAGE_DIMS_IN * DPI).astype(int)
NCOLS = 3
NROWS = 5
OVERLAY_FRAC = 0.4

# Check values OK
combined_img_width = int(PAGE_DIMS_PX[0] / NCOLS)
combined_img_height = int(combined_img_width / (2 - OVERLAY_FRAC))
assert combined_img_width * NCOLS <= PAGE_DIMS_PX[0], "Combined image height exceeds page width"
assert combined_img_height * NROWS <= PAGE_DIMS_PX[1], "Combined image width exceeds page height"

# Calculate y downshift (take up full space)
y_margin = np.floor((PAGE_DIMS_PX[1] - combined_img_height * NROWS) / NROWS).astype("int")

#############################################
### Render images at specified resolution ###
#############################################

# Load stl files
stl_dir = Path("/home/oconnorlab/Desktop/all_parts")
stl_list = list(stl_dir.glob("*.stl"))
stl_list = [s for s in stl_list if "Q" not in s.name]
print("Number of stl files:", len(stl_list))

# # Render in orientation A and B
# for stl in stl_list:
#     png_A = save_mesh_as_png(stl, suffix="_A", z_rotation=0, resolution=combined_img_height)
#     print(png_A)
#     png_B = save_mesh_as_png(stl, suffix="_B", z_rotation=np.pi, resolution=combined_img_height)
#     print(png_B)

#     print("Breaking early.")
#     png_dir = png_A.parents[0]


def render_stl(stl):
    png_A = save_mesh_as_png(stl, suffix="_A", z_rotation=0, resolution=combined_img_height)
    png_B = save_mesh_as_png(stl, suffix="_B", z_rotation=np.pi / 2, resolution=combined_img_height)
    return png_A, png_B


from multiprocessing import Pool
from tqdm import tqdm

# # Use a multiprocessing pool
# with Pool() as pool:
#     # Use tqdm to display the progress bar
#     for _ in tqdm(pool.imap_unordered(render_stl, stl_list), total=len(stl_list), desc="Processing STL files"):
#         pass

#################################
### Combine images into pages ###
#################################
png_dir = Path("/home/oconnorlab/png/all_parts")

png_A_list = list(png_dir.glob("*_A.png"))
png_B_list = list(png_dir.glob("*_B.png"))

# Sort by name
png_A_list = sorted(png_A_list)
png_B_list = sorted(png_B_list)

# Place G images first
G_idx = [i for i, p in enumerate(png_A_list) if "G" in p.stem]
png_A_list = [png_A_list[i] for i in G_idx] + [png_A_list[i] for i in range(len(png_A_list)) if i not in G_idx]
png_B_list = [png_B_list[i] for i in G_idx] + [png_B_list[i] for i in range(len(png_B_list)) if i not in G_idx]

print(len(png_A_list))


page_num = 0
i = 0

while i < len(png_A_list):
    # Create a new page
    img_page = np.zeros((PAGE_DIMS_PX[1], PAGE_DIMS_PX[0], 3), dtype=np.uint8)
    page_name = f"page_{page_num}"
    text_x_y = []
    for r in range(NROWS):
        for c in range(NCOLS):
            if i >= len(png_A_list):
                break  # Stop if there are no more images

            # Get filenames
            A_fname = png_A_list[i]
            B_fname = png_B_list[i]
            assert A_fname.stem[:4] == B_fname.stem[:4], "Mismatched filenames"
            page_name += f"_{A_fname.stem[:4]}"

            # Load images
            img_A = cv2.imread(A_fname, cv2.IMREAD_UNCHANGED)
            img_B = cv2.imread(B_fname, cv2.IMREAD_UNCHANGED)

            # Assign A image
            x0 = int(c * combined_img_width)
            y0 = int(r * combined_img_height) + y_margin * (r + 1)
            img_page[y0 : y0 + combined_img_height, x0 : x0 + combined_img_height] = img_A

            # Assign B image (overlay, ignoring white pixels)
            xB0 = x0 + combined_img_width - combined_img_height
            WHITE_THRES = 240
            mask_B = np.all(img_B[:, :, :3] > WHITE_THRES, axis=2)
            for ch in range(3):
                img_page[y0 : y0 + combined_img_height, xB0 : xB0 + combined_img_height, ch] = np.where(
                    mask_B,
                    img_page[y0 : y0 + combined_img_height, xB0 : xB0 + combined_img_height, ch],
                    img_B[:, :, ch],
                )

            # Assign text label
            txt = A_fname.stem[:4]
            text_x = x0 + 730
            text_y = y0 - 75
            text_x_y.append([txt, text_x, text_y])

            # Iterate counter
            i += 1

    # Turn black pixels white
    mask_black = np.all(img_page == 0, axis=2)
    img_page[mask_black] = [255, 255, 255]

    # Add text labels
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 3
    # font_thickness = 10
    # font_color = (0.5, 0.5, 0.5)
    # for txt, text_x, text_y in text_x_y:
    #     text_size = cv2.getTextSize(txt, font, font_scale, font_thickness)[0]
    #     cv2.putText(img_page, txt, (text_x, text_y), font, font_scale, font_color, font_thickness)

    # Display and save the page
    fig, ax = plt.subplots(figsize=(PAGE_DIMS_PX[0] / DPI, PAGE_DIMS_PX[1] / DPI))

    # Add labels
    for txt, text_x, text_y in text_x_y:
        ax.text(
            text_x,
            text_y,
            txt,
            fontsize=12,
            color="black",
            verticalalignment="top",
            horizontalalignment="center",
        )

    ax.imshow(img_page)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    page_savename = Path(png_dir, page_name + ".pdf")
    # plt.show()
    fig.savefig(page_savename, dpi=DPI)
    print(f"Saved page {page_name}")
    plt.close(fig)

    page_num += 1
