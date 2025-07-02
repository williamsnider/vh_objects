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
from multiprocessing import Pool
from tqdm import tqdm


def save_mesh_as_png(fname_stl, suffix="", z_rotation=None, resolution=1200):
    """
    Saves the mesh as a png.
    """

    fname_png = Path(
        fname_stl.parents[2], "png" + suffix, fname_stl.parts[-2], fname_stl.name.replace(".stl", suffix + ".png")
    )

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


# # Inputs
# DPI = 600
# PAGE_DIMS_IN = np.array([6, 8])  # inches
# PAGE_DIMS_PX = (PAGE_DIMS_IN * DPI).astype(int)
# NCOLS = 3
# NROWS = 5
# OVERLAY_FRAC = 0.4

# # Check values OK
# combined_img_width = int(PAGE_DIMS_PX[0] / NCOLS)
# combined_img_height = int(combined_img_width / (2 - OVERLAY_FRAC))
# assert combined_img_width * NCOLS <= PAGE_DIMS_PX[0], "Combined image height exceeds page width"
# assert combined_img_height * NROWS <= PAGE_DIMS_PX[1], "Combined image width exceeds page height"

# Calculate y downshift (take up full space)
# y_margin = np.floor((PAGE_DIMS_PX[1] - combined_img_height * NROWS) / NROWS).astype("int")


# # Render in orientation A and B
# for stl in stl_list:
#     png_A = save_mesh_as_png(stl, suffix="_A", z_rotation=0, resolution=combined_img_height)
#     print(png_A)
#     png_B = save_mesh_as_png(stl, suffix="_B", z_rotation=np.pi, resolution=combined_img_height)
#     print(png_B)

#     print("Breaking early.")
#     png_dir = png_A.parents[0]


def render_stl(stl, resolution):
    png_A = save_mesh_as_png(stl, suffix="_A", z_rotation=0, resolution=resolution)
    png_B = save_mesh_as_png(stl, suffix="_B", z_rotation=np.pi / 2, resolution=resolution)
    return png_A, png_B


def combine_png_AB(A_fname, B_fname, IMG_HEIGHT):
    assert A_fname.stem[:4] == B_fname.stem[:4], "Mismatched filenames"

    overlay_frac = 0.4
    y_offset = 50
    combined_img_width = int(IMG_HEIGHT * (1 + overlay_frac))
    combined_img_height = IMG_HEIGHT + y_offset
    DPI = 1200

    # Load images
    img_A = cv2.imread(A_fname, cv2.IMREAD_UNCHANGED)
    img_B = cv2.imread(B_fname, cv2.IMREAD_UNCHANGED)

    # Assign A image
    x0 = 0
    y0 = y_offset

    img_combined = np.zeros((combined_img_height, combined_img_width, 3), dtype=np.uint8)
    img_combined[y0 : y0 + combined_img_height, x0 : x0 + IMG_HEIGHT] = img_A

    # cv2.imshow("Image A", img_combined)
    # cv2.waitKey(0)
    # Assign B image (overlay, ignoring white pixels)
    xB0 = x0 + combined_img_width
    WHITE_THRES = 240
    mask_B = np.all(img_B[:, :, :3] > WHITE_THRES, axis=2)
    for ch in range(3):
        img_combined[y0 : y0 + combined_img_height, -IMG_HEIGHT:, ch] = np.where(
            mask_B,
            img_combined[y0 : y0 + combined_img_height, -IMG_HEIGHT:, ch],
            img_B[:, :, ch],
        )

    # Turn black pixels white
    black_threshold = 2
    mask_black = np.all(img_combined < black_threshold, axis=2)
    img_combined[mask_black] = [255, 255, 255]

    # Assign text label
    txt = A_fname.stem[:4]
    text_x = x0 + 640
    text_y = y0 - 40

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 3
    # font_thickness = 10
    # font_color = (0.5, 0.5, 0.5)
    # text_size = cv2.getTextSize(txt, font, font_scale, font_thickness)[0]
    # cv2.putText(img_combined, txt, (text_x, text_y), font, font_scale, font_color, font_thickness)

    fig, ax = plt.subplots(figsize=(2 * combined_img_width / DPI, 2 * combined_img_height / DPI))

    # Add labels
    ax.text(
        text_x,
        text_y,
        txt,
        fontsize=12,
        color="black",
        verticalalignment="top",
        horizontalalignment="center",
    )

    ax.imshow(img_combined)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # page_savename = Path(png_dir, page_name + ".pdf")
    AB_fname = Path(str(A_fname).replace("_A", "_AB"))
    AB_fname.parent.mkdir(parents=True, exist_ok=True)
    # plt.show()
    fig.savefig(AB_fname, dpi=DPI)
    print(f"Saved {AB_fname}")
    plt.close(fig)


def make_rotated_pngs(stl_dir, resolution):
    """
    Make rotated PNGs of STL files in the specified directory.
    """
    stl_list = list(stl_dir.rglob("*.stl"))
    stl_list = [s for s in stl_list if "Q" not in s.name]
    print("Number of STL files:", len(stl_list))

    for stl in tqdm(stl_list, desc="Processing STL files"):
        png_A, png_B = render_stl(stl, resolution)
        # print(f"Rendered {png_A} and {png_B}")

    # Combine images A and B, overlaying correctly
    base_dir = stl_dir.parents[0]
    sub_dir = stl_dir.relative_to(base_dir)
    png_A_dir = Path(base_dir, str(sub_dir).replace("stl", "png_A"))
    png_A_list = list(png_A_dir.rglob("*.png"))
    png_A_list.sort()

    png_B_list = [Path(str(p).replace("_A", "_B")) for p in png_A_list]

    for i in range(len(png_A_list)):
        A_fname = png_A_list[i]
        B_fname = png_B_list[i]
        combine_png_AB(A_fname, B_fname, resolution)


if __name__ == "__main__":

    # Inputs
    stl_dir = Path("/home/oconnorlab/Downloads/stl_rotated_correctly/stl")
    resolution = 750  # Resolution for the rendered images

    # Make rotated PNGs
    make_rotated_pngs(stl_dir, resolution)


# if __name__ == "__main__":

#     # # Render in orientation A and B
#     # for stl in tqdm(stl_list, desc="Processing STL files"):
#     #     png_A, png_B = render_stl(stl)
#     #     print(f"Rendered {png_A} and {png_B}")

#     # Combine images A and B, overlaying correctly
#     png_A_dir = Path("/home/oconnorlab/Downloads/stl_rotated_correctly/png_A")
#     png_A_list = list(png_A_dir.rglob("*.png"))
#     png_A_list.sort()

#     png_B_list = [Path(str(p).replace("_A", "_B")) for p in png_A_list]

#     for i in range(len(png_A_list)):
#         A_fname = png_A_list[i]
#         B_fname = png_B_list[i]
#         combine_png_AB(A_fname, B_fname)


# if __name__ == "__main__":

#     #############################################
#     ### Render images at specified resolution ###
#     #############################################

#     # Load stl files
#     stl_dir = Path("/home/oconnorlab/Downloads/stl_rotated_correctly/stl")
#     stl_list = list(stl_dir.rglob("*.stl"))
#     stl_list = [s for s in stl_list if "Q" not in s.name]
#     print("Number of stl files:", len(stl_list))
