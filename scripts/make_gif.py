# Load STL file and create a gif of the object rotating

from pathlib import Path
import trimesh
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pyrender
import scipy
import cv2
from trimesh.transformations import rotation_matrix as rotvec2T


def size_check(fname_stl):

    EXTENT_THRESHOLD = 45.1

    # Load the STL file
    mesh = trimesh.load_mesh(fname_stl)
    extents = mesh.extents

    if any(extents[:2] > EXTENT_THRESHOLD):
        print(f"Shape {fname_stl} is too large and has extents {extents}; FIX THIS")


def make_gif(fname_stl):

    # Load the STL file
    mesh = trimesh.load_mesh(fname_stl)

    # Set up the plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create a list to store the images
    images = []

    # Number of frames for the rotation
    frames = 36

    # Rotate the model and save each frame
    for angle in np.linspace(0, 360, frames):
        ax.clear()

        # Rotate the mesh
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])
        )  # Rotate around y-axis

        # Plot the mesh
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            cmap=plt.cm.Spectral,
            linewidth=0.2,
            edgecolor="k",
        )

        # Set the plot limits
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([-100, 100])

        # Hide axes
        ax.axis("off")

        # Save the frame
        plt.tight_layout()
        fig.canvas.draw()

        # Convert to image and add to list
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    # Save the images as a GIF
    fname_gif = Path(fname_stl.parents[1], "gifs", fname_stl.parts[-2], fname_stl.name.replace(".stl", ".gif"))
    if fname_gif.parents[0].exists() is False:
        fname_gif.parents[0].mkdir(parents=True)
    imageio.mimsave(str(fname_gif), images, fps=10)

    plt.close(fig)


def make_gif_pyrender(fname_stl):
    # Load the STL file using trimesh
    mesh = trimesh.load(fname_stl)

    # Create a pyrender mesh from the trimesh mesh
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Set up the scene and add the mesh
    scene = pyrender.Scene()
    mesh_node = scene.add(pyrender_mesh)

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array(
        [
            [1.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scene.add(camera, pose=camera_pose)

    # Set up the light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    # Set up the renderer
    renderer = pyrender.OffscreenRenderer(400, 400)

    # Create a list to store images
    images = []

    # Number of frames for the rotation
    frames = 36

    # Rotate the model and save each frame
    for angle in np.linspace(0, 360, frames):
        # Rotate the mesh by applying a transformation matrix
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 0, 1])  # Rotate around y-axis
        scene.set_pose(mesh_node, pose=rotation_matrix)

        # Render the scene
        color, _ = renderer.render(scene)

        # Add the image to the list
        images.append(color)

    # Save the images as a GIF
    fname_gif = Path(fname_stl.parents[1], "gifs", fname_stl.parts[-2], fname_stl.name.replace(".stl", ".gif"))
    if fname_gif.parents[0].exists() is False:
        fname_gif.parents[0].mkdir(parents=True)
    imageio.mimsave("rotating_model_pyrender.gif", images, fps=10)

    # Clean up the renderer
    renderer.delete()


def make_gif_pyvista(fname_stl):
    import numpy as np

    import pyvista as pv
    import scipy.spatial
    import pickle
    import trimesh.exchange

    tri_mesh = trimesh.load_mesh(fname_stl)

    mesh = pv.wrap(tri_mesh)
    pts = mesh.points

    window_size = (1200, 1200)

    FPS = 10
    FRAMES = 60
    plotter = pv.Plotter(notebook=False, off_screen=True, window_size=window_size)
    plotter.add_mesh(
        mesh,
        lighting=True,
        show_edges=False,
    )

    fname_gif = Path(fname_stl.parents[2], "gif", fname_stl.parts[-2], fname_stl.name.replace(".stl", ".gif"))
    if fname_gif.parents[0].exists() is False:
        fname_gif.parents[0].mkdir(parents=True)
    plotter.open_gif(str(fname_gif), fps=FPS)

    # Write a frame. This triggers a render.
    for th in np.linspace(0, 2 * np.pi, FRAMES, endpoint=False):
        shape_euler = np.array([0, 0, th])
        R = scipy.spatial.transform.Rotation.from_euler("xyz", shape_euler).as_matrix()
        R_pts = pts @ R
        mesh.points = R_pts
        plotter.write_frame()

    # Closes and finalizes movie
    plotter.close()

    return fname_gif


from PIL import Image, ImageSequence


def combine_gif_and_image(fname_gif):

    # Open the rotating GIF
    rotating_gif = Image.open(fname_gif)

    # Prepare a list to store the combined frames
    frames = []

    # Combine each frame of the GIF with the static image
    first_frame = None
    count = -1
    for frame in ImageSequence.Iterator(rotating_gif):

        if first_frame == None:
            first_frame = frame.copy()

        # Create a new image with double the width of the frame
        new_frame = Image.new("RGB", (2 * frame.width, frame.height))

        # Paste the rotating frame and the static image side by side
        new_frame.paste(first_frame, (0, 0))
        new_frame.paste(frame, (frame.width, 0))

        # Convert to palette-based (if the GIF is palette-based)
        new_frame = new_frame.convert("P", palette=Image.ADAPTIVE)

        # Add the new frame to the list
        frames.append(new_frame)

    # Save the frames as a new GIF
    fname_gif_combined = Path(fname_gif.parents[2], "gifs_combined", fname_gif.parts[-2], fname_gif.name)
    if fname_gif_combined.parents[0].exists() is False:
        fname_gif_combined.parents[0].mkdir(parents=True)
    frames[0].save(fname_gif_combined, save_all=True, append_images=frames[1:], loop=0, duration=100)


def save_mesh_as_png(fname_stl):
    """
    Saves the mesh as a png.
    """

    fname_png = Path(fname_stl.parents[2], "png", fname_stl.parts[-2], fname_stl.name.replace(".stl", ".png"))
    if fname_png.parents[0].exists() is False:
        fname_png.parents[0].mkdir(parents=True)

    # Compose scene
    scene = pyrender.Scene(ambient_light=[0.1, 0.5, 0.3], bg_color=[1, 1, 1])

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
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    # Rotate mesh to get better view
    mesh_pose = np.eye(4)
    # TZ90 = rotvec2T(-np.pi / 2, [0, 0, 1])
    TX90 = rotvec2T(-np.pi / 2.1, [1, 0, 0])
    mesh_pose = TX90
    mesh_pose[2, 3] -= 10

    scene.add(mesh, pose=mesh_pose)

    # Camera pose explained:
    # +X axis is towards the right of the screen
    # +Y axis is towards the top of the screen
    # -Z axis points into the screen (camera looks into the screen)
    yfov = np.pi / 4.0
    ywidth = 120  # mm
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

    # Add directional light
    light_euler = np.array([-np.pi / 4, 0, 0])
    R = scipy.spatial.transform.Rotation.from_euler("xyz", light_euler).as_matrix()
    light_pose = np.eye(4)
    light_pose[:3, :3] = R
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5e3)
    scene.add(light, pose=light_pose)

    resolution = (1200, 1200)
    r = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color, _ = r.render(scene)

    cv2.imwrite(str(fname_png), color)

    return fname_png


# def make_combined_gif(fname_gif):

#     # Load gif
#     images = imageio.mimread(str(fname_gif))

#     # Create new gif twice as wide, with first frame on left and all frames (gif) on right
#     new_images = []
#     for i, image in enumerate(images):
#         new_image = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
#         new_image[:, : image.shape[1], :] = image
#         new_image[:, image.shape[1] :, :] = images[0]
#         new_images.append(new_image)

#     # Save new gif
#     fname_gif_combined = Path(fname_stl.parents[1], "gifs", fname_stl.parts[-2], fname_stl.name.replace(".stl", ".gif"))
#     fname_combined_gif = fname_gif.replace(".gif", "_combined.gif")


# Use multiprocessing to go through list
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm


def process_file(fname_stl):
    size_check(fname_stl)
    fname_png = save_mesh_as_png(fname_stl)
    # fname_gif = make_gif_pyvista(fname_stl)
    # # combine_gif_and_image(fname_gif)


if __name__ == "__main__":

    base_dir = Path(__file__).parents[1]
    overall_dir = Path(base_dir, "sample_shapes/stl/")

    fname_stl_all = list(overall_dir.rglob("*.stl"))
    # Use a multiprocessing pool
    with Pool() as pool:
        # Use tqdm to display the progress bar
        for _ in tqdm(
            pool.imap_unordered(process_file, fname_stl_all), total=len(fname_stl_all), desc="Processing STL files"
        ):
            pass

    # fname_gif = make_gif_pyvista(fname_stl)
    # combine_gif_and_image(fname_gif)
# fname_stl = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/axial_component/axial_component_009.stl")
# fname_png = save_mesh_as_png(fname_stl)
# fname_gif = make_gif_pyvista(fname_stl)

# combine_gif_and_image(fname_gif)
