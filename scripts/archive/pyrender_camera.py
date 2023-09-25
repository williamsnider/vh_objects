import pyrender
import trimesh
import numpy as np
import scipy.spatial

resolution = [1900 / 4, 1200 / 4]
# Compose scene
scene = pyrender.Scene(ambient_light=[0.1, 0.5, 0.3], bg_color=[1, 1, 1])

# Add mesh and interface
box = trimesh.primitives.Box(extents=[100, 50, 25])
mesh_pose = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
mesh = pyrender.Mesh.from_trimesh(box, smooth=False)
scene.add(mesh, pose=mesh_pose)


# Camera pose explained:
# +X axis is towards the right of the screen
# +Y axis is towards the top of the screen
# -Z axis points into the screen (camera looks into the screen)
yfov = np.pi / 4.0
ywidth = 100  # mm
camera_pose = np.eye(4)
camera_pose[2, 3] = (
    ywidth / 2 / np.tan(yfov / 2)
)  # Calculate correct distance for camera to have ywidth and yfov
camera = pyrender.PerspectiveCamera(yfov=yfov)
scene.add(camera, pose=camera_pose)

# Add directional light
light_euler = np.array([-np.pi / 4, np.pi / 4, 0])
R = scipy.spatial.transform.Rotation.from_euler("xyz", light_euler).as_matrix()
light_pose = np.eye(4)
light_pose[:3, :3] = R
light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.5e3)
scene.add(light, pose=light_pose)

pyrender.Viewer(scene, use_raymond_lighting=True)
# # TODO: This is not 16 bit depth
# r = pyrender.Renderer(resolution[0], resolution[1])
# # r = pyrender.OffscreenRenderer(resolution[0], resolution[1], bitdepth="16bit")
# color, _ = r.render(scene)
