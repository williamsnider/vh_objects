import trimesh
import numpy as np
from pathlib import Path


demo_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/demo")
stl_list = list(demo_dir.rglob("*.stl"))
stl_list.sort()
stl_list.extend(stl_list)  # Duplicate so final shelf is full

offset = -15


radius_list = np.array(
    [
        0.675,
        0.765,
        0.81,
        0.835,
        0.85,
        0.85,
        0.85,
        0.83,
        0.815,
        0.775,
        0.725,
        0.645,
        0.525,
    ]
)
radius_list *= 1000  # m to mm
radius_list += offset
radius_list = radius_list.tolist()

quartet_width = 290
shelf_spacing = 100

scene = trimesh.Scene()
meshes_to_combineA = []
meshes_to_combineB = []
meshes_to_combine_counter = 0

stl_start_idx = 0
for shelf_num, radius in enumerate(radius_list[::-1]):

    if shelf_num % 2 == 0:
        meshes_to_combine_counter = 0
    else:
        meshes_to_combine_counter = 1

    # Create shelf
    # num_stim = 7
    # stl_list = stl_list[:num_stim]
    # th_list = np.linspace(0 + th_offset, 3 * np.pi / 2 - th_offset, num_stim)

    arc_length = 2 * np.pi * radius * (3 / 4)
    num_quartets_on_shelf = int(arc_length / quartet_width)
    # centers_arc_length = quartet_width * np.arange(num_quartets_on_shelf, 0, -1)
    centers_arc_length = quartet_width * np.arange(num_quartets_on_shelf)
    centers_thetas = -centers_arc_length / radius

    stl_list_sub = stl_list[stl_start_idx : stl_start_idx + num_quartets_on_shelf]

    for i, stl_path in enumerate(stl_list_sub):
        print("Adding", stl_path.name)
        stl = trimesh.load_mesh(stl_path)

        # Align to base of shape
        stl.apply_translation([0, 0, -stl.bounds[0][2]])

        # Rotate to be pointing outward
        stl.apply_transform(
            trimesh.transformations.rotation_matrix(centers_thetas[i], [0, 0, 1])
            @ trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            @ trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0))
        )

        T = np.eye(4)
        T[0, 3] = radius * np.cos(centers_thetas[i])
        T[1, 3] = radius * np.sin(centers_thetas[i])
        T[2, 3] = -shelf_spacing * shelf_num

        stl.apply_transform(T)

        if meshes_to_combine_counter == 0:
            meshes_to_combineA.append(stl)
        else:
            meshes_to_combineB.append(stl)
        meshes_to_combine_counter = (meshes_to_combine_counter + 1) % 2
        # scene.add_geometry(stl)

    stl_start_idx += num_quartets_on_shelf
# scene.show()


def export_meshes_to_combine(meshes_to_combine, fname):
    comb_faces = []
    comb_verts = []
    for mesh in meshes_to_combine:
        comb_faces.extend(mesh.faces + len(comb_verts))
        comb_verts.extend(mesh.vertices)
    comb = trimesh.Trimesh(np.vstack(comb_verts), np.vstack(comb_faces), process=False)
    comb.export(fname)


export_meshes_to_combine(meshes_to_combineA, Path(demo_dir, "shelfA.stl"))
export_meshes_to_combine(meshes_to_combineB, Path(demo_dir, "shelfB.stl"))


# scene.show(smooth=False)
