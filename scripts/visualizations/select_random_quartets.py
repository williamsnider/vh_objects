import trimesh
from pathlib import Path
import numpy as np
import fast_simplification

# def sample_shapes(stl_dir, cart_dir):


def simplify_mesh(mesh, target_reduction=0.8):
    """
    Simplify a mesh using fast_simplification.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        target_reduction (float): The target reduction ratio.

    Returns:
        trimesh.Trimesh: The simplified mesh.
    """
    mesh_simplified = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=target_reduction)
    return trimesh.Trimesh(mesh_simplified[0], mesh_simplified[1])


def combine_quartet(fname_cart, fname_meshA, fname_meshB, fname_meshC, fname_meshD):
    cart = trimesh.load_mesh(fname_cart)
    meshA = trimesh.load_mesh(fname_meshA)
    meshB = trimesh.load_mesh(fname_meshB)
    meshC = trimesh.load_mesh(fname_meshC)
    meshD = trimesh.load_mesh(fname_meshD)

    # Simplify meshes to reduce complexity
    # cart = simplify_mesh(cart, target_reduction=0.8)
    if "sheet" not in str(fname_meshA):
        meshA = simplify_mesh(meshA, target_reduction=0.0)
    if "sheet" not in str(fname_meshB):
        meshB = simplify_mesh(meshB, target_reduction=0.0)
    if "sheet" not in str(fname_meshC):
        meshC = simplify_mesh(meshC, target_reduction=0.0)
    if "sheet" not in str(fname_meshD):
        meshD = simplify_mesh(meshD, target_reduction=0.0)

    center_dist = 75  # mm
    spacing = np.linspace(-3 * center_dist / 2, 3 * center_dist / 2, 4)
    spacing = np.flip(spacing)  # Reverse order for ABCD (right to left)

    # Apply translations
    meshA.apply_translation([0, spacing[0], 0])
    meshB.apply_translation([0, spacing[1], 0])
    meshC.apply_translation([0, spacing[2], 0])
    meshD.apply_translation([0, spacing[3], 0])

    # Combine into single meshes
    faces = []
    vertices = []
    for m in [cart, meshA, meshB, meshC, meshD]:
        faces.extend(m.faces + len(vertices))
        vertices.extend(m.vertices)

    combined = trimesh.Trimesh(vertices=vertices, faces=faces)
    return combined


def sample_randomly_and_select_quartets(stl_dir, cart_dir):

    stl_list = list(stl_dir.rglob("*.stl"))
    stl_list = [x for x in stl_list if "texture" not in x.parts]
    cart_list = list(cart_dir.rglob("*.stl"))
    cart_list.sort()

    np.random.shuffle(stl_list)

    # Pad to multiple of 4
    i = 0
    while len(stl_list) % 4 != 0:
        stl_list.append(stl_list[i])
        print(f"Padding final quartet with {stl_list[i].name}")
        i += 1

    quartet_count = 0
    quartet_and_shape_groups = []

    # Group into sets of 4
    for i in range(0, len(stl_list), 4):
        quartet = cart_list[quartet_count]
        shapeA = stl_list[i]
        shapeB = stl_list[i + 1]
        shapeC = stl_list[i + 2]
        shapeD = stl_list[i + 3]

        quartet_and_shape_groups.append((quartet, shapeA, shapeB, shapeC, shapeD))

        quartet_count += 1

    return quartet_and_shape_groups


if __name__ == "__main__":

    # Sample shapes
    stl_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes_no_duplicates/stl")
    cart_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/quartet")
    quartet_and_shape_groups = sample_randomly_and_select_quartets(stl_dir, cart_dir)

    # Combine each quartet with 4 shapes
    save_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes/stl/demo_full_quality")
    # Save examples
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    demo_quartets = []
    for i in range(len(quartet_and_shape_groups)):
        print(i)
        demo = combine_quartet(*quartet_and_shape_groups[i])
        demo_quartets.append(demo)

        # Save each demo
        demo.export(save_dir / f"D{str(i).zfill(3)}.stl")

    # # Visualize
    # for i in range(len(demo_quartets)):
    #     scene = trimesh.Scene()
    #     scene.add_geometry(demo_quartets[i])
    #     scene.show(smooth=False)
