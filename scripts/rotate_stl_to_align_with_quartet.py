# Roates the STLs in a given directory such that they align with their actual placement in a quartet. Specifically, during the original creation of the quartets, there was some discrepancy in how the interface portion was rotated based on the different script used. When assembling the quartets, I made sure that in all cases, the small square mark on each shape is in the up position. For ease of future analysis, this script will generate a new stl that has been rotated, and will check that the small square mark is in the right position.
from pathlib import Path
import trimesh
import numpy as np


def check_if_rotated_correctly(
    mesh, test_box_T, text_box_extents, num_vertices_threshold_min, num_vertices_rotated_max
):
    """
    Check if the mesh is rotated correctly by verifying the number of vertices within a defined test box.
    """
    vertices = mesh.vertices
    within_box = np.all(
        (vertices >= test_box_T[:3, 3] - text_box_extents / 2) & (vertices <= test_box_T[:3, 3] + text_box_extents / 2),
        axis=1,
    )
    num_vertices_within = np.sum(within_box)

    if num_vertices_within < num_vertices_threshold_min or num_vertices_within > num_vertices_rotated_max:
        return False
    else:
        return True


def rotate_stls_to_align_notch(old_stl_dir, new_stl_dir):

    # Inputs
    num_vertices_threshold_min = 59
    num_vertices_rotated_max = 61
    test_box_side_length = 5
    test_box_T = np.eye(4)
    test_box_T[:3, 3] = [20, 0, -43]

    if new_stl_dir.exists() == False:
        new_stl_dir.mkdir(parents=True, exist_ok=True)

    # Read old STL files
    old_stl_files = list(old_stl_dir.rglob("*.stl"))
    old_stl_files = [stl for stl in old_stl_files if not stl.name.startswith("Q")]  # Filter out hidden files
    old_stl_files.sort()

    # Check if there are any STL files to process
    all_correct = True
    for stl_file in old_stl_files:
        mesh = trimesh.load_mesh(stl_file)
        is_rotated_correctly = check_if_rotated_correctly(
            mesh, test_box_T, np.array([test_box_side_length] * 3), num_vertices_threshold_min, num_vertices_rotated_max
        )
        if not is_rotated_correctly:
            print(f"Mesh {stl_file.name} is not rotated correctly.")
            all_correct = False
            break

    if all_correct:
        print("All meshes are already rotated correctly. No changes made.")
        return

    # Define where the small square should be, as well as a test of its location
    for i in range(len(old_stl_files)):
        mesh = trimesh.load_mesh(old_stl_files[i])
        # mesh.show()

        test_box = trimesh.primitives.Box(
            extents=(test_box_side_length, test_box_side_length, test_box_side_length),
            transform=test_box_T,
        )

        scene = trimesh.Scene([mesh, test_box])
        # scene.show()

        # Check if the mesh is rotated correctly
        is_rotated_correctly = check_if_rotated_correctly(
            mesh, test_box_T, np.array([test_box_side_length] * 3), num_vertices_threshold_min, num_vertices_rotated_max
        )
        if is_rotated_correctly == False:

            # Rotate about Z-axis 90deg
            rotation_angle = np.pi / 2
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, [0, 0, 1])
            mesh.apply_transform(rotation_matrix)

            is_rotated_correctly = check_if_rotated_correctly(
                mesh,
                test_box_T,
                np.array([test_box_side_length] * 3),
                num_vertices_threshold_min,
                num_vertices_rotated_max,
            )

            if is_rotated_correctly == False:
                print(
                    f"Mesh {old_stl_files[i].name} still does not have enough vertices within the test box after rotation."
                )
                scene = trimesh.Scene([mesh, test_box])
                scene.show()
                break

        # Save in a new directory
        new_stl_path = new_stl_dir / old_stl_files[i].relative_to(old_stl_dir)

        # Mae sure the parent directory exists
        new_stl_path.parent.mkdir(parents=True, exist_ok=True)

        # Export the rotated mesh
        mesh.export(new_stl_path)
        print(f"Saved rotated mesh to {new_stl_path}")


if __name__ == "__main__":
    old_stl_dir = Path("/home/oconnorlab/Downloads/stl")
    new_stl_dir = Path("/home/oconnorlab/Downloads/stl_rotated_correctly/stl")

    rotate_stls_to_align_notch(old_stl_dir, new_stl_dir)
