import trimesh
from objects.parameters import (
    CUBE_SIDE_LENGTH,
    PEG_SIDE_LENGTH,
    PEG_DEPTH,
    POST_RADIUS,
    POST_HEIGHT,
    POST_SECTIONS,
    FINGERTIP_SLOT_SIDE_LENGTH,
    PEG_CORNER_RADIUS,
    PEG_CORNER_NUM_STEPS,
    HARMONIC_POWER,
    PEG_SPHERE_SUBDIVISIONS,
)
from compas_cgal.booleans import boolean_intersection, boolean_union, boolean_difference
import numpy as np
from shapely.geometry import Polygon
import igl
from objects.utilities import plot_mesh_and_specific_indices
import scipy
import matplotlib.pyplot as plt


class Interface:
    """
    What the robot will grab onto
    """

    def __init__(
        self,
        label="test123",
    ):
        self.label = label

        # Make the interface
        self.make_base_interface()

    def make_base_interface(self):

        # TODO: Rewrite using compas.geometry primitives and see if that allows the interface to be watertight with POST_SECTIONS=8.

        ################ Make cube ################
        cube = trimesh.primitives.Box(extents=(CUBE_SIDE_LENGTH, CUBE_SIDE_LENGTH, CUBE_SIDE_LENGTH))
        cube_VF = [cube.vertices, cube.faces]

        ################ Make post ################
        T = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, POST_HEIGHT / 2 + CUBE_SIDE_LENGTH / 2],
                [0, 0, 0, 1],
            ]
        )
        post = trimesh.primitives.Cylinder(
            radius=POST_RADIUS,
            height=POST_HEIGHT,
            sections=POST_SECTIONS,
            transform=T,
        )
        post_VF = [post.vertices, post.faces]
        base_interface_VF = boolean_union(cube_VF, post_VF)

        ################ Make fingertip slots ################
        # +Y side
        T = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, CUBE_SIDE_LENGTH / 2 - FINGERTIP_SLOT_SIDE_LENGTH / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        fingertip_slot_A = trimesh.primitives.Box(
            extents=(FINGERTIP_SLOT_SIDE_LENGTH, FINGERTIP_SLOT_SIDE_LENGTH, FINGERTIP_SLOT_SIDE_LENGTH),
            transform=T,
        )
        fingertip_slot_A_VF = [fingertip_slot_A.vertices, fingertip_slot_A.faces]
        base_interface_VF = boolean_difference(base_interface_VF, fingertip_slot_A_VF)

        # -Y side
        fingertip_slot_B = trimesh.primitives.Box(
            extents=(FINGERTIP_SLOT_SIDE_LENGTH, FINGERTIP_SLOT_SIDE_LENGTH, FINGERTIP_SLOT_SIDE_LENGTH),
            transform=-T,
        )
        fingertip_slot_B_VF = [fingertip_slot_B.vertices, fingertip_slot_B.faces]
        base_interface_VF = boolean_difference(base_interface_VF, fingertip_slot_B_VF)

        ################ Make peg ################
        # Peg needs
        # 1) rounded edges (because the waterjet-cut slots have rounded corners) and
        # 2) a roundover on the bottom/at the tip (to guide the peg into the slot)

        # Construct shapely polygon that we can then extrude
        # Rename variables for brevity
        L = PEG_SIDE_LENGTH
        R = PEG_CORNER_RADIUS
        c = np.cos
        s = np.sin

        # Sample unit circle from 0 to np.pi/2
        t = np.linspace(0, np.pi / 2, PEG_CORNER_NUM_STEPS)
        quarter_circle = np.stack([R * c(t).T, R * s(t).T, np.ones(t.shape)], axis=1)  # Homogenous coordinates

        # Translate and rotate
        curve_pts = np.zeros([4 * PEG_CORNER_NUM_STEPS + 1, 2])
        for i, a in enumerate([0, np.pi / 2, np.pi, 3 * np.pi / 2]):

            translation = np.array([[1, 0, L / 2 - R], [0, 1, L / 2 - R], [0, 0, 1]])
            rotation = np.array([[c(a), -s(a), 0], [s(a), c(a), 0], [0, 0, 1]])
            T = rotation @ translation
            pts = quarter_circle @ T.T

            start = PEG_CORNER_NUM_STEPS * i
            stop = PEG_CORNER_NUM_STEPS * (i + 1)
            curve_pts[start:stop, :] = pts[:, :2]
        curve_pts[-1] = curve_pts[0]  # Make first and last point the same
        poly = Polygon(curve_pts)

        # Extrude this polygon to make the peg_base
        peg_base = trimesh.creation.extrude_polygon(polygon=poly, height=PEG_DEPTH)
        peg_base_VF = [peg_base.vertices, peg_base.faces]

        # Create icosphere which will be fused to be the tip of the peg.
        # Sphere is chosen here because its highest (and centralmost) vertex will be at (0, 0, highest_point), and we can thus easily identify it and move its position to control the depth of the roundover.
        sphere = trimesh.primitives.Sphere(
            radius=(PEG_SIDE_LENGTH / 2),
            center=np.array([0, 0, PEG_DEPTH]),
            subdivisions=PEG_SPHERE_SUBDIVISIONS,
        )
        sphere_VF = [sphere.vertices, sphere.faces]

        # Combine peg_base and sphere to form peg_template
        peg_template_VF = boolean_union(peg_base_VF, sphere_VF)
        peg_template = trimesh.Trimesh(
            vertices=peg_template_VF[0],
            faces=peg_template_VF[1],
        )

        # Before we fair the peg_tip, we need to introduce another set of vertices at the height of the peg_base that we want to maintain. We can do this by slicing the peg_template at this height and stitching the two pieces back together.
        Z_at_roundover_beginning = PEG_DEPTH - PEG_SIDE_LENGTH / 2
        plane_origin = np.array([0, 0, Z_at_roundover_beginning])
        plane_normal = np.array([0, 0, 1])
        peg_bottom = peg_template.slice_plane(
            plane_origin=plane_origin,
            plane_normal=-plane_normal,
        )
        peg_top = peg_template.slice_plane(
            plane_origin=plane_origin,
            plane_normal=plane_normal,
        )

        # Remove duplicate vertices - necessary to create watertight union
        peg_bottom.process()
        peg_top.process()

        ###### Stitch together peg_bottom and peg_top #####
        renumber_table = np.zeros(peg_top.vertices.shape[0], dtype="int")
        tree = scipy.spatial.KDTree(peg_bottom.vertices)
        dd, ii = tree.query(peg_top.vertices, k=1)
        peg_bottom_num_verts = peg_bottom.vertices.shape[0]
        count = peg_bottom_num_verts  # Start new numbering to account for number of peg_bottom verts
        for i, v in enumerate(peg_top.vertices):
            old_num = i

            if dd[i] == 0:
                new_num = ii[i]
            else:
                new_num = count
                count += 1
            renumber_table[old_num] = new_num

        nonoverlapping = dd != 0
        peg_top_verts_nonoverlapping = peg_top.vertices[nonoverlapping]  # Verts not in peg_bottom
        peg_top_faces_renumbered = renumber_table[
            peg_top.faces
        ]  # indices are old numbers, values are new numbers (faster than using a dict)

        # Concatenate peg_bottom and renumbered peg_top vertices/faces
        unfaired_verts = np.concatenate([peg_bottom.vertices, peg_top_verts_nonoverlapping], axis=0)
        unfaired_faces = np.concatenate([peg_bottom.faces, peg_top_faces_renumbered], axis=0)
        unfaired = trimesh.Trimesh(
            vertices=unfaired_verts, faces=unfaired_faces, process=True
        )  # TODO: Figure out how to make it watertight without processing. No vertices seem to be duplicated? But this causes vertices to be dropped? Maybe some are just really close together
        assert unfaired.is_watertight, "Unfaired Peg is not watertight."

        # Boundary vertices -- located at or below Z = Z_at_roundover_beginning
        boundary = unfaired.vertices[:, 2] <= Z_at_roundover_beginning

        # Set tip of rounder at specified Z, as well as adjacent
        roundover_tip_idx = np.argmax(unfaired.vertices[:, 2])  # Assume highest point is tip
        boundary[roundover_tip_idx] = True

        # Shift all non_boundary points up a similar amount to the roundover_tip_idx
        # z_shift = PEG_DEPTH - unfaired.vertices[roundover_tip_idx, 2]

        # neighbors = unfaired.vertex_neighbors[roundover_tip_idx]
        # roundover_adjacent = set()
        # for n in neighbors:
        #     roundover_adjacent.update(unfaired.vertex_neighbors[n])
        # roundover_top = list(roundover_adjacent)
        # # roundover_top = roundover_adjacent + [roundover_tip_idx]
        # unfaired.vertices[roundover_top, 2] = PEG_DEPTH  # Flatten tip; set to PEG_DEPTH height
        # boundary[roundover_top] = True  # Use this to control depth of roundover

        # bbox = peg_base.bounds
        # bbox = (bbox.T + np.array([-1e-8, 1e-8])).T  # Slightly increase bbox range to include points exactly on border
        # new_verts_mask = ~trimesh.bounds.contains(bbox, peg.vertices)

        # # Move roundover_tip and adjacent verts to max position

        # # # And remove where sphere intersects edges of peg
        # # intersection_pts = np.array(
        # #     [
        # #         [0, PEG_SIDE_LENGTH / 2, height_to_roundover_beginning],
        # #         [0, -PEG_SIDE_LENGTH / 2, height_to_roundover_beginning],
        # #         [-PEG_SIDE_LENGTH / 2, 0, height_to_roundover_beginning],
        # #         [PEG_SIDE_LENGTH / 2, 0, height_to_roundover_beginning],
        # #     ]
        # # )
        # # for pt in intersection_pts:
        # #     idx = np.argwhere(np.all(pt == peg.vertices, axis=1))
        # #     print(idx)
        # #     new_verts_mask[idx] = False
        # new_verts = np.arange(peg.vertices.shape[0], dtype="int64")[~new_verts_mask]
        # plot_mesh_and_specific_indices(peg, new_verts)

        # Fair the new_verts to make the roundover
        v = unfaired.vertices
        f = unfaired.faces
        b = np.argwhere(boundary).ravel()  # Boundary - do not fair vertices from peg_bottom
        bc = v[b]  # XYZ coordinates of the boundary indices
        faired_verts = igl.harmonic_weights(v, f, b, bc, 3)  # Smooths indices at creases

        # Assign to mesh
        peg = trimesh.Trimesh(
            vertices=faired_verts,
            faces=f,
        )
        peg.show(smooth=False)

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=-90, azim=90)

        x, y, z = peg.vertices[:peg_bottom_num_verts].T
        ax.plot(x, y, z, "b.")

        # x, y, z = peg_bottom.vertices.T
        # ax.plot(x, y, z, "r*")
        # plt.show()

        x, y, z = peg_top.vertices.T
        ax.plot(x, y, z, "r*")
        plt.show()

        plot_mesh_and_specific_indices(unfaired, b)

        assert mesh.is_watertight, "Mesh is not watertight. Try adjusting the number of POST_SECTIONS."

        pass

    def add_label(self):
        pass

    def extrude_post_to_shape(self):
        pass


# Outline
# Generate shape
# Determine bounding box of shape (use this to exclude shapes that are too large)
# Orient interface to this bounding box (bottommost portion of shape should be along same plane as bottom of peg)
# Use pre-built interface (containing cube, fingertip, peg, and post) (missing label and extrusion to shape)
# Extrude the end of the post to the base of the shape
