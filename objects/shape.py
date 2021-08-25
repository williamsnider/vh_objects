import trimesh
import numpy as np
from objects.parameters import SAMPLING_DENSITY_V, SAMPLING_DENSITY_U, ORDER
from objects.utilities import open_uniform_knot_vector
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy
import networkx as nx
from splipy import BSplineBasis, Curve, Surface


class Shape:
    def __init__(self, ac_list):

        self.ac_list = ac_list

    def check_inputs(self):

        assert (
            type(self.ac_list) is list
        ), "ac_list must be a list, even if it has just 1 ac."

    def fuse_meshes(self, parent_ac, child_ac):

        # TODO: Do these on a simpler mesh to go faster

        parent = parent_ac.mesh  # Parent is a mesh
        child = child_ac  # Child is a single axial component B Splien Surface

        # Identify the slice on the child that is closest, but outside to the surface of the parent.
        num_steps_long_axis = 10  # Try 10 different slices
        num_steps_round_axis = (
            6  # If all 6 points in a slice are outside mesh, safe to assume all is
        )
        (us, vs) = child.surface.start()
        (ue, ve) = child.surface.end()
        u = np.linspace(us, ue, num_steps_round_axis, endpoint=False)
        v = np.linspace(vs, ve, num_steps_long_axis)
        verts_array = child.surface(u, v)

        for slice_num, _ in enumerate(v):
            print(slice_num)
            points = verts_array[:, slice_num, :]

            points_outside_mesh = ~parent.contains(points)
            if np.all(points_outside_mesh):  # All points outside
                break

            if slice_num == len(v) - 1:
                raise NotImplementedError

        slice_dist_approx = np.array(
            v[slice_num]
        )  # Need to round this up to next highest value in actual v

        # Grab the full-size slice
        uu = SAMPLING_DENSITY_U
        vv = SAMPLING_DENSITY_V
        (us, vs) = child.surface.start()
        (ue, ve) = child.surface.end()
        v = np.linspace(vs, ve, vv)
        u = np.linspace(us, ue, uu, endpoint=False)
        slice_dist = v[v > slice_dist_approx][0]  # First value of v > slice_dist_approx
        full_slice = child.surface(u, slice_dist)
        # full_slice = child.surface(u, slice_dist)

        # Expand this slice and project it onto the surface of the parent

        TNB_current = np.stack(
            [
                child.T(slice_dist)[0],
                child.N(slice_dist)[0],
                child.B(slice_dist)[0],
            ],
            axis=0,
        )

        TNB_goal = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

        R = np.linalg.inv(TNB_current) @ TNB_goal

        EXPANSION_FACTOR = 1.5
        center = child.r(slice_dist)
        full_slice_rotated = ((full_slice - center) @ R * EXPANSION_FACTOR) + center
        mesh_verts_rotated = (parent.vertices - center) @ R + center

        # Remove x-axis
        full_slice_yz = np.squeeze(full_slice_rotated[:, :, 1:])
        mesh_verts_yz = mesh_verts_rotated[:, 1:]

        # Identify the 5 nearest neighbors for each point on the slice
        NUM_NN = 5
        tree = scipy.spatial.KDTree(mesh_verts_yz)
        dd, ii = tree.query(full_slice_yz, k=NUM_NN)

        # Choose the NN with the shortest distance in 3D
        mesh_points = mesh_verts_rotated[ii]
        slice_points = np.repeat(full_slice_rotated, NUM_NN, axis=1)
        dist = np.sqrt(np.sum((mesh_points - slice_points) ** 2, axis=2))
        min_idx = np.argmin(dist, axis=1)
        closest_NN = np.zeros((len(ii)), dtype="int")
        for i, _ in enumerate(closest_NN):
            closest_NN[i] = ii[i, min_idx[i]]

        _, unique_idx = np.unique(closest_NN, return_index=True)
        unique_NN = [closest_NN[i] for i in sorted(unique_idx)]
        unique_NN.append(unique_NN[0])  # Wrap starting point

        # Shortest path between two vertices
        mesh = parent

        # edges without duplication
        edges = mesh.edges_unique

        # the actual length of each unique edge
        length = mesh.edges_unique_length

        # create the graph with edge attributes for length
        g = nx.Graph()
        for edge, L in zip(edges, length):
            g.add_edge(*edge, length=L)

        # alternative method for weighted graph creation
        # you can also create the graph with from_edgelist and
        # a list comprehension, which is like 1.5x faster
        ga = nx.from_edgelist(
            [(e[0], e[1], {"length": L}) for e, L in zip(edges, length)]
        )

        # arbitrary indices of mesh.vertices to test with
        full_path = []
        for i in range(len(unique_NN)):
            print(i)
            start = unique_NN[i - 1]
            end = unique_NN[i]

            # run the shortest path query using length for edge weight
            new_path = nx.shortest_path(g, source=start, target=end, weight="length")

            # If this path overlaps previous path, remove the previous path
            repeated_verts = [p for p in new_path[1:] if p in full_path]
            if repeated_verts:

                # Find the intermediate vertices (between the repeated) and cut them out

                # Find the index of the last repeat in the new path
                idx_path_repeat = -1
                for r in repeated_verts:
                    idx = new_path.index(r)
                    if idx > idx_path_repeat:
                        idx_path_repeat = idx

                # Find the index of the first repeat in the full path
                idx_full_repeat = full_path.index(repeated_verts[0])  # Initilize
                for r in repeated_verts:
                    idx = full_path.index(r)
                    if idx < idx_full_repeat:
                        idx_full_repeat = idx

                # Get the vertices to be appended (skip the repeating ones)
                try:
                    verts_to_append = new_path[idx_path_repeat + 1 :]
                except:
                    verts_to_append = []

                # Get the vertices for which we need to replace the assigned NN
                NNs_to_replace = (
                    full_path[idx_full_repeat + 1 :] + new_path[:idx_path_repeat]
                )
                NN_replacement = full_path[idx_full_repeat]  # Just assign all

                # Replace these NNs for the closest_NN array
                for NN in NNs_to_replace:

                    closest_NN[closest_NN == NN] = NN_replacement
            else:
                verts_to_append = new_path[1:]  # Don't duplicate first vert

            # Append to the full_path
            full_path.extend(verts_to_append)

        full_path.append(full_path[0])  # Add first element to end to close loop
        # Plot 2D points
        fig, ax = plt.subplots()
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_xlim([full_slice_yz.min(), full_slice_yz.max()])
        ax.set_ylim([full_slice_yz.min(), full_slice_yz.max()])

        y = full_slice_yz[:, 0]
        z = full_slice_yz[:, 1]
        ax.plot(y, z, "b*")

        # y = mesh_verts_yz[:, 0]
        # z = mesh_verts_yz[:, 1]
        # ax.plot(y, z, "k.")

        # TODO: Why is 611731 not in full_path

        # Plot nearest neighbors
        for slice_i, NN_IDX in enumerate(closest_NN):

            # for mesh_i in NN_IDX:
            #     p1 = full_slice_yz[slice_i, :]
            #     p2 = mesh_verts_yz[mesh_i, :]

            #     x, y = zip(p1, p2)
            #     ax.plot(x, y, "g-")

            p1 = full_slice_yz[slice_i, :]
            p2 = mesh_verts_yz[NN_IDX, :]

            x, y = zip(p1, p2)
            ax.plot(x, y, "g-")

        # # Plot nearest neighbors
        # x, y = mesh_verts_yz[closest_NN].T
        # ax.plot(x.ravel(), y.ravel(), "r-")

        # # Plot slice points
        # x, y = full_slice_yz.T
        # ax.plot(x.ravel(), y.ravel(), "r-")

        # Plot shortest path
        x, y = mesh_verts_yz[full_path].T
        ax.plot(x, y, "r-")
        plt.show()

        # # Identify derivatives at points along parent mesh
        p_V = parent.vertices[closest_NN]
        p_T = parent.vertices[closest_NN] - parent.vertices[np.roll(closest_NN, -1)]

        # Replace 0 values of tangent with last nonzero value
        for i, row in enumerate(p_T):

            if np.all(row == np.array([0.0, 0.0, 0.0])):

                p_T[i] = p_T[i - 1]

        p_N = parent.vertex_normals[closest_NN]
        p_B = np.cross(p_N, p_T)
        p_B = p_B / np.linalg.norm(p_B, axis=1, keepdims=True)  # norm

        # Identify derivatives at points along child's full slice
        uuu = u
        c_V = child.surface(uuu, slice_dist).squeeze()
        c_T = child.surface.derivative(uuu, slice_dist, d=(0, 1)).squeeze()
        c_T = c_T / np.linalg.norm(c_T, axis=1, keepdims=True)

        # Plot mesh derivatives
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=-90, azim=90)

        x, y, z = child.verts[::100].T
        ax.plot3D(x, y, z, "k.")

        for i in range(c_V.shape[0]):

            p1 = c_V[i]
            p2 = p1 + c_T[i] * 0.1

            x, y, z = zip(p1, p2)
            ax.plot3D(x, y, z, "b-")

        # Plot parent
        x, y, z = parent.vertices[::100].T
        ax.plot3D(x, y, z, "r.")
        x, y, z = parent.vertices[full_path].T
        ax.plot3D(x, y, z, "g-")
        plt.show()

        # Create B-Spline Surface linking child slice and projection on parent

        # Inputs
        degree = ORDER - 1

        # Basis 1 - cross section
        # With >100 controlpoints, the curve essentially passes through the points, so when we go to switch this segment in, if we skip the first and last elements, I think it will work.
        num_cp_per_cross_section = c_V.shape[0]
        num_knots = num_cp_per_cross_section + ORDER + degree
        knot = np.linspace(0, 1, num_knots)
        basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)

        curve = Curve(basis1, c_V, rational=False)

        # Get points
        t = np.linspace(curve.start(), curve.end(), 100).ravel()
        x, y, z = curve(t).T
        cx, cy, cz = curve.controlpoints.T
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot3D(x, y, z, ".")
        ax.plot3D(cx, cy, cz, "rx")
        plt.show()

        # Basis 2 - along the major axis of the axial component
        num_rows = 4  # End termini + 2 intermediate points to determine slope
        knot = open_uniform_knot_vector(num_rows, ORDER)
        basis2 = BSplineBasis(order=ORDER, knots=knot, periodic=-1)

        # Controlpoints
        SCALE_FACTOR = 0.1
        cp = np.zeros([c_V.shape[0], 4, 3])
        cp[:, 0, :] = c_V
        cp[:, 1, :] = c_V - c_T * SCALE_FACTOR
        cp[:, 2, :] = p_V - p_B * SCALE_FACTOR
        cp[:, 3, :] = p_V
        cp = cp.reshape(num_rows * num_cp_per_cross_section, cp.shape[2], order="F")

        # Surface
        surface = Surface(basis1, basis2, cp, rational=False)
        self.surface = surface

        # Surface
        (us, vs) = surface.start()
        (ue, ve) = surface.end()
        uu = np.linspace(us, ue, 20)
        vv = np.linspace(vs, ve, 20)
        grid = surface(uu, vv)
        sx, sy, sz = grid.reshape(-1, surface.dimension).T
        cx, cy, cz = surface.controlpoints.T

        # Plot
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.view_init(elev=-90, azim=90)
        # ax.plot3D(r[:, 0], r[:, 1], r[:, 2], "k.")
        ax.plot3D(cx.ravel(), cy.ravel(), cz.ravel(), "g-")

        # Child
        x, y, z = child.verts.T
        ax.plot3D(x, y, z, "k.")

        # Parent
        x, y, z = parent.vertices.T
        ax.plot3D(x, y, z, "r.")

        # Surface
        ax.plot3D(sx, sy, sz, "b.")

        plt.show()

        # for i in range(len(full_path[:-1])):

        #     p1 = p_v[i]
        #     B = p_B[i]
        #     p2 = p1 + B

        #     x, y, z = zip(p1, p2)
        #     ax.plot3D(x, y, z, "b-")
        #     ax.plot3D(x[0], y[0], z[0], "k.")

        # Child derivatives

        # Add 2 sets of controlpoints to get these derivatives to work

        # Create new axial component

        # Delete the hole in the parent mesh

        # Stitch together the two

        pass

        # Alternative strategy

        # Figure out which faces were altered in the merge
        mesh0 = self.ac_list[0].mesh
        mesh1 = self.ac_list[1].mesh

        mesh2 = self.merged_meshes
        # Run a gaussian somothing along these edges (vertices along the edge are shifted to be the midpoint of the surrounding vertices within X distance, and this shift trails off for vertices further away from the edge)

    def plot_meshes(self):

        trimesh.Scene([ac.mesh for ac in self.ac_list]).show()

    def merge_meshes(self):

        merged_meshes = trimesh.boolean.union(
            [ac.mesh for ac in self.ac_list], engine="scad"
        )
        bf = trimesh.repair.broken_faces(merged_meshes)
        self.merged_meshes = merged_meshes
