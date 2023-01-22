# Fit a B-Spline to a given radius
from re import A
import numpy as np
import matplotlib.pyplot as plt
from splipy import BSplineBasis, Curve
from scipy.optimize import minimize_scalar
from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
import trimesh


ORDER = 3



def fit_radius(target_radius, num_cp_per_cross_section):
    """Calculates the radius of controlpoints that will result in a B-spline with the target radius.
    
    B-splines do not pass through the controlpoints, so a larger controlpoint radius is needed to achieve a target B-spline radius."""

    def make_bspline_curve(cp_r):
        
        # Make controlpoints
        c = np.cos
        s = np.sin
        th = np.linspace(0, 2*np.pi, num_cp_per_cross_section, endpoint=False).reshape(-1,1)
        cp = np.hstack((cp_r*c(th), cp_r*s(th)))

        # Make curve
        degree = ORDER - 1
        num_knots = num_cp_per_cross_section + ORDER + degree
        knot = np.linspace(0, 1, num_knots)
        basis1 = BSplineBasis(order=ORDER, knots=knot, periodic=1)
        curve = Curve(basis1, controlpoints=cp, rational=False)

        # Sample curve
        curve.reparam()
        t = np.linspace(0, 1, num_cp_per_cross_section)
        xy = curve(t)

        return xy

    def objective_function(cp_r):

        # Make curve
        xy = make_bspline_curve(cp_r)

        # Calc distances
        dists = np.linalg.norm(xy, axis=1)
        avg_radius = dists.mean()

        return (avg_radius-target_radius)**2


    res = minimize_scalar(objective_function, method='bounded', bounds=(0, target_radius**2))
    if res.success == True:
        return res.x
    else:
        return None
# # Plot
# fig, axs = plt.subplots(1)
# axs.plot(xy[:,0], xy[:,1], "b.", linewidth=10)
# axs.set_aspect("equal")
# # plt.show()


# # Test if B-Spline is also fit when axial component


# # # Backbone
# # # Create controlpoints
# x = np.linspace(0,60, 5)
# y = np.zeros(5)
# z = np.zeros(5)
# backbone_cp = np.vstack([x,y,z]).T

# from objects.utilities import approximate_arc
# c = np.cos
# s = np.sin
# q = np.pi
# # backbone_cp = approximate_arc(q, 60)
# R = np.array([[c(q/2), s(q/2), 0],[-s(q/2), c(q/2), 0],[0,0,1]])
# backbone_R = backbone_cp @ R
# backbone = Backbone(controlpoints = backbone_R, reparameterize=True)

# # Cross section
# th = np.linspace(0, 2*np.pi, NUM_CP_PER_CROSS_SECTION, endpoint=False).reshape(-1,1)
# cp = np.hstack((res.x*c(th), res.x*s(th)))


# # base_cp_round = np.array(
# #     [
# #         [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
# #         [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
# #         [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
# #         [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
# #         [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
# #         [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
# #     ]
# # )


# cs_list = [CrossSection(controlpoints=cp, position=position) for position in np.linspace(0.2,1.0,20)]

# # Axial Component
# # ac = AxialComponent(backbone, cs_list)

# # Slice mesh, find distances
# # lines = trimesh.intersections.mesh_plane(ac.mesh, ac.T(0.5)[0], ac.r(0.5)[0])
# # pts = lines.reshape(-1,3)
# # dists = np.linalg.norm(pts[:,1:],axis=1)


# ### Post
# # cs_list_shape = [CrossSection(controlpoints=xy, position=position) for position in np.linspace(0.1,0.8,10)]
# # cs_list_post = [CrossSection(controlpoints=0.5*xy, position=position) for position in np.linspace(0.85,1,5)]
# # cs_list = [*cs_list_shape, *cs_list_post]

# ac = AxialComponent(backbone, cs_list, smooth_with_post=False)
# ac.mesh.show(smooth=False)



# ### Round endcaps
# def make_arc_array(a,r):
#     arc_array = np.array([[0, r], [a, r],[r,a],[r, 0]])
#     return arc_array

# from objects.utilities   import open_uniform_knot_vector
# def radius_error(a):

#     # Make the arc array
#     arc_array = make_arc_array(a, res.x)

#     # Make curve
#     knot = open_uniform_knot_vector(arc_array.shape[0], ORDER)
#     basis = BSplineBasis(order=ORDER, knots=knot, periodic=-1)
#     curve = Curve(basis=basis, controlpoints=arc_array, rational=False)

#     # Sample points along the backbone
#     t = np.linspace(0, 1, 100)
#     r = curve(t)

#     # distance from origin should be close to radius if points are well_aligned
#     dist = np.linalg.norm(r, axis=1)
#     return ((dist - res.x) ** 2).sum()

# out = minimize_scalar(radius_error, method='bounded', bounds=(0, TARGET_RADIUS**2))



# # # Sphere origin
# # ti = 1-res.x/40
# # origin = backbone.r(ti)

# # # Cross section
# # cs_list = [CrossSection(controlpoints=xy, position=position) for position in np.linspace(0.05,ti,10)]

# # # Axial Component
# # # ac = AxialComponent(backbone, cs_list)

# # # Slice mesh, find distances
# # # lines = trimesh.intersections.mesh_plane(ac.mesh, ac.T(0.5)[0], ac.r(0.5)[0])
# # # pts = lines.reshape(-1,3)
# # # dists = np.linalg.norm(pts[:,1:],axis=1)


# # ### Post
# # # cs_list_shape = [CrossSection(controlpoints=xy, position=position) for position in np.linspace(0.1,0.8,10)]
# # # cs_list_post = [CrossSection(controlpoints=0.5*xy, position=position) for position in np.linspace(0.85,1,5)]
# # # cs_list = [*cs_list_shape, *cs_list_post]

# # ac = AxialComponent(backbone, cs_list, smooth_with_post=True)
# # ac.mesh.show()