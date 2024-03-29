# Make shaft given length, r1, r2, r3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from objects.utilities import make_surface, make_mesh
from objects.backbone import Backbone


def plot_profile(xx, yy, x, y, lxx, lyy, rxx, ryy):
    ax = plt.figure().add_subplot()
    ax.plot(xx, yy, "-b")
    ax.plot(x, y, "r*")
    ax.plot(lxx, lyy, "-g")
    ax.plot(rxx, ryy, "-g")
    ax.set_aspect("equal")
    plt.show()


def piecewise_func(x, lr, la_s, quad_s, spacing, rr, ra_s, ldist):

    # Left hemisphere
    if x < ldist:
        th = np.arccos(np.round((x - la_s) / lr, 8))
        assert np.isclose(x, lr * np.cos(th) + la_s)
        return np.sin(th) * lr

    # Center quadratic region
    elif ldist <= x <= ldist + 2 * spacing:
        return np.polyval(quad_s, x)

    # Right hemisphere
    else:

        th = np.arccos(np.round((x - ra_s) / rr, 8))  # Round to avoid >1
        assert np.isclose(x, rr * np.cos(th) + ra_s)
        return np.sin(th) * rr


def calc_profile_hemi_hemi(spacing, r1, r2, r3, lengthtype="", plot=False):

    assert lengthtype in ["one_hemi", "two_hemi"]

    # Fit quadratic
    x = np.arange(3) * spacing
    y = np.array([r1, r2, r3])
    quad = np.polyfit(x, y, 2)

    # Fit circular arc - leftside
    lx = 0
    ly = np.polyval(quad, lx)
    lder = np.polyder(quad, 1)
    lm = np.polyval(lder, lx)
    la = lx + ly * lm  # Solve for a
    lr = np.sqrt((lx - la) ** 2 + ly**2)  # Solve for r
    lth = np.arctan2(ly, (lx - la))

    # Fit circular arc - rightside
    rx = 2 * spacing
    ry = np.polyval(quad, rx)
    rder = np.polyder(quad, 1)
    rm = np.polyval(rder, rx)
    ra = rx + ry * rm  # Solve for a
    rr = np.sqrt((rx - ra) ** 2 + ry**2)  # Solve for r
    rth = np.arctan2(ry, (rx - ra))

    # Calculate length
    if lengthtype == "two_hemi":
        length = lr * (np.cos(0) - np.cos(np.pi - lth)) + 2 * spacing + rr * (np.cos(0) - np.cos(rth))
    elif lengthtype == "one_hemi":
        length = 2 * spacing + rr * (np.cos(0) - np.cos(rth))

    # Sample and graph
    xx = np.linspace(0, 2 * spacing)
    yy = np.polyval(quad, xx)

    ltt = np.linspace(np.pi, lth)
    lxx = lr * np.cos(ltt) + la
    lyy = lr * np.sin(ltt)

    rtt = np.linspace(rth, 0)
    rxx = rr * np.cos(rtt) + ra
    ryy = rr * np.sin(rtt)

    if plot == True:
        plot_profile(xx - lxx[0], yy, x - lxx[0], y, lxx - lxx[0], lyy, rxx - lxx[0], ryy)
    # assert np.isclose(length, rxx[-1] - lxx[0])

    # Shift everything so that shapes starts at x=0
    if lengthtype == "two_hemi":
        ldist = np.abs(lxx[0] - lxx[-1])
        la_s = la + ldist
        quad_s = np.polyfit(x + ldist, y, 2)
        ra_s = ra + ldist
        assert np.isclose(la_s, lr)

    # Do not shift since we only care about far hemi
    elif lengthtype == "one_hemi":
        ldist = 0
        la_s = la
        quad_s = quad
        ra_s = ra

    return length, lr, la_s, lth, quad_s, spacing, rr, ra_s, rth, ldist


def optimize_spacing(*inputs):
    spacing, r1, r2, r3, GOAL_LENGTH, lengthtype = inputs

    length, _, _, _, _, _, _, _, _, _ = calc_profile_hemi_hemi(spacing, r1, r2, r3, lengthtype, plot=False)
    return (length - GOAL_LENGTH) ** 2


class Shaft:
    def __init__(self, length, r1, r2, r3, theta, lengthtype, num_cs, num_cp_per_cs):
        self.length = length
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.theta = theta
        self.lengthtype = lengthtype
        self.num_cs = num_cs
        self.num_cp_per_cs = num_cp_per_cs

        success = self.calc_optimal_spacing()
        if success == False:
            print("Failed to create shaft")
            return
        self.make_shaft()
        self.calc_sphere_origins()

    def calc_optimal_spacing(self):

        # Optimize distance between radii given goal_length
        x0 = self.length / 2
        res = minimize(
            optimize_spacing,
            x0=x0,
            args=(self.r1, self.r2, self.r3, self.length, self.lengthtype),
            bounds=[(0.0001, self.length)],
        )
        if res.fun > 1e-10:
            print("Failed to find an optimal spacing, the radii are probably too large for the given length.")
            self.spacing = None
            return False
        else:
            self.spacing = res.x
            return True

    def make_shape(self):

        x = self.x
        y = self.y

        # Adjust second and second-to-last x to be zero (ensures non-sharp ending)
        x[1] = x[0]
        x[-2] = x[-1]

        # Base cross section
        th = np.linspace(0, 2 * np.pi, self.num_cp_per_cs, endpoint=False).reshape(-1, 1)
        base_cs = np.hstack([np.zeros(th.shape), np.cos(th), np.sin(th)])

        if self.theta != 0:

            # Calculate l portion
            NUM_L_CP = 5
            l_cp = np.hstack(
                [
                    np.linspace(0, self.ldist, NUM_L_CP).reshape(-1, 1),
                    np.zeros((NUM_L_CP, 1)),
                    np.zeros((NUM_L_CP, 1)),
                ]
            )

            # Calculate arc
            arc_length = self.length - self.ldist - self.rdist
            radius = arc_length / self.theta
            t = np.linspace(3 * np.pi / 2, 3 * np.pi / 2 + self.theta, 100).reshape(-1, 1)
            arc_cp = np.hstack([radius * np.cos(t), radius * np.sin(t), np.zeros(t.shape)])  # Sample arc
            arc_cp -= arc_cp[0]  # Shift start to origin
            arc_cp += l_cp[-1]  # Shift start to end of l_cp

            # Calculate r portion
            T_vec = np.array([-np.sin(t[-1][0]), np.cos(t[-1][0]), 0]).reshape(1, -1)  # vector tangent at end of arc
            r_cp = (
                T_vec * np.linspace(0, self.rdist, NUM_L_CP).reshape(-1, 1) + arc_cp[-1]
            )  # Shift start to end of arc_cp

            # Combine
            b_cp = np.vstack([l_cp[:-1], arc_cp, r_cp[1:]])

        else:
            T_vec = np.array([1, 0, 0])
            NUM_L_CP = 5
            b_cp = np.hstack(
                [
                    np.linspace(0, self.length, NUM_L_CP).reshape(-1, 1),
                    np.zeros((NUM_L_CP, 1)),
                    np.zeros((NUM_L_CP, 1)),
                ]
            )

        self.backbone = Backbone(b_cp, reparameterize=True)

        # Shift points according to backbone
        cp = np.zeros((len(x), self.num_cp_per_cs, 3))
        for i in range(len(x)):

            new_cs = base_cs.copy()

            # Scale according to y
            new_cs *= y[i]

            # Shift according to x
            pos = np.round(x[i] / x[-1], 8)
            T = np.eye(4)
            T[:3, 0] = self.backbone.T(pos).reshape(-1)
            T[:3, 1] = self.backbone.N(pos).reshape(-1)
            T[:3, 2] = self.backbone.B(pos).reshape(-1)
            T[:3, 3] = self.backbone.r(pos).reshape(-1)

            # Homogenous coordinates
            homo_cs = np.hstack([new_cs, np.ones((new_cs.shape[0], 1))])

            # Transform
            T_cs = homo_cs @ T.T

            # Populate
            cp[i] = T_cs[:, :3]

        # cp[:, :, 0] = x.reshape(-1, 1)

        surf = make_surface(cp)
        mesh = make_mesh(surf, 100, 100)

        return mesh, cp

    def make_shaft(self):

        # Calculate profile, which will be used to scale the controlpoints
        _, lr, la_s, lth, quad_s, _, rr, ra_s, rth, ldist = calc_profile_hemi_hemi(
            self.spacing, self.r1, self.r2, self.r3, self.lengthtype, plot=False
        )
        self.l_sphere_radius = lr
        self.r_sphere_radius = rr

        # Construct piecewise function (left: arc, middle: quadratic, right: arc)
        l_thetas = np.linspace(np.pi, lth, self.num_cs, endpoint=False)
        lx = lr * np.cos(l_thetas) + la_s
        qx = np.linspace(ldist, ldist + 2 * self.spacing, self.num_cs)
        r_thetas = np.linspace(0, rth, self.num_cs, endpoint=False)[::-1]
        rx = rr * np.cos(r_thetas) + ra_s
        x = np.concatenate([lx.ravel(), qx.ravel(), rx.ravel()])
        y = np.zeros(len(x))

        for i in range(len(x)):
            y[i] = piecewise_func(x[i], lr, la_s, quad_s, self.spacing, rr, ra_s, ldist)

        self.x = x
        self.y = y
        self.ldist = ldist
        self.rdist = self.length - 2 * self.spacing[0] - ldist

        mesh, cp = self.make_shape()

        self.cp = cp
        self.mesh = mesh

    def calc_sphere_origins(self):
        self.l_sphere_origin = (self.backbone.r(0.0) + self.backbone.T(0.0) * self.l_sphere_radius).ravel()
        self.r_sphere_origin = (self.backbone.r(1.0) + -self.backbone.T(1.0) * self.r_sphere_radius).ravel()

    def apply_transform(self, T):
        self.mesh = self.mesh.apply_transform(T)
        self.cp = (np.hstack([self.cp, np.ones(len(self.cp))]) @ T)[:, :3]
        self.l_sphere_origin = (np.hstack([self.l_sphere_origin, np.ones(len(self.l_sphere_origin))]) @ T)[:, :3]
        self.r_sphere_origin = (np.hstack([self.r_sphere_origin, np.ones(len(self.r_sphere_origin))]) @ T)[:, :3]


if __name__ == "__main__":

    shaft = Shaft(25, 2, 4, 2, np.pi, "two_hemi", 11, 50)
    shaft.mesh.visual.vertex_colors = np.array([255, 255, 0, 50])
    import trimesh

    l = trimesh.primitives.creation.icosphere()
    l.apply_translation(shaft.l_sphere_origin)
    r = trimesh.primitives.creation.icosphere()
    r.apply_translation(shaft.r_sphere_origin)

    scene = trimesh.Scene()
    scene.add_geometry([shaft.mesh, l, r])
    scene.show()

    # shaft.mesh.show()
