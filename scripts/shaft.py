# Make shaft given length, r1, r2, r3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from objects.utilities import make_surface, make_mesh


class Shaft:
    def __init__(self):
        pass


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


def make_shape(x, y, NUM_CP_PER_CS):

    # Adjust second and second-to-last x to be zero (ensures non-sharp ending)
    x[1] = x[0]
    x[-2] = x[-1]

    # Base cross section
    th = np.linspace(0, 2 * np.pi, NUM_CP_PER_CS, endpoint=False).reshape(-1, 1)
    base_cs = np.hstack([np.ones(th.shape), np.cos(th), np.sin(th)])

    # Tile
    cp = np.tile(base_cs, (len(x), 1, 1))

    # Scale according to y_axis
    cp[:, :, 1:] *= y.reshape(-1, 1, 1)

    # Expand along x-axis (long axis of axial component)
    cp[:, :, 0] = x.reshape(-1, 1)

    surf = make_surface(cp)
    mesh = make_mesh(surf, 100, 100)
    return mesh, cp


def make_shaft(goal_length, r1, r2, r3, lengthtype, NUM_CS, NUM_CP_PER_CS):

    x0 = goal_length / 2

    res = minimize(
        optimize_spacing,
        x0=x0,
        args=(r1, r2, r3, goal_length, lengthtype),
        bounds=[(0.0001, goal_length)],
    )
    if res.fun > 1e-10:
        print("Failed to find an optimal spacing, the radii are probably too large for the given length.")
        return None
    else:

        spacing = res.x
        print("Spacing: ", spacing)
        length, lr, la_s, lth, quad_s, spacing, rr, ra_s, rth, ldist = calc_profile_hemi_hemi(
            spacing, r1, r2, r3, lengthtype, plot=False
        )

        # Determine optimal number of controlpoints

        # Sample piecewise func
        # Construct piecewise polynomial
        l_thetas = np.linspace(np.pi, lth, NUM_CS, endpoint=False)
        lx = lr * np.cos(l_thetas) + la_s
        qx = np.linspace(ldist, ldist + 2 * spacing, NUM_CS)
        r_thetas = np.linspace(0, rth, NUM_CS, endpoint=False)[::-1]
        rx = rr * np.cos(r_thetas) + ra_s
        x = np.concatenate([lx.ravel(), qx.ravel(), rx.ravel()])
        y = np.zeros(len(x))

        for i in range(len(x)):
            y[i] = piecewise_func(x[i], lr, la_s, quad_s, spacing, rr, ra_s, ldist)

        mesh, cp = make_shape(x, y, NUM_CP_PER_CS)

        s = Shaft()
        s.mesh = mesh
        s.cp = cp
        return s


if __name__ == "__main__":

    shaft = make_shaft(25, 5, 10, 5, "one_hemi", 11, 50)
    shaft.mesh.show()
