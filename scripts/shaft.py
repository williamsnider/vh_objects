# Make shaft given length, r1, r2, r3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def plot_profile(xx, yy, x, y, lxx, lyy, rxx, ryy):
    ax = plt.figure().add_subplot()
    ax.plot(xx, yy, "-b")
    ax.plot(x, y, "r*")
    ax.plot(lxx, lyy, "-g")
    ax.plot(rxx, ryy, "-g")
    ax.set_aspect("equal")
    plt.show()


def piecewise_func(x, lr, la, quad, spacing, rr, ra):

    if x < 0:
        th = np.arccos((x - la) / lr)
        assert np.isclose(x, lr * np.cos(th) + la)
        return np.sin(th) * lr
    elif 0 <= x <= 2 * spacing:
        return np.polyval(quad, x)
    else:
        th = np.arccos((x - ra) / rr)
        assert np.isclose(x, rr * np.cos(th) + ra)
        return np.sin(th) * rr


def calc_profile_hemi_hemi(spacing, r1, r2, r3, plot=False):
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
    length = (
        lr * (np.cos(0) - np.cos(np.pi - lth))
        + 2 * spacing
        + rr * (np.cos(0) - np.cos(rth))
    )

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
        plot_profile(
            xx - lxx[0], yy, x - lxx[0], y, lxx - lxx[0], lyy, rxx - lxx[0], ryy
        )
    assert np.isclose(length, rxx[-1] - lxx[0])

    # Shift everything so that shapes starts at x=0
    breakpoints = np.cumsum([0, lxx[-1], spacing[0], spacing[0], rxx[-1] - rxx[0]])
    print(breakpoints)
    return length, lr, la, quad, spacing, rr, ra


def optimize_spacing(*inputs):
    spacing, r1, r2, r3, GOAL_LENGTH = inputs

    length, _, _, _, _, _, _ = calc_profile_hemi_hemi(spacing, r1, r2, r3, plot=False)
    return (length - GOAL_LENGTH) ** 2


if __name__ == "__main__":

    # Inputs
    r1, r2, r3 = np.array([5, 5, 5])
    GOAL_LENGTH = 20
    x0 = GOAL_LENGTH / 2

    res = minimize(
        optimize_spacing,
        x0=x0,
        args=(r1, r2, r3, GOAL_LENGTH),
        bounds=[(0.0001, GOAL_LENGTH)],
    )

    if res.fun > 1e-10:
        print(
            "Failed to find an optimal spacing, the radii are probably too small for the given length."
        )
    else:

        spacing = res.x
        print("Spacing: ", spacing)
        length, lr, la, quad, spacing, rr, ra = calc_profile_hemi_hemi(
            spacing, r1, r2, r3, plot=True
        )

        # Sample piecewise func
        # Construct piecewise polynomial
        x = np.linspace(-1, length - 1)
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = piecewise_func(x[i], lr, la, quad, spacing, rr, ra)

        ax = plt.figure().add_subplot()
        ax.plot(x, y, "r-*")
        ax.set_aspect("equal")
        plt.show()
        print("Done")
