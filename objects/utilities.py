# General functions used by different classes for objects project

import numpy as np

##########
# B-Spline Functions


def open_uniform_knot_vector(num_cps, order):

    num_knots = num_cps + order
    knots = np.zeros(num_knots)
    knots[order:-order] = range(1, num_knots - 2 * order + 1)
    knots[-order:] = knots[-order - 1] + 1
    knots = knots / knots.max()  # Scale from 0 to 1
    return knots
