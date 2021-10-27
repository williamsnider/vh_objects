import numpy as np
from objects.backbone import Backbone
from objects.parameters import NUM_CP_PER_SEGMENT, GOAL_LENGTH_SEGMENT
from scipy.optimize import minimize

# Calculate the optimal radius of the curved digit segments so that their lengths=1
GOAL_LENGTH_SEGMENT = 1


def arc_array(r, t, NUM_CP_PER_SEGMENT):
    """Constructs the controlpoint array corresponding to an arc based on the radius, t parameterization, and number of controlpoints"""
    arc_array = np.stack(
        [
            r * np.sin(t),
            r * (1 - np.cos(t)),
            r * np.zeros(NUM_CP_PER_SEGMENT),
        ]
    )
    return arc_array.T


def length_from_r(r, angle_of_circle):
    """A function that returns the squared distance between the length of the b-spline made from this segment and the GOAL_LENGTH_SEGMENT.

    r is the radius of the circle (the variable we will seek to optimize)
    angle_of_circle is the radians of the arc that this digit_segment forms. The largest such arc we want is 1/8 of a circle (angle_of_circle=np.pi/4) because 4 such segments oriented together could make a half circle when combined, which is the most rounded backbone we want."""

    t = np.linspace(0, angle_of_circle, NUM_CP_PER_SEGMENT)  # Choose this as 4 such segments could make a half circle
    cp_arc = arc_array(r, t, NUM_CP_PER_SEGMENT)
    backbone = Backbone(cp_arc, reparameterize=False)
    return (backbone.length() - 1) ** 2


### 1/8 circle
x0 = [0.1]
args = np.pi / 4
bounds = [[0.0001, 100 * GOAL_LENGTH_SEGMENT]]
result = minimize(length_from_r, x0=x0, args=args, bounds=bounds)
r_1_8 = result.x

### 1/16 circle
x0 = [0.2]
args = np.pi / 8
bounds = [[0.0001, 100 * GOAL_LENGTH_SEGMENT]]
result = minimize(length_from_r, x0=x0, args=args, bounds=bounds)
r_1_16 = result.x


####################
### Digit Segments

### Flat
cp_flat = np.array(
    [
        np.linspace(0, 1, NUM_CP_PER_SEGMENT),
        np.zeros(5),
        np.zeros(5),
    ]
).T
segment_flat = Backbone(cp_flat, reparameterize=False)

### 1/8 circle
r = r_1_8
t = np.linspace(0, np.pi / 4, NUM_CP_PER_SEGMENT)
cp_arc = arc_array(r, t, NUM_CP_PER_SEGMENT)
segment_arc_1_8 = Backbone(cp_arc, reparameterize=False)
assert np.isclose(segment_arc_1_8.length(), 1), "Arc segment not close to length 1."

### 1/16 circle
r = r_1_16
t = np.linspace(0, np.pi / 8, NUM_CP_PER_SEGMENT)
cp_arc = arc_array(r, t, NUM_CP_PER_SEGMENT)
segment_arc_1_16 = Backbone(cp_arc, reparameterize=False)
assert np.isclose(segment_arc_1_16.length(), 1), "Arc segment not close to length 1."
