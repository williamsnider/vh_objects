from objects.backbone import Backbone
from objects.cross_section import CrossSection
from objects.axial_component import AxialComponent
from objects.shape import Shape
import numpy as np

# Parameters
y_max = 20  # Backbone
cs_all_scale = [2/3, 4/3]
cs_single_scale = [1/4, 2]
c = np.cos
s = np.sin
round_cp = np.array(
    [
        [c(0 / 6 * 2 * np.pi), s(0 / 6 * 2 * np.pi)],
        [c(1 / 6 * 2 * np.pi), s(1 / 6 * 2 * np.pi)],
        [c(2 / 6 * 2 * np.pi), s(2 / 6 * 2 * np.pi)],
        [c(3 / 6 * 2 * np.pi), s(3 / 6 * 2 * np.pi)],
        [c(4 / 6 * 2 * np.pi), s(4 / 6 * 2 * np.pi)],
        [c(5 / 6 * 2 * np.pi), s(5 / 6 * 2 * np.pi)],
    ]
)
round_cp *= 10

count = 0

def rand_backbone():

    # Create controlpoints
    x = np.linspace(0,40, 5)

    # Choose random y value
    y_peak = np.random.rand(1)*y_max
    y_sub = np.linspace(0, y_peak, 3).ravel()
    y = np.zeros(5)
    y[:3] = y_sub
    y[3:] = y_sub[1::-1]

    z = np.zeros(5)

    backbone_cp = np.vstack([x,y,z]).T
    backbone = Backbone(controlpoints = backbone_cp, reparameterize=True)

    return backbone

def rand_cross_section(position):
    
    concave_cp = round_cp.copy()

    # Scale one cp
    num_cp = concave_cp.shape[0]
    cp_idx = np.random.randint(0, num_cp)
    concave_cp[cp_idx] *= np.random.uniform(cs_single_scale[0], cs_single_scale[1])

    # Scale all cps
    concave_cp *= np.random.uniform(cs_all_scale[0], cs_all_scale[1])

    cs = CrossSection(controlpoints=concave_cp, position=position)

    return cs

def rand_shape():
    global count

    backbone = rand_backbone()

    cs_list = []
    for position in [0.1, 0.5, 0.9]:
        cs_list.append(rand_cross_section(position=position))

    ac = AxialComponent(backbone=backbone, cross_sections=cs_list)
    
    s = Shape([ac], label='Test_{}'.format(count))
    count += 1
    return s