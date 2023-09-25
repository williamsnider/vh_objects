from random_shapes import rand_shape
import numpy as np


for i in range(200):
    np.random.seed(i)
    s = rand_shape()
    s.save_mesh_as_png("./samples/")
