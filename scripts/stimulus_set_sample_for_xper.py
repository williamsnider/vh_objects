from random_shapes import rand_shape


for i in range(20):
    s = rand_shape()
    s.save_mesh_as_png('./samples/')