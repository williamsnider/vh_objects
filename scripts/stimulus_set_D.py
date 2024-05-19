import trimesh
import numpy  as np


# Plot base cylinder
base_cylinder_radius = 20  # mm
base_cylinder_height = 10  # mm
base_cylinder = trimesh.creation.cylinder(radius=base_cylinder_radius, height=base_cylinder_height)

# Shift down to origin
base_cylinder.apply_translation([0, 0, -base_cylinder_height / 2])
scene = trimesh.Scene([base_cylinder])

# Create appendage capsule
capsule_radius = 5 # mm
capsule_height = base_cylinder_radius-capsule_radius # mm
capsule = trimesh.creation.capsule(radius=capsule_radius, height=capsule_height)

# Create various transformations
T_up = np.eye(4)

T_up45 = trimesh.transformations.rotation_matrix(angle = np.pi/4, direction = [1, 0, 0], point = [0, 0, 0])

T_up90 =trimesh.transformations.rotation_matrix(angle = np.pi/2, direction = [1, 0, 0], point = [0, 0, 0])


# capsule.apply_transform(T_up90)
# scene.add_geometry(capsule)


# Bend capsule into elbow shape
capsule_K = capsule.copy()

num_verts = len(capsule_K.vertices)
for i in range(num_verts):

    v = capsule_K.vertices[i]

    # Determine scale factor based on height of vertex
    scale = v[2] / capsule_height
    if scale<0:
        scale = 0
    elif scale>1:
        scale = 1

    # Determine angle of rotation
    angle = np.pi/2*scale

    # Rotate vertex around x-axis
    T = trimesh.transformations.rotation_matrix(angle = angle, direction = [1, 0, 0], point = [0, 0, 0])
    v = np.append(v, 1)
    v = np.dot(T, v)
    capsule_K.vertices[i] = v[:3]

scene.add_geometry(capsule_K)

    

# Plot axes
axes = trimesh.creation.axis(origin_size=10, axis_radius=0.5, axis_length=50)
scene.show()
