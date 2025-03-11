from setuptools import setup, find_packages

setup(
    name="vh_objects",
    packages=find_packages(),
    install_requires=[
        "splipy",
        "triangle",
        "pyglet<2",
        # "pyrender @ git+https://github.com/williamsnider/pyrender-fork.git",
        "pyrender",
        "numpngw @ git+https://github.com/WarrenWeckesser/numpngw.git",
        "opencv-python",
        "manifold3d==2.3.0",
        "tqdm",
    ],
)
