from setuptools import setup, find_packages

setup(
    name="objects",
    packages=find_packages(),
    install_requires=[
        "scipy",
        "numpy",
        "trimesh",
        "splipy",
        "matplotlib",
        "open3d",
        "networkx",
        "pytest",
        "rtree",
        "black"

    ],
)
