from setuptools import setup, find_packages

setup(
    name="objects",
    packages=find_packages(),
    install_requires=[
        "splipy",
        "triangle",
        "pyglet",
        "pyrender @ git+https://github.com/williamsnider/pyrender-fork.git",
        "numpngw @ git+https://github.com/WarrenWeckesser/numpngw.git",
    ],
)
