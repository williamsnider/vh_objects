# Generate a .py file that will be transformed into a .ipynb for easier sharing on colab
import pickle
from pathlib import Path

pickle_path = Path(
    r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set\Z\stimulus_set_ 23_03_22__16_04_43.pickle"
)
with open(pickle_path, "rb") as f:
    shapes = pickle.load(f)

filename = "generated.py"
txt = """
import trimesh
import pickle

"""
for i in range(len(shapes)):

    comment = "# shapes[{}].label)".format(i)
    code = "shapes[{}].mesh.show()".format(i)
    txt += comment + "\n" + code + "\n"

with open(filename, "w") as f:
    f.write(txt)
