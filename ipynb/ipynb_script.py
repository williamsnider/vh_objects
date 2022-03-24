# Generate a .py file that will be transformed into a .ipynb for easier sharing on colab

# Step 1: Run this file to produce stimulus_set_outpuy.py
# Step 2: On command line run "p2j stimulus_set_output.py" to generate stimulus_set.ipynb
# Step 3: Run all in stimulus_set.ipynb to show rendered shapes
# Step 4: Push to git
# Step 5: Copy git link to nbviewer.com


import pickle
from pathlib import Path


pickle_path = Path(
    r"C:\Users\William\Files\OConnor\Code\Projects\objects\stimulus_set\Z\stimulus_set_ 24_03_22__12_08_50.pickle"
)
with open(pickle_path, "rb") as f:
    shapes = pickle.load(f)

filename = "stimulus_set_output_0.py"
txt = """
import trimesh
import pickle
from pathlib import Path

pickle_path = Path(r"{pickle_path}")
with open(pickle_path, "rb") as f:
    shapes = pickle.load(f)

# Shapes Rendered

""".format(
    pickle_path=pickle_path
)
for i in range(1, len(shapes)):

    code = "shapes[{}].mesh.show()".format(i)
    comment = "# Above shape is: {label}\n# {description}".format(
        label=shapes[i].label, description=shapes[i].description
    )
    code = "shapes[{}].mesh_with_interface.show()".format(i)
    txt += code + "\n\n" + comment + "\n\n"

with open(filename, "w") as f:
    f.write(txt)
