# Roates the STLs in a given directory such that they align with their actual placement in a quartet. Specifically, during the original creation of the quartets, there was some discrepancy in how the interface portion was rotated based on the different script used. When assembling the quartets, I made sure that in all cases, the small square mark on each shape is in the up position. For ease of future analysis, this script will generate a new stl that has been rotated, and will check that the small square mark is in the right position.
from pathlib import Path


# Inputs
old_stl_dir = Path('/Users/williamsnider/Library/CloudStorage/GoogleDrive-williamgraysonsnider@gmail.com/My Drive/OConnor/Stimuli/stimulus_set_G/sample_shapes_no_duplicates/stl')
new_stl_dir = Path("/Users/williamsnider/Code/vh_objects/scripts/stl_rotated_properly")

# Read old STL files
old_stl_files = list(old_stl_dir.rglob("*.stl"))

# Define where the small square should be, as well as a test of its location

# Rotate the stl if necessary

# Save in a new directory