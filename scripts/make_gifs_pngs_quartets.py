# This is a master script to generate images of a nested directory containing .STL files. This script:


from pathlib import Path
import trimesh
import numpy as np
from scripts.rotate_stl_to_align_with_quartet import rotate_stls_to_align_notch
from scripts.make_gif import make_gifs_of_stls
from scripts.make_rotated_pngs import make_rotated_pngs
from scripts.make_quartet_page import make_quartet_page

##############
### Inputs ###
##############
old_stl_dir = Path("/home/oconnorlab/Downloads/test/stl")
new_stl_dir = Path("/home/oconnorlab/Downloads/test2/stl_rotated_correctly/stl")
quartet_arrangement_fname = Path("/home/oconnorlab/Code/vh_objects/scripts/2025-04-07_quartets.csv")

if __name__ == "__main__":

    # Ensure that STLs are rotated correctly (the square notch is aligned)
    rotate_stls_to_align_notch(old_stl_dir, new_stl_dir)

    # Render GIFs of the STLs
    make_gifs_of_stls(new_stl_dir)

    # Make PNGs in two rotations (base and 90 degrees about Z)
    make_rotated_pngs(new_stl_dir, resolution=750)

    # Make a page with the quartet arrangements
    combined_png_dir = new_stl_dir.parent / "png_AB"
    make_quartet_page(combined_png_dir, quartet_arrangement_fname=quartet_arrangement_fname)
