Order of things to run to recreate the stimulus set:

1. switch to git commit:
2. run medial_axis.py
3. run sheets.py
4. run torso.py
5. optional: run make_gif.py, check that eveerything looks OK
5. run duplicates_find.py, identify which of the above STLs are redundant
6. delete the non-duplicates from those CSV files
7. identify the shapes to apply texture to (do not choose duplicate shapes), run texture.py 
8. run make_gifs.py again for just the texture
7. run label_shapes.py to apply the filename to the base of the shape
6. run duplicates_remove.py, which copies all the files EXCEPT the duplicates
