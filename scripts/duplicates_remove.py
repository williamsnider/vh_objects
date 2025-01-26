from pathlib import Path
import pandas as pd
import shutil

# Load duplicates, set threshold, copy directory


csv_list = [
    Path("/home/williamsnider/Code/vh_objects/scripts/duplicate_pairs_medial_axis.csv"),
    Path("/home/williamsnider/Code/vh_objects/scripts/duplicate_pairs_sheet.csv"),
]


# Assume the non-duplicates were removed from the CSVs
unique_list = []
duplicate_list = []
for csv in csv_list:
    df = pd.read_csv(csv)
    duplicate_list.extend(df["Mesh2"].values)
    unique_list.extend(df["Mesh1"].values)

# Paranoia - group all duplicates into subgroups of the same time, then choose one from each subgroup
duplicate_groups = []

for i in range(len(unique_list)):

    mesh1 = unique_list[i]
    mesh2 = duplicate_list[i]

    for j in range(len(duplicate_groups)):

        if (mesh1) in duplicate_groups[j] or (mesh2) in duplicate_groups[j]:
            duplicate_groups[j].add(mesh1)
            duplicate_groups[j].add(mesh2)
            break
    else:
        duplicate_groups.append({mesh1, mesh2})

# Choose one from each group (lowest alphebitcally)
keep_list = []
for i in range(len(duplicate_groups)):
    group = list(duplicate_groups[i])
    group.sort()
    keep_list.append(group[0])
keep_list.sort()

duplicates_less_keep = [d for d in duplicate_list if d not in keep_list]

# Sanity check
for d in duplicates_less_keep:
    assert d not in keep_list, f"Duplicate {d} is in keep_list"
    assert d in duplicate_list, f"Duplicate {d} is not in duplicate_list"

# Proving that the above was unnecessary
assert set(duplicates_less_keep) == set(duplicate_list)

duplicates_set = set(duplicate_list)
# duplicate_list = list(set(duplicate_list))
# unique_list = list(set(unique_list))

# # Overlapping
# overlapping = [d for d in duplicate_list if d in unique_list]

old_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes")
new_dir = Path("/home/williamsnider/Code/vh_objects/sample_shapes_no_duplicates")
new_dir.mkdir(exist_ok=True, parents=True)

stl_list = list(old_dir.rglob("*.stl"))
for i, stl in enumerate(stl_list):
    print(f"Processing {i+1}/{len(stl_list)}: {stl}")

    if stl.name not in duplicates_less_keep:

        # Copy STL
        new_stl_path = Path(new_dir, stl.relative_to(old_dir))
        new_stl_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(stl, new_stl_path)

        # Copy PNG
        old_png_path = Path(str(stl).replace("stl", "png"))
        new_png_path = Path(new_dir, old_png_path.relative_to(old_dir))
        new_png_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(old_png_path, Path(new_png_path))

        # Copy gif
        old_gif_path = Path(str(stl).replace("stl", "gif"))
        new_gif_path = Path(new_dir, old_gif_path.relative_to(old_dir))
        new_gif_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(old_gif_path, Path(new_gif_path))

    else:
        duplicates_set.remove(stl.name)


assert len(duplicates_set) == 0, "Not all duplicates were removed. {}".format(duplicates_set)
