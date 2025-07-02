from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Must come before pyplot import
import matplotlib.pyplot as plt


def make_quartet_page(new_stl_dir, quartet_arrangement_fname):
    # Load fnames of combined
    combined_png_dir = new_stl_dir.parent / "png_AB"
    combined_png_files = list(combined_png_dir.rglob("*.png"))
    shape_id_to_path_dict = {c.name.split("_")[0]: c for c in combined_png_files}

    # Confirm they are all the same dimensions
    test_file = combined_png_files[0]
    img = cv2.imread(test_file, cv2.IMREAD_COLOR)
    img_width = img.shape[1]
    img_height = img.shape[0]

    for i in range(len(combined_png_files)):
        img = cv2.imread(combined_png_files[i], cv2.IMREAD_COLOR)
        if img.shape[1] != img_width or img.shape[0] != img_height:
            print(
                f"Image {combined_png_files[i].name} has different dimensions: {img.width}x{img.height} instead of {img_width}x{img_height}"
            )
            exit()
    print("All images have the same dimensions.")

    # Read quartet orders from a text file
    quartet_df = pd.read_csv(quartet_arrangement_fname, index_col=0)

    # Make
    num_quartets = len(quartet_df)
    num_rows = num_quartets + 1
    num_letters = 4
    num_cols = 5
    quartet_page = np.ones((num_rows * img_height, img_width * num_cols, 3), dtype=np.uint8) * 255  # White background

    # Add labels for columns

    for quartet_num, quartet_id in enumerate(quartet_df.index):
        for letter_num, letter in enumerate(["A", "B", "C", "D"]):

            row_num = quartet_num + 1  # Shift 1 to allow for header row
            col_num = letter_num + 1  # Shift 1 to allow quartet_id text

            shape_id = quartet_df.loc[quartet_id, letter]
            shape_fname = shape_id_to_path_dict.get(shape_id)

            # Load image, handling incomplete final quartet
            if shape_fname == None:
                shape_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # White square for missing images
            else:
                shape_img = cv2.imread(shape_fname, cv2.IMREAD_COLOR)

            # Calculate position in the grid
            row_start = row_num * img_height
            col_start = col_num * img_width

            # Place the image in the correct position
            quartet_page[row_start : row_start + img_height, col_start : col_start + img_width] = np.array(shape_img)

            print("Finished: ", quartet_id, letter)

    DPI = 1200
    fig, ax = plt.subplots(figsize=(quartet_page.shape[1] // DPI, quartet_page.shape[0] // DPI))

    ax.imshow(quartet_page)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Add text
    for quartet_num, quartet_id in enumerate(quartet_df.index):
        # Add quartet ID text
        ax.text(
            0.5 * img_width,
            (quartet_num + 2) * img_height - img_height / 2,
            quartet_id,
            fontsize=30,
            ha="center",
            va="center",
            color="black",
        )

    # Add column labels
    for letter_num, letter in enumerate(["A", "B", "C", "D"]):
        ax.text(
            (letter_num + 2) * img_width - img_width / 2,
            img_height / 2,
            letter,
            fontsize=30,
            ha="center",
            va="center",
            color="black",
        )

    pdf_fname = new_stl_dir.parent / quartet_arrangement_fname.name.replace(".csv", ".pdf")
    fig.savefig(pdf_fname, bbox_inches="tight", dpi=300)


if __name__ == "__main__":

    combined_png_dir = Path("/home/oconnorlab/Downloads/stl_rotated_correctly/png_AB")
    quartet_arrangement_fname = Path("/home/oconnorlab/Code/vh_objects/scripts/2025-04-07_quartets.csv")
    make_quartet_page(combined_png_dir, quartet_arrangement_fname=quartet_arrangement_fname)
    print("Quartet page created successfully.")
