# Tile images and output a new image
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


NUM_COL = 4
NUM_ROW = 4
PADDING = 10  # pixels

base_dir = "/home/williamsnider/Code/objects/sample_shapes/stimulus_set_C"
png_dir = Path(base_dir, "png")
group_dir = Path(base_dir, "group")
if group_dir.exists() == False:
    group_dir.mkdir(parents=True)


# Read directory
png_list = [str(p) for p in Path.glob(png_dir, "*")]
png_list.sort(key=natural_sort_key)

# Get size of images
idx = 0
test_img = np.asarray(Image.open(png_list[0]))
h, w, d = test_img.shape

# Group into sets of NUM_COL * NUM_ROW
num_fill = NUM_COL * NUM_ROW - len(png_list) % (NUM_COL * NUM_ROW)
png_groups = np.concatenate([png_list, ["" for _ in range(num_fill)]]).astype("object")
png_groups = png_groups.reshape(-1, NUM_COL * NUM_ROW)


for group_num, group in enumerate(png_groups):
    count = 0
    # Make combined image
    big_img = 255 * np.ones(
        (h * NUM_ROW + (NUM_ROW - 1) * PADDING, w * NUM_COL + (NUM_COL - 1) * PADDING, d), dtype=test_img.dtype
    )
    for r in range(NUM_ROW):
        for c in range(NUM_COL):

            # Load image
            filename = group[count]
            if filename == "":
                continue
            img = Image.open(filename)

            # Draw text
            text = Path(filename).stem[5:]
            d1 = ImageDraw.Draw(img)
            font = ImageFont.truetype("/home/williamsnider/Code/objects/assets/consolab.ttf", 150)
            d1.text((28, 36), text, fill=(255, 255, 255), font=font)
            out = np.asarray(img)

            x = r * (h + PADDING)
            y = c * (w + PADDING)
            big_img[x : x + h, y : y + w] = out

            count += 1
    # Save result
    im = Image.fromarray(big_img)
    im.save(Path(group_dir, "big_" + str(group_num) + ".png"))
