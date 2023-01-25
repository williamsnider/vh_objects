# Find bounding box of all images in a set
# Choose largest box
# Apply this crop to all images


import cv2
import numpy as np
import os
from pathlib import Path


def get_bbox(filename):
    # read image
    img = cv2.imread(filename)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

    # get contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return x, y, w, h


padding = 25  # pixels

image_dir = "/home/oconnorlab/code/objects/sample_shapes/stimulus_set_A/png"
crop_dir = "/home/oconnorlab/code/objects/sample_shapes/stimulus_set_A/png_cropped"
f_list = os.listdir(image_dir)

corners = np.zeros((len(f_list), 4), dtype="int")
for i, f in enumerate(f_list):
    filename = image_dir + "/" + f
    x, y, w, h = get_bbox(filename)

    # Convert to corners
    minx = x
    miny = y
    maxx = x + w
    maxy = y + h

    corners[i] = minx, miny, maxx, maxy

# Extreme values
xmin = corners[:, 0].min() - padding
ymin = corners[:, 1].min() - padding
xmax = corners[:, 2].max() + padding
ymax = corners[:, 3].max() + padding

# Handle values out of range
if xmin < 0:
    xmin = 0
if ymin < 0:
    ymin = 0
if xmax > 1919:
    xmax = 1919
if ymax > 1199:
    ymax = 1199

# Crop images
for f in f_list:

    # Load image
    filename = image_dir + "/" + f
    img = cv2.imread(filename)

    if Path(crop_dir).exists() == False:
        Path(crop_dir).mkdir(parents=True)
    # Crop
    img_cropped = img[ymin:ymax, xmin:xmax, :]
    filename_cropped = crop_dir + "/" + f
    cv2.imwrite(filename_cropped, img_cropped)

    print("Wrote", filename)
