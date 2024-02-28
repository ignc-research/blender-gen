#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import random
import json
import sys
import os

sys.path.append(os.getcwd())
import config

cfg = config.cfg()

with open(f"DATASET/{cfg.out_folder}/annotations/instances_default.json") as f:
    data = json.load(f)

images = data["images"]
labels = data["annotations"]
for idx in range(len(images)):
    img_name = images[idx]["file_name"]
    print("showing annotation for img: " + img_name)
    print("close the window to see the next label")

    bbox = labels[idx]["bbox"]

    I = mpimg.imread(
        f"DATASET/{cfg.out_folder}/images/" + img_name
    )  # load rendered image

    fig, ax = plt.subplots(1)
    ax.imshow(I)
    plt.axis("off")
    [x, y, w, h] = bbox
    rect = patches.Rectangle(
        (x, y), w, h, linewidth=2, edgecolor="g", facecolor="none"
    )  # add bounding box annotation
    ax.add_patch(rect)

    plt.show()
