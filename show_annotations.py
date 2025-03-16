#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

sys.path.append(os.getcwd())
import config

# Constants
DATASET_DIR = "DATASET"
ANNOTATIONS_FILE = "instances_default.json"
IMAGES_DIR = "images"


def load_config() -> Any:
    """Load the configuration."""
    return config.cfg()


def load_annotations(cfg: Any) -> Dict[str, Any]:
    """Load annotations from the specified configuration."""
    annotations_path = os.path.join(
        DATASET_DIR, cfg.out_folder, "annotations", ANNOTATIONS_FILE
    )
    try:
        with open(annotations_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {annotations_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {annotations_path}.")
        sys.exit(1)


class AnnotationViewer:
    def __init__(
        self, images: List[Dict[str, Any]], labels: List[Dict[str, Any]], cfg: Any
    ):
        """Initialize the AnnotationViewer with images, labels, and configuration."""
        self.images = images
        self.labels = labels
        self.cfg = cfg
        self.index = 0
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.show_image()

    def show_image(self) -> None:
        """Display the current image with its annotation."""
        self.ax.clear()
        image = self.images[self.index]
        label = self.labels[self.index]
        img_name = image["file_name"]
        print(f"Showing annotation for img: {img_name}")
        bbox = label["bbox"]
        img_path = os.path.join(DATASET_DIR, self.cfg.out_folder, IMAGES_DIR, img_name)
        try:
            I = mpimg.imread(img_path)  # load rendered image
        except FileNotFoundError:
            print(f"Error: Image file {img_path} not found.")
            return
        self.ax.imshow(I)
        plt.axis("off")
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="g", facecolor="none"
        )  # add bounding box annotation
        self.ax.add_patch(rect)
        self.fig.canvas.draw()

    def on_key(self, event: Any) -> None:
        """Handle key press events."""
        if event.key == "escape":
            plt.close(self.fig)
        elif event.key == "right":
            self.index = (self.index + 1) % len(self.images)
            self.show_image()
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.images)
            self.show_image()

    def on_click(self, event: Any) -> None:
        """Handle mouse click events."""
        if event.button == 1:  # left mouse button
            self.index = (self.index + 1) % len(self.images)
            self.show_image()
        elif event.button == 3:  # right mouse button
            self.index = (self.index - 1) % len(self.images)
            self.show_image()


def main() -> None:
    """Main function to run the annotation viewer."""
    cfg = load_config()
    data = load_annotations(cfg)
    images = data["images"]
    labels = data["annotations"]
    print("Use the mouse buttons or arrow keys to navigate, close the viewer with Esc.")
    viewer = AnnotationViewer(images, labels, cfg)
    plt.show()


if __name__ == "__main__":
    main()
