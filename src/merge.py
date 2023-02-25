import cv2 as cv
import numpy as np
import json
import os
import sys
import click
import grequests as requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import util

cfg = None
with open("/data/intermediate/config/render.json") as f:
    cfg = json.load(f)

storage = None

def reset():
    global storage
    storage = {
        "bg": {},
        "object": {},
        "distractor": {}
    }

reset()

def load(target: str, name: str):

    if name is storage[target]:
        return storage[target][name]
    
    if target == "bg":
        storage[target][name] = cv.resize(
            cv.imread(f"/data/intermediate/bg/{name}", cv.IMREAD_UNCHANGED),
            (cfg["resolution_x"], cfg["resolution_y"]),
            interpolation=cv.INTER_AREA
        )

    elif target == "object":
        storage[target][name] = cv.imread(f"/data/intermediate/render/renders/object/{name}", cv.IMREAD_UNCHANGED)

    elif target == "distractor":
        storage[target][name] = cv.imread(f"/data/intermediate/render/renders/distractor/{name}", cv.IMREAD_UNCHANGED)

    return storage[target][name]

def transform(img: np.ndarray, x: np.float64, y: np.float64, z: np.float64):
    z = z if z > -1+1e-1 else -1+1e-1 #maybe clip to zero instead idk

    trf = np.array([
            [1/(1+z), 0, (x +.5*z/(1+z))*cfg["resolution_x"]],
            [0, 1/(1+z), (y +.5*z/(1+z))*cfg["resolution_y"]]
        ], dtype=np.float32)

    return cv.warpAffine(
        img,
        trf,
        (cfg["resolution_x"], cfg["resolution_y"])
    ), trf

def layer(img: np.ndarray, overlay: np.ndarray):
    alpha = overlay[...,-1]/255.
    alphas = np.dstack((alpha,alpha,alpha))
    img *= 1-alphas
    img += alphas*overlay[...,:-1]
    return img

def merge(bg, obj, distractor):
    im_bg = load("bg", bg["name"])
    im_obj = load("object", obj["name"])
    im_distractor = list(map(lambda x: load("distractor",x["name"]), distractor))

    img = np.asarray(im_bg.copy(), dtype=np.float64)

    im_obj, trf = transform(im_obj, *obj["translation"])
    layer(img, im_obj)

    for im_dist, dist in zip(im_distractor, distractor):
        layer(img, transform(im_dist, *dist["translation"])[0])

    return img, trf
    

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--endpoint", default=None, help="http endpoint for sending current progress")
@click.option("--coco-image-root", default="/data/output/dataset/", help="http endpoint for sending current progress")


def main(endpoint, coco_image_root):

    merges = None
    with open("/data/intermediate/config/merge.json") as f:
        merges = json.load(f)
        
    annotations = None
    with open("/data/intermediate/render/annotations.json") as f:
        annotations = json.load(f)
    
    camera_K = None
    with open("/data/intermediate/render/camera_intrinsic.json") as f:
        camera_K = json.load(f)
    
    os.makedirs("/data/output/dataset/", exist_ok=True)

    coco_img = []
    coco_label = []

    total = len(merges)
    digits = len(str(total))

    print(f"\r{0:0{digits}} / {total}", end="", flush=True)

    for i, conf in enumerate(merges):

        merged, trf = merge(conf["bg"], conf["object"], conf["distractor"])

        id = f"{i:0{digits}}"
        path = os.path.join(coco_image_root, f"{id}.png")

        cv.imwrite(path, merged)

        coco_img.append({
            "id": id,
            "file_name": path,
            "height": cfg["resolution_x"],
            "width": cfg["resolution_y"],
        })

        box = annotations[conf["object"]["name"]]

        box["bbox"] = [
            trf[0,0] * box["bbox"][0] + trf[0,2],   #x1
            trf[1,1] * box["bbox"][1] + trf[1,2],   #x1
            trf[0,0] * box["bbox"][2],              #x2
            trf[1,1] * box["bbox"][3]               #y2
        ]

        coco_label.append({
            "id": id,
            "image_id": id,
            "bbox": box["bbox"],
            "category_id": 0,
            "segmentation": [],
            "iscrowd": 0,
            "area": box["bbox"][2] * box["bbox"][3],
            "keypoints": box["keypoints"],
            "num_keypoints": len(box["keypoints"])
        })

        print(f"\r{i+1:0{digits}} / {total}", end="", flush=True)
        
        if not endpoint == None:
            print(endpoint)
            requests.post(endpoint, data=dict(
                progress=i+1,
                total=total
            )).send()

    print()
    util.saveCOCOlabel(coco_img, coco_label, camera_K, "/data/output/")
    
if __name__ == "__main__":
    main()