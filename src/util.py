import random
import datetime
import json
import os

def bench():
    ts = datetime.datetime.now()
    return lambda x: print(f"bench:: {x}: {datetime.datetime.now()-ts}")    

def orderCorners(objBB):
    """change bounding box corner order."""
    # change bounding box order according to
    # https://github.com/Microsoft/singleshotpose/blob/master/label_file_creation.md
    out = [objBB[i][:] for i in [0,1,3,2,4,5,7,6]]
    corners = [v[:] for v in objBB]  # list of tuples (x,y,z)
    out.append(corners[0])  # -1 -1 -1
    out.append(corners[1])  # -1 -1 1
    out.append(corners[3])  # -1 1 -1
    out.append(corners[2])  # -1 1 1
    out.append(corners[4])  # 1 -1 -1
    out.append(corners[5])  # 1 -1 1
    out.append(corners[7])  # 1 1 -1
    out.append(corners[6])  # 1 1 1
    return out


def kelvin_to_rgb(K):
    """converts color temperature in Kelvin to RGB values according to
    http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html"""
    table = {4000: (1.0000, 0.6636, 0.3583),
             5000: (1.0000, 0.7992, 0.6045),
             6000: (1.0000, 0.9019, 0.8473),
             7000: (0.9337, 0.9150, 1.0000),
             8000: (0.7874, 0.8187, 1.0000),
             9000: (0.6693, 0.7541, 1.0000),
             0:    (1, 1, 1)
             }
    rgb = table[K]
    return rgb

def get_random_temperature_color():  # 4K-9K test
        color_list = [(1.0000, 0.6636, 0.3583),  # 4000K
             (1.0000, 0.7992, 0.6045),  # 5000K
             (1.0000, 0.9019, 0.8473),  # 6000K
             (0.9337, 0.9150, 1.0000),  # 7000K
             (0.7874, 0.8187, 1.0000),  # 8000K
             (0.6693, 0.7541, 1.0000),  # 9000K
             (1.0,1.0,1.0) # white
             ]
        idx = random.randint(0, len(color_list)-1)
        return color_list[idx]


#def get_random_temperature_color():
#    color_list = [(1.0000, 0.2434, 0.0000),  # 1900K
#                  (1.0000, 0.3786, 0.0790),  # 2600K
#                  (1.0000, 0.4668, 0.1229),  # 2900K
#                  (1.0000, 0.4970, 0.1879),  # 3200K
#                  (1.0000, 0.8221, 0.6541),  # 5200K
#                  (1.0000, 0.8286, 0.7187),  # 5400K
#                  (1.0000, 0.9019, 0.8473),  # 6000K
#                  (0.9337, 0.9150, 1.0000),  # 7000K
#                  (0.3928, 0.5565, 1.0000)  # 20000K
#                  ]
#    idx = random.randint(0, len(color_list)-1)
#    return color_list[idx]


def saveCOCOlabel(images, annotations, Kdict, path):
    # https://cocodataset.org/#format-data
    info = {
        "year": datetime.datetime.now().year,
        "version": "1.0",
        "description": "Synthetic Dataset created with Blender Python script",
        "contributor": "IGNC",
        "url": "https://www.ignc.tu-berlin.de",
        "date_created": str(datetime.datetime.now()),
        "camera matrix K": Kdict
    }

    coco = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": [{
            "supercategory": "object_category",
            "id": 0,
            "name": "object",
            "skeleton": [],
            "keypoints": []
        }],
        "licenses": "",
    }

    with open(os.path.join(path, "annotation_coco.json"), "w") as write_file:
        json.dump(coco, write_file, indent=2)