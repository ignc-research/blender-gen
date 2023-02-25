import json
import os
import numpy
import random
from math import ceil as cl

meta_config = {
    "just_merge": .99,
    "skew_angle:mat": 4
}

def draw_samples(range, samples):
    return numpy.random.uniform(*range, size=int(samples or 1)).tolist()

def main():

    config = None
    with open("/data/input/config/config.json") as f:
        config = json.load(f)


    pos_dof = isinstance(config["random"]["x_pos"], list) or isinstance(config["random"]["y_pos"], list) or isinstance(config["random"]["z_pos"], list)

    os.makedirs("/data/intermediate/config/", exist_ok=True)

    ###RENDER

    conf_render = config["render"]

    with open("/data/intermediate/config/render.json", "w") as f:
        json.dump(conf_render, f)




    ###TARGETS

    to_produce = config["output"]["images"]
    to_produce *= (1-meta_config["just_merge"])
    to_produce /= len(config["input"]["distractor"]) or 1
    to_produce /= (len(config["input"]["texture_distractor"])) or 1
    to_produce /= (len(config["input"]["bg"]) + len(config["input"]["environment"])) or 1

    dof_ang = isinstance(config["random"]["inc"], list) + isinstance(config["random"]["azi"], list)
    dof_mat = isinstance(config["random"]["metallic"], list) + isinstance(config["random"]["roughness"], list)

    each = cl((to_produce / (meta_config["skew_angle:mat"] ** dof_ang)) ** (1/(dof_ang + dof_mat)))

    targets = {
        "inc": draw_samples(config["random"]["inc"], meta_config["skew_angle:mat"]*each) if isinstance(config["random"]["inc"], list) else [config["random"]["inc"]],
        "azi": draw_samples(config["random"]["azi"], meta_config["skew_angle:mat"]*each) if isinstance(config["random"]["azi"], list) else [config["random"]["azi"]],
        "metallic":  draw_samples(config["random"]["metallic"], each) if isinstance(config["random"]["metallic"], list) else [config["random"]["metallic"]],
        "roughness":  draw_samples(config["random"]["roughness"], each) if isinstance(config["random"]["roughness"], list) else [config["random"]["roughness"]],
    }

    total = (1 + len(config["input"]["distractor"])  * len(config["input"]["texture_distractor"])) * numpy.prod([len(targets[k]) for k in targets])

    conf_targets = {
        "object":{
        },
        "distractor":{
        }
    }

    conf_targets["object"][config["input"]["object"]] = {
        "texture": [config["input"]["texture_object"]],
        "inc": targets["inc"],
        "azi": targets["azi"],
        "metallic": targets["metallic"],
        "roughness": targets["roughness"]
    }

    for distractor in config["input"]["distractor"]:
        conf_targets["distractor"][distractor] = {
            "texture": config["input"]["texture_distractor"],
            "inc": targets["inc"],
            "azi": targets["azi"],
            "metallic": targets["metallic"],
            "roughness": targets["roughness"]
        }

    with open("/data/intermediate/config/targets.json", "w") as f:
        json.dump(conf_targets, f)




    ###MERGE

    bg = config["input"]["bg"]  #360° environment not implemented yet

    conf_merge = []

    dof_pos_x = isinstance(config["random"]["x_pos"], list)
    dof_pos_y = isinstance(config["random"]["y_pos"], list)
    dof_pos_z = isinstance(config["random"]["z_pos"], list)

    for i in range(config["output"]["images"]):
        merge = {
            "bg":{
                "name": random.choice(bg)
            },
            "object":{
                "name": f'{config["input"]["object"]}-{config["input"]["texture_object"]}-{random.choice(targets["inc"])}-{random.choice(targets["azi"])}-{random.choice(targets["metallic"])}-{random.choice(targets["roughness"])}.png',
                "translation": [
                    draw_samples(config["random"]["x_pos"], 1)[0]  if dof_pos_x else config["random"]["x_pos"],
                    draw_samples(config["random"]["y_pos"], 1)[0]  if dof_pos_y else config["random"]["y_pos"],
                    draw_samples(config["random"]["z_pos"], 1)[0]  if dof_pos_z else config["random"]["z_pos"]
                ]
            },
            "distractor":[]
        }

        for j in range(random.randint(*config["random"]["distractors"])):
            merge["distractor"].append({
                "name": f'{random.choice(config["input"]["distractor"])}-{random.choice(config["input"]["texture_distractor"])}-{random.choice(targets["inc"])}-{random.choice(targets["azi"])}-{random.choice(targets["metallic"])}-{random.choice(targets["roughness"])}.png',
                "translation": [
                    draw_samples(config["random"]["x_pos"], 1)[0]  if dof_pos_x else config["random"]["x_pos"],
                    draw_samples(config["random"]["y_pos"], 1)[0]  if dof_pos_y else config["random"]["y_pos"],
                    draw_samples(config["random"]["z_pos"], 1)[0]  if dof_pos_z else config["random"]["z_pos"]
                ]
            },)

        conf_merge.append(merge)

    with open("/data/intermediate/config/merge.json", "w") as f:
        json.dump(conf_merge, f)

    print(f"Configured {total} renders for {len(conf_merge)} images")

    

if __name__ == "__main__":
    main()