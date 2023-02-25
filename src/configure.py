import json
import os
import numpy
import random
from math import ceil as cl
import click

def draw_samples(range, samples):
    return numpy.random.uniform(*range, size=int(samples or 1)).tolist()

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--mode", default="train", help="train|val create training or validation dataset")

def main(mode):

    config = None
    with open("/data/input/config/config.json") as f:
        config = json.load(f)

    if mode=="val":
        config["output"]["just_merge"] = 0
    config["output"]["just_merge"] = min(max(config["output"]["just_merge"], 0), 1)

    pos_dof = isinstance(config["random"]["x_pos"], list) or isinstance(config["random"]["y_pos"], list) or isinstance(config["random"]["z_pos"], list)

    os.makedirs("/data/intermediate/config/", exist_ok=True)

    ###RENDER

    conf_render = config["render"]

    with open("/data/intermediate/config/render.json", "w") as f:
        json.dump(conf_render, f)




    ###TARGETS

    to_produce = config["output"]["images"]
    to_produce *= (1-config["output"]["just_merge"])
    to_produce /= len(config["input"]["distractor"]) or 1
    to_produce /= (len(config["input"]["texture_distractor"])) or 1
    to_produce /= (len(config["input"]["bg"]) + len(config["input"]["environment"])) or 1

    dof_ang = isinstance(config["random"]["inc"], list) + isinstance(config["random"]["azi"], list)
    dof_mat = isinstance(config["random"]["metallic"], list) + isinstance(config["random"]["roughness"], list)

    each = (to_produce / (config["output"]["skew_angle:material"] ** dof_ang)) ** (1/(dof_ang + dof_mat))

    targets = dict(
        inc         =  max(1, cl(config["output"]["skew_angle:material"]*each)),
        azi         =  max(1, cl(config["output"]["skew_angle:material"]*each)),
        metallic    =  max(1, cl(each)),
        roughness   =  max(1, cl(each))
    )

    for target in targets:
        if isinstance(config["random"][target], list):
            targets[target] = draw_samples(config["random"][target], targets[target])
        else:
            targets[target] = [config["random"][target]]


    

    conf_targets = dict(
        object = dict(),
        distractor = dict()
    )

    conf_targets["object"][config["input"]["object"]] = dict(
        texture = [config["input"]["texture_object"]],
        **targets
    )

    for distractor in config["input"]["distractor"]:
        conf_targets["distractor"][distractor] = dict(
            texture = config["input"]["texture_distractor"],
            **targets
        )

    with open("/data/intermediate/config/targets.json", "w") as f:
        json.dump(conf_targets, f)




    ###MERGE

    bg = config["input"]["bg"]  #360° environment not implemented yet

    conf_merge = []

    dof_pos_x = isinstance(config["random"]["x_pos"], list)
    dof_pos_y = isinstance(config["random"]["y_pos"], list)
    dof_pos_z = isinstance(config["random"]["z_pos"], list)

    for i in range(config["output"]["images"]):
        merge = dict(
            bg = dict(
                name = random.choice(bg)
            ),
            object = dict(
                name        = f'{config["input"]["object"]}-{config["input"]["texture_object"]}-{random.choice(targets["inc"])}-{random.choice(targets["azi"])}-{random.choice(targets["metallic"])}-{random.choice(targets["roughness"])}.png',
                translation = [
                    draw_samples(config["random"]["x_pos"], 1)[0]  if dof_pos_x else config["random"]["x_pos"],
                    draw_samples(config["random"]["y_pos"], 1)[0]  if dof_pos_y else config["random"]["y_pos"],
                    draw_samples(config["random"]["z_pos"], 1)[0]  if dof_pos_z else config["random"]["z_pos"]
                ]
            ),
            distractor = []
        )

        for j in range(random.randint(*config["random"]["distractors"])):
            merge["distractor"].append(dict(
                name = f'{random.choice(config["input"]["distractor"])}-{random.choice(config["input"]["texture_distractor"])}-{random.choice(targets["inc"])}-{random.choice(targets["azi"])}-{random.choice(targets["metallic"])}-{random.choice(targets["roughness"])}.png',
                translation = [
                    draw_samples(config["random"]["x_pos"], 1)[0]  if dof_pos_x else config["random"]["x_pos"],
                    draw_samples(config["random"]["y_pos"], 1)[0]  if dof_pos_y else config["random"]["y_pos"],
                    draw_samples(config["random"]["z_pos"], 1)[0]  if dof_pos_z else config["random"]["z_pos"]
                ]
            ),)

        conf_merge.append(merge)

    with open("/data/intermediate/config/merge.json", "w") as f:
        json.dump(conf_merge, f)

    total = (1 + len(config["input"]["distractor"])  * len(config["input"]["texture_distractor"])) * numpy.prod([len(targets[k]) for k in targets])

    print(f"Configured {total} renders for {len(conf_merge)} images")
    print("Breakdown:")
    print(f'Objects:    {1 + len(config["input"]["distractor"]) * len(config["input"]["texture_distractor"])}')
    print(f'inc:        {len(targets["inc"])}')
    print(f'azi:        {len(targets["azi"])}')
    print(f'metallic:   {len(targets["metallic"])}')
    print(f'roughness:  {len(targets["roughness"])}')
    print("")
    
    

if __name__ == "__main__":
    main()