#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# to install packages with PIP into the blender python:
# e.g. /PATH/TO/BLENDER/python/bin$ ./python3.7m -m pip install pandas

import bpy
import bpy_extras
import os
import sys
import random
import math
import numpy as np
import json
import argparse
import datetime
import colorsys
import shutil
import glob
from mathutils import Vector, Matrix

sys.path.append(os.getcwd())
import util
import config

cfg = config.cfg()


def saveCOCOlabel(images, annotations, Kdict):
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

    with open("/data/annotation_coco.json", "w") as write_file:
        json.dump(coco, write_file, indent=2)


def importPLYobject(filepath, scale):
    """import PLY object from path and scale it."""

    bpy.ops.import_mesh.ply(filepath=filepath)
    obj_list = bpy.context.selected_objects[:]
    obj_list[0].name = "Object"
    #obj = bpy.data.objects['Object']
    obj = bpy.context.selected_objects[0]
    obj.scale = (scale, scale, scale)  # scale PLY object

    # add vertex color to PLY object
    obj.select_set(True)
    mat = bpy.data.materials.new('Material.001')
    obj.active_material = mat
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    vcol = nodes.new(type="ShaderNodeVertexColor")
    vcol.layer_name = "Col"

    mat_links.new(vcol.outputs['Color'], bsdf.inputs['Base Color'])

    # save object material inputs
    cfg.metallic.append(bsdf.inputs['Metallic'].default_value)
    cfg.roughness.append(bsdf.inputs['Roughness'].default_value)

    return obj


def importOBJobject(filepath, distractor=False):
    """import an *.OBJ file to Blender"""

    name = "Object"
    file_path = filepath
    if (distractor == True):
        name = "Distractor"
        file_path = glob.glob(filepath + '/*.obj')[0]
        texture_path = glob.glob(filepath + '/*.png')
        if (texture_path):
            texture_path = glob.glob(filepath + '/*.png')[0]

    bpy.ops.import_scene.obj(filepath=file_path, axis_forward='Y', axis_up='Z')
    print("importing model with axis_forward=Y, axis_up=Z")

    obj_objects = bpy.context.selected_objects[:]
    #ctx = bpy.context.copy()
    #ctx['active_object'] = obj_objects[0]
    #ctx['selected_objects'] = obj_objects
    # bpy.ops.object.join(ctx)  # join multiple elements into one element
    # bpy.ops.object.join(obj_objects)  # join multiple elements into one eleme
    obj_objects[0].name = name  # set object name

    # get BSDF material node
    obj = obj_objects[0]
    mat = obj.active_material
    nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    if (distractor == True and
            texture_path):  # import original distractor png texture
        texture = nodes.get("Image Texture")
        bpy.data.images.load(
            texture_path)  # texture needs the same name as obj file
        texture.image = bpy.data.images[os.path.split(texture_path)[1]]

    elif (distractor == True and len(cfg.distractor_texture_path) >
          0):  # distractor with random texture
        texture = nodes.new(type="ShaderNodeTexImage")  # new node
        mat_links.new(
            texture.outputs['Color'],
            bsdf.inputs['Base Color'])  # link texture node to bsdf node

    if (len(cfg.object_texture_path) > 0 and
            distractor == False):  # use random image texture on object
        texture = nodes.new(type="ShaderNodeTexImage")  # new node
        mat_links.new(
            texture.outputs['Color'],
            bsdf.inputs['Base Color'])  # link texture node to bsdf node

    # save object material inputs
    cfg.metallic.append(bsdf.inputs['Metallic'].default_value)
    cfg.roughness.append(bsdf.inputs['Roughness'].default_value)

    return obj


def project_by_object_utils(cam, point):
    """returns normalized (x, y) image coordinates in OpenCV frame for a given blender world point."""
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    # convert y coordinate to opencv coordinate system!
    # return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))
    return Vector((co_2d.x, 1 - co_2d.y))  # normalized


def setup_bg_image_nodes(rl):
    """setup all compositor nodes to render background images"""
    # https://henryegloff.com/how-to-render-a-background-image-in-blender-2-8/

    bpy.context.scene.render.film_transparent = True

    # create nodes
    tree = bpy.context.scene.node_tree
    links = tree.links
    alpha_node = tree.nodes.new(type="CompositorNodeAlphaOver")
    composite_node = tree.nodes.new(type="CompositorNodeComposite")
    scale_node = tree.nodes.new(type="CompositorNodeScale")
    image_node = tree.nodes.new(type="CompositorNodeImage")

    scale_node.space = 'RENDER_SIZE'
    scale_node.frame_method = 'CROP'

    # link nodes
    links.new(rl.outputs['Image'], alpha_node.inputs[2])
    links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    links.new(scale_node.outputs['Image'], alpha_node.inputs[1])
    links.new(alpha_node.outputs['Image'], composite_node.inputs['Image'])


def setup_camera():
    """set camera config."""
    camera = bpy.data.objects['Camera']

    # camera config
    bpy.data.cameras['Camera'].clip_start = cfg.clip_start
    bpy.data.cameras['Camera'].clip_end = cfg.clip_end

    # CAMERA CONFIG
    camera.data.sensor_height = cfg.cam_sensor_height
    camera.data.sensor_width = cfg.cam_sensor_width
    #width = cfg.resolution_x
    #height = cfg.resolution_y
    # camera.data.lens_unit = 'FOV'#'MILLIMETERS'
    if cfg.cam_lens_unit == 'FOV':
        camera.data.lens_unit = 'FOV'
        camera.data.angle = (cfg.cam_fov / 360) * 2 * math.pi
    else:
        camera.data.lens_unit = 'MILLIMETERS'
        camera.data.lens = cfg.cam_lens

    return camera


def get_camera_KRT(camera):
    """return 3x3 camera matrix K and 4x4 rotation, translation matrix RT.
    Experimental feature, the matrix might be wrong!"""
    # https://www.blender.org/forum/viewtopic.php?t=20231
    # Extrinsic and Intrinsic Camera Matrices
    scn = bpy.data.scenes['Scene']
    w = scn.render.resolution_x * scn.render.resolution_percentage / 100.
    h = scn.render.resolution_y * scn.render.resolution_percentage / 100.
    # Extrinsic
    RT = camera.matrix_world.inverted()
    # Intrinsic
    K = Matrix().to_3x3()
    K[0][0] = -w / 2 / math.tan(camera.data.angle / 2)
    ratio = w / h
    K[1][1] = -h / 2. / math.tan(camera.data.angle / 2) * ratio
    K[0][2] = w / 2.
    K[1][2] = h / 2.
    K[2][2] = 1.
    return K, RT


def save_camera_matrix(K):
    """save blenders camera matrix K to a file."""
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    Kdict = {
        "fx": K[0][0],
        "cx": K[0][2],
        "fy": K[1][1],
        "cy": K[1][2],
    }

    with open("camera_intrinsic.json", "w") as write_file:
        json.dump(Kdict, write_file)
        # save as json for better readability
    np.savetxt("K.txt", K)
    return Kdict


def get_sphere_coordinates(radius, inclination, azimuth):
    """convert sphere to cartesian coordinates."""
    #  https://de.m.wikipedia.org/wiki/Kugelkoordinaten
    #  radius r, inclination θ, azimuth φ)
    #  inclination [0, pi]
    #  azimuth [0, 2pi]
    x = radius * math.sin(inclination) * math.cos(azimuth)
    y = radius * math.sin(inclination) * math.sin(azimuth)
    z = radius * math.cos(inclination)
    return (x, y, z)


def place_camera(camera, radius, inclination, azimuth):
    """sample x,y,z on sphere and place camera (looking at the origin)."""
    x, y, z = get_sphere_coordinates(radius, inclination, azimuth)
    camera.location.x = x
    camera.location.y = y
    camera.location.z = z

    bpy.context.view_layer.update()
    return camera


def setup_light(scene, light_number=1, random_color=None):
    """create a random point light source."""
    if (random_color == "temperature"):
        light_color = util.get_random_temperature_color()
    for i in range(light_number):
        #  place new random light in cartesian coordinates
        x, y, z = get_sphere_coordinates(
            random.uniform(cfg.cam_rmin, cfg.cam_rmax),
            inclination=random.uniform(cfg.cam_incmin, cfg.cam_incmax),
            azimuth=random.uniform(cfg.cam_azimin, cfg.cam_azimax))
        light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
        light_data.energy = random.uniform(
            cfg.light_energymin, cfg.light_energymax)  # random energy in Watt
        if (random_color == "projector"):
            light_data.color = colorsys.hsv_to_rgb(random.random(), 1, 1)
        elif (random_color == "temperature"):
            light_data.color = light_color  # util.get_random_temperature_color()
        light_object = bpy.data.objects.new(name="my-light",
                                            object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = (x, y, z)


def get_bg_image(bg_path=cfg.bg_paths):
    """get list of all background images in folder 'bg_path' then choose random image."""
    idx = random.randint(0, len(bg_path) - 1)

    img_list = os.listdir(bg_path[idx])
    randomImgNumber = random.randint(0, len(img_list) - 1)
    bg_img = img_list[randomImgNumber]
    bg_img_path = os.path.join(bg_path[idx], bg_img)
    return bg_img, bg_img_path


def add_shader_on_world():
    """needed for Environment Map Background."""
    bpy.data.worlds['World'].use_nodes = True
    env_node = bpy.data.worlds['World'].node_tree.nodes.new(
        type='ShaderNodeTexEnvironment')
    emission_node = bpy.data.worlds['World'].node_tree.nodes.new(
        type='ShaderNodeEmission')
    world_node = bpy.data.worlds['World'].node_tree.nodes['World Output']

    # connect env node with emission node
    bpy.data.worlds['World'].node_tree.links.new(env_node.outputs['Color'],
                                                 emission_node.inputs['Color'])
    # connect emission node with world node
    bpy.data.worlds['World'].node_tree.links.new(
        emission_node.outputs['Emission'], world_node.inputs['Surface'])


def scene_cfg(camera, i):
    """configure the blender scene with random distributions."""
    scene = bpy.data.scenes['Scene']
    if (not cfg.use_environment_maps):
        light_nr = random.randint(
            cfg.light_number_min,
            cfg.light_number_max)  # sample number n of Point Lights
        setup_light(scene, light_number=light_nr, random_color=cfg.random_color)

    # background
    if (cfg.use_environment_maps):
        # set HDRI Environment texture
        bg_img, bg_img_path = get_bg_image(cfg.environment_paths)
        bpy.data.images.load(bg_img_path)
        bpy.data.worlds['World'].node_tree.nodes[
            'Environment Texture'].image = bpy.data.images[bg_img]

        # set Emission Node Strength E
        bpy.data.worlds['World'].node_tree.nodes['Emission'].inputs[
            1].default_value = random.uniform(cfg.emission_min,
                                              cfg.emission_max)

    if (cfg.use_bg_image):
        bg_img, bg_img_path = get_bg_image(cfg.bg_paths)
        # set camera background image
        img = bpy.data.images.load(bg_img_path)
        tree = bpy.context.scene.node_tree
        image_node = tree.nodes.get("Image")
        image_node.image = img

    obj_list = bpy.context.selectable_objects  # camera, objects
    mesh_list_objects = []
    mesh_list_distractors = []

    # hide all objects
    for o in obj_list:
        if o.type == 'MESH':
            if o.name.find('Distractor') != -1:
                o.hide_render = True
                mesh_list_distractors.append(o)
            elif o.name.find('Object') != -1:
                o.hide_render = True
                mesh_list_objects.append(o)

    x = random.randint(
        0,
        len(cfg.model_paths) -
        1)  # select random number of objects to render, hide the rest
    obj = mesh_list_objects[x]
    obj.hide_render = False
    #mat = obj.active_material

    # change distractor object texture
    if (len(cfg.distractor_texture_path) > 0):
        for distractor in mesh_list_distractors:
            mat = distractor.active_material
            nodes = mat.node_tree.nodes
            texture = nodes.get("Image Texture")
            texture_list = os.listdir(cfg.distractor_texture_path)
            texture_path = texture_list[random.randint(0,
                                                       len(texture_list) - 1)]
            #texture.image = bpy.data.images[os.path.split(texture_path)[1]]
            bpy.data.images.load(cfg.distractor_texture_path + '/' +
                                 texture_path)
            texture.image = bpy.data.images[texture_path]

    # change object texture
    if (len(cfg.object_texture_path) > 0):
        mat = obj.active_material
        nodes = mat.node_tree.nodes
        texture = nodes.get("Image Texture")
        texture_list = os.listdir(cfg.object_texture_path)
        texture_path = texture_list[random.randint(0, len(texture_list) - 1)]
        #texture.image = bpy.data.images[os.path.split(texture_path)[1]]
        #  load object textures
        bpy.data.images.load(cfg.object_texture_path + '/' + texture_path)
        texture.image = bpy.data.images[texture_path]

    if (not cfg.distractor_paths):  # an empty list is False
        n = 0
    else:
        n = random.randint(0, cfg.max_distractor_objects)
    for j in range(n):
        # select random object to render, hide the rest
        y = random.randint(0, len(cfg.distractor_paths) - 1)
        dis_obj = mesh_list_distractors[y]
        dis_obj.hide_render = False

        # position distractor objects
        dis_obj.location.x = random.uniform(cfg.obj_location_xmin,
                                            cfg.obj_location_xmax)
        dis_obj.location.y = random.uniform(cfg.obj_location_ymin,
                                            cfg.obj_location_ymax)
        dis_obj.location.z = random.uniform(cfg.obj_location_zmin,
                                            cfg.obj_location_zmax)
        rot_angle1 = random.uniform(cfg.cam_rotation_min, cfg.cam_rotation_max)
        rot_angle2 = random.uniform(cfg.cam_rotation_min, cfg.cam_rotation_max)
        rot_angle3 = random.uniform(cfg.cam_rotation_min, cfg.cam_rotation_max)
        dis_obj.rotation_euler = (rot_angle1, rot_angle2, rot_angle3)

    # random metallic material
    if (cfg.random_metallic_value):
        if (random.random() >= 0.5):
            mat.node_tree.nodes["Principled BSDF"].inputs[
                'Metallic'].default_value = random.random()
        else:
            mat.node_tree.nodes["Principled BSDF"].inputs[
                'Metallic'].default_value = cfg.metallic[x - 1]

    # random roughness material
    if (cfg.random_roughness_value):
        if (random.random() >= 0.5):
            mat.node_tree.nodes["Principled BSDF"].inputs[
                'Roughness'].default_value = random.random()
        else:
            mat.node_tree.nodes["Principled BSDF"].inputs[
                'Roughness'].default_value = cfg.roughness[x - 1]

    # random projector augmentation (point light with random color)
    if (cfg.random_color == "projector"):
        if (random.random() >= 0.5):
            setup_light(scene, light_number=1, random_color="projector")

    repeat = True
    while (repeat):
        # random camera position x_c, y_c, z_c
        camera = place_camera(camera,
                              radius=random.uniform(cfg.cam_rmin, cfg.cam_rmax),
                              inclination=random.uniform(
                                  cfg.cam_incmin, cfg.cam_incmax),
                              azimuth=random.uniform(cfg.cam_azimin,
                                                     cfg.cam_azimax))

        empty_obj = bpy.data.objects["empty"]

        # random object pose
        obj.location.x = random.uniform(cfg.obj_location_xmin,
                                        cfg.obj_location_xmax)  # x_o
        obj.location.y = random.uniform(cfg.obj_location_ymin,
                                        cfg.obj_location_ymax)  # y_o
        obj.location.z = random.uniform(cfg.obj_location_zmin,
                                        cfg.obj_location_zmax)  # z_o

        rot_angle1 = random.uniform(cfg.cam_rotation_min,
                                    cfg.cam_rotation_max)  #  alpha 1
        rot_angle2 = random.uniform(cfg.cam_rotation_min,
                                    cfg.cam_rotation_max)  #  alpha 2
        rot_angle3 = random.uniform(cfg.cam_rotation_min,
                                    cfg.cam_rotation_max)  #  alpha 3
        empty_obj.rotation_euler = (rot_angle1, rot_angle2, rot_angle3
                                   )  # XYZ euler rotation on the empty object

        # update blender object world_matrices!
        bpy.context.view_layer.update()

        # Some point in 3D you want to project
        #v = obj.location
        # Projecting v with the camera
        #K, RT = get_camera_KRT(camera)
        #p = K @ (RT @ v)
        #p /= p[2]
        # p[1] = 512 - p[1]  # openCV frame

        center = project_by_object_utils(camera,
                                         obj.location)  # object 2D center

        class_ = 0  # class label for object
        labels = [class_]
        labels.append(center[0])  # center x coordinate in image space
        labels.append(center[1])  # center y coordinate in image space
        corners = util.orderCorners(
            obj.bound_box)  # change order from blender to SSD paper
        if (cfg.use_fps_keypoints):
            corners = np.loadtxt("fps.txt")

        kps = []
        repeat = False
        for corner in corners:
            p = obj.matrix_world @ Vector(corner)  # object space to world space
            p = project_by_object_utils(camera, p)  # world space to image space
            labels.append(p[0])
            labels.append(p[1])
            if (p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1):
                v = 1  # v=1: labeled but not visible
            else:
                v = 2  # v=2: labeled and visible
            # 8 bounding box keypoints
            kps.append([p[0] * cfg.resolution_x, p[1] * cfg.resolution_y, v])

            # filter out objects outside of the image view
            if (p[0] < -cfg.max_boundingbox or p[0] >
                (1 + cfg.max_boundingbox) or p[1] < -cfg.max_boundingbox or
                    p[1] > (1 + cfg.max_boundingbox)):
                repeat = True
                print('Repeating this Scene CFG')
                print(p)

        # P=[RT] ground truth pose of the object in camera coordinates???
        P = camera.matrix_world.inverted() @ obj.matrix_world

        # compute bounding box either with 3D bbox or by going through vertices
        if (cfg.compute_bbox == 'tight'
           ):  # loop through all vertices and transform to image coordinates
            min_x, max_x, min_y, max_y = 1, 0, 1, 0
            vertices = obj.data.vertices
            for v in vertices:
                vec = project_by_object_utils(camera,
                                              obj.matrix_world @ Vector(v.co))
                x = vec[0]
                y = vec[1]
                if x > max_x:
                    max_x = x
                if x < min_x:
                    min_x = x
                if y > max_y:
                    max_y = y
                if y < min_y:
                    min_y = y
        else:  # use blenders 3D bbox (simple but fast)
            min_x = np.min([
                labels[3], labels[5], labels[7], labels[9], labels[11],
                labels[13], labels[15], labels[17]
            ])
            max_x = np.max([
                labels[3], labels[5], labels[7], labels[9], labels[11],
                labels[13], labels[15], labels[17]
            ])

            min_y = np.min([
                labels[4], labels[6], labels[8], labels[10], labels[12],
                labels[14], labels[16], labels[18]
            ])
            max_y = np.max([
                labels[4], labels[6], labels[8], labels[10], labels[12],
                labels[14], labels[16], labels[18]
            ])

        # save labels in txt file (deprecated)
        x_range = max_x - min_x
        y_range = max_y - min_y
        labels.append(x_range)
        labels.append(y_range)

        # fix center point
        labels[1] = (max_x + min_x) / 2
        labels[2] = (max_y + min_y) / 2

        #  keypoints (kps) for 6D Pose Estimation
        #kps.insert(0, [
        #    cfg.resolution_x * (max_x + min_x) / 2, cfg.resolution_y *
        #    (max_y + min_y) / 2, 2
        #])  # center is the 1st keypoint

        if (cfg.use_fps_keypoints == False):
            kps.insert(
                0,
                [cfg.resolution_x * center[0], cfg.resolution_y * center[1], 2
                ])  # center is the 1st keypoint

        if (not repeat):
            # save COCO label
            image = {
                "id": i,
                "file_name": "object/images/{:06}".format(i) + '.jpg',
                "height": cfg.resolution_y,
                "width": cfg.resolution_x,
            }
            annotation = {
                "id": i,
                "image_id": i,
                "bbox": [
                    min_x * cfg.resolution_x, min_y * cfg.resolution_y,
                    x_range * cfg.resolution_x, y_range * cfg.resolution_y
                ],
                "category_id": 0,
                "segmentation": [],
                "iscrowd": 0,
                "area": x_range * cfg.resolution_x * y_range * cfg.resolution_y,
                "keypoints": kps,
                "num_keypoints": len(kps)
            }

    return bg_img, image, annotation


def setup():
    """one time config setup for blender."""
    bpy.ops.object.select_all(action='TOGGLE')
    camera = setup_camera()

    # delete Light
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)

    # configure rendered image's parameters
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '8'  # Bit depth per channel [8,16,32]
    bpy.context.scene.render.image_settings.file_format = 'JPEG'  # 'PNG'
    bpy.context.scene.render.image_settings.compression = 0  # JPEG compression
    bpy.context.scene.render.image_settings.quality = 100

    # constrain camera to look at blenders (0,0,0) scene origin (empty_object)
    cam_constraint = camera.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_constraint.use_target_z = True
    empty_obj = bpy.data.objects.new("empty", None)
    cam_constraint.target = empty_obj

    # composite node
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new(type="CompositorNodeRLayers")

    if (cfg.use_bg_image):
        setup_bg_image_nodes(rl)

    # save depth output file? not tested!
    if (cfg.output_depth):
        depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.base_path = ''
        depth_file_output.format.file_format = 'PNG'  # 'OPEN_EXR'
        depth_file_output.format.color_depth = '16'  # cfg.depth_color_depth
        depth_file_output.format.color_mode = 'BW'

        map_node = tree.nodes.new(type="CompositorNodeMapRange")
        map_node.inputs[1].default_value = 0  # From Min
        map_node.inputs[2].default_value = 20  # From Max
        map_node.inputs[3].default_value = 0  # To Min
        map_node.inputs[4].default_value = 1  # To Max
        links.new(rl.outputs['Depth'], map_node.inputs[0])
        links.new(map_node.outputs[0], depth_file_output.inputs[0])
    else:
        depth_file_output = None

    #bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True

    #  delete Cube from default blender scene
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()

    #  import Model Object
    NumberOfObjects = len(cfg.model_paths)
    for i in range(NumberOfObjects):
        if (cfg.model_paths[i][-4:] == '.obj' or
                cfg.model_paths[i][-4:] == '.OBJ'):
            obj = importOBJobject(filepath=cfg.model_paths[i])
        elif (cfg.model_paths[i][-4:] == '.ply' or
              cfg.model_paths[i][-4:] == '.PLY'):
            obj = importPLYobject(filepath=cfg.model_paths[i],
                                  scale=cfg.model_scale)

    #  import Distractor Objects
    NumberOfObjects = len(cfg.distractor_paths)
    for i in range(NumberOfObjects):
        obj = importOBJobject(filepath=cfg.distractor_paths[i], distractor=True)

    #  save Model real world Bounding Box for PnP algorithm
    np.savetxt("model_bounding_box.txt", util.orderCorners(obj.bound_box))

    if (cfg.use_environment_maps):
        add_shader_on_world()  # used for HDR background image

    return camera, depth_file_output


def render_cfg():
    """setup Blenders render engine (EEVEE or CYCLES) once"""
    # refresh the list of devices
    devices = bpy.context.preferences.addons["cycles"].preferences.get_devices()
    if devices:
        devices = devices[0]
        for d in devices:
            d["use"] = 1  # activate all devices
            print("activating device: " + str(d["name"]))
    if (cfg.use_cycles):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = cfg.samples
        bpy.context.scene.cycles.max_bounces = 8
        bpy.context.scene.cycles.use_denoising = cfg.use_cycles_denoising
        bpy.context.scene.cycles.use_adaptive_sampling = cfg.use_adaptive_sampling
        bpy.context.scene.cycles.adaptive_min_samples = 50
        bpy.context.scene.cycles.adaptive_threshold = 0.001
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # Intel OpenImage AI denoiser on CPU
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = cfg.samples
    if (cfg.use_GPU):
        #bpy.context.scene.render.tile_x = 64
        #bpy.context.scene.render.tile_y = 64
        bpy.context.preferences.addons[
            'cycles'].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = 'GPU'

    # https://docs.blender.org/manual/en/latest/files/media/image_formats.html
    # set image width and height
    bpy.context.scene.render.resolution_x = cfg.resolution_x
    bpy.context.scene.render.resolution_y = cfg.resolution_y


def render(camera, depth_file_output):
    """main loop to render images"""

    render_cfg()  # setup render config once
    annotations = []
    images = []
    labels = []

    start_time = datetime.datetime.now()

    #  render loop
    if (cfg.test):
        cfg.numberOfRenders = 1
    for i in range(cfg.numberOfRenders):

        bpy.context.scene.render.filepath = '/data/' + cfg.out_folder + '/images/{:06}.jpg'.format(
            i)
        bg_img, image, annotation = scene_cfg(camera, i)
        images.append(image)
        annotations.append(annotation)

        if (cfg.output_depth):
            depth_file_output.file_slots[
                0].path = bpy.context.scene.render.filepath + '_depth'
        bpy.ops.render.render(write_still=True,
                              use_viewport=False)  # render current scene

        #for block in bpy.data.images:  # delete loaded images (bg + hdri)
        #    if (not cfg.test):
        #        bpy.data.images.remove(block)
        for block in bpy.data.lights:  # delete lights
            if (not cfg.test):
                bpy.data.lights.remove(block)

    end_time = datetime.datetime.now()
    dt = end_time - start_time
    secondsPerRender = dt.seconds / cfg.numberOfRenders
    print('---------------')
    print('finished rendering')
    print('total render time (hh:mm:ss): ' + str(dt))
    print('average seconds per image: ' + str(secondsPerRender))

    return images, annotations


def main():
    """
    call this script with 'blender --background --python main.py'

    edit the config.py file to change configuration parameters

    """
    random.seed(cfg.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--python")
    parser.add_argument("--background",
                        action="store_true")  # run blender in the background
    args = parser.parse_args()

    camera, depth_file_output = setup()  # setup once

    images, annotations = render(camera, depth_file_output)  # render loop
    K, RT = get_camera_KRT(bpy.data.objects['Camera'])
    Kdict = save_camera_matrix(K)  # save Camera Matrix to K.txt 
    scene_path = os.path.join(os.getcwd(), 'scene.blend') # save current scene as .blend file, by providing the absolute path
    bpy.ops.wm.save_as_mainfile(
        filepath=scene_path,
        check_existing=False)  # save current scene as .blend file
    shutil.copy2('config.py',
                 'data/' + cfg.out_folder)  # save config.py file
    saveCOCOlabel(images, annotations,
                  Kdict)  # save COCO annotation file at the end

    return True


if __name__ == '__main__':
    main()
