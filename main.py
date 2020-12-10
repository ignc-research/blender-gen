#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# install packages with PIP into the blender python
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
from mathutils import Vector, Matrix

sys.path.append(os.getcwd())
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
      "categories": [{"supercategory": "object_category", "id": 0, "name": "object", "skeleton": [], "keypoints": []}],
      "licenses": "",
    }

    with open("annotation_coco.json", "w") as write_file:
        json.dump(coco, write_file, indent=2)



def orderCorners(objBB):
    """change bounding box corner order."""
    # change bounding box order according to
    # https://github.com/F2Wang/ObjectDatasetTools/blob/master/create_label_files.py
    out = []
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
    mat = bpy.data.materials.new('material_1')
    obj.active_material = mat
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    mat_links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs[7].default_value = cfg.roughness
    vcol = nodes.new(type="ShaderNodeVertexColor")
    vcol.layer_name = "Col"
    hsv = nodes.new(type="ShaderNodeHueSaturation")
    hsv.inputs[0].default_value = cfg.hsv_hue # hue
    hsv.inputs[1].default_value = cfg.hsv_saturation # saturation
    hsv.inputs[2].default_value = cfg.hsv_value # value
    #mat_links.new(vcol.outputs['Color'], bsdf.inputs['Base Color'])
    mat_links.new(vcol.outputs['Color'], hsv.inputs['Color'])
    mat_links.new(hsv.outputs['Color'], bsdf.inputs['Base Color'])
    return obj

def importOBJobject(filepath):
    bpy.ops.import_scene.obj(filepath=filepath)
    obj_objects = bpy.context.selected_objects[:]
    ctx = bpy.context.copy()
    ctx['active_object'] = obj_objects[0]
    ctx['selected_objects'] = obj_objects
    bpy.ops.object.join(ctx)  # join multiple elements into one element
    obj_objects[0].name = "Object"  # set object name to "Object"
    obj = bpy.data.objects['Object']
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
    #return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))
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

    # load bg image into ImageNode
    #bg_img = get_bg_image(cfg.bg_path)
    #img = bpy.data.images.load(os.path.join(cfg.bg_path, bg_img))
    #image_node.image = img

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
    #width = cfg.resolution_x
    #height = cfg.resolution_y
    camera.data.lens_unit = 'MILLIMETERS'
    camera.data.lens = cfg.cam_lens
    camera.data.sensor_height = cfg.cam_sensor_height
    camera.data.sensor_width = cfg.cam_sensor_width
    #camera.data.shift_x = 0
    #camera.data.shift_y = 0
    #camera.data.sensor_height = camera.data.sensor_width * height / width
    #camera.data.lens = (cfg.cam_fx + cfg.cam_fy) / 2 * camera.data.sensor_width / width
    #camera.data.shift_x = (width / 2 - cfg.cam_cx) / width
    #camera.data.shift_y = (cfg.cam_cy - height / 2) / width

    return camera


def get_camera_KRT(camera):
    """return 3x3 camera matrix K and 4x4 rotation, translation matrix RT."""
    # https://www.blender.org/forum/viewtopic.php?t=20231
    # Extrinsic and Intrinsic Camera Matrices
    scn = bpy.data.scenes['Scene']
    w = scn.render.resolution_x*scn.render.resolution_percentage/100.
    h = scn.render.resolution_y*scn.render.resolution_percentage/100.
    # Extrinsic
    RT = camera.matrix_world.inverted()
    # Intrinsic
    K = Matrix().to_3x3()
    K[0][0] = -w/2 / math.tan(camera.data.angle/2)
    ratio = w/h
    K[1][1] = -h/2. / math.tan(camera.data.angle/2) * ratio
    K[0][2] = w / 2.
    K[1][2] = h / 2.
    K[2][2] = 1.
    return K, RT


def save_camera_matrix(K):
    """save blenders camera matrix K to a file."""
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    Kdict = {
      "fx": K[0][0],
      "cy": K[0][2],
      "fy": K[1][1],
      "cx": K[1][2],
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


def setup_light(scene):
    """create a random point light source."""
    #  delete old lights
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)

    #  place new random light in cartesian coordinates
    x,y,z = get_sphere_coordinates(random.uniform(cfg.cam_rmin, cfg.cam_rmax), inclination=random.uniform(cfg.cam_incmin, cfg.cam_incmax), azimuth=random.uniform(cfg.cam_azimin, cfg.cam_azimax))
    #x = random.uniform(2*cfg.cam_xmin, 2*cfg.cam_xmax)
    #y = random.uniform(2*cfg.cam_ymin, 2*cfg.cam_ymax)
    #z = random.uniform(2*cfg.cam_zmin, 2*cfg.cam_zmax)
    lamp_data = bpy.data.lights.new(name='Light', type='POINT')
    lamp_data.energy = np.random.uniform(60, 5000)  # random energy in Watt
    lamp = bpy.data.objects.new(name='Light', object_data=lamp_data)
    bpy.context.collection.objects.link(lamp)
    lamp.location = (x, y, z)


def get_bg_image(bg_path=cfg.bg_path):
    """get list of all background images in folder 'bg_path' then choose random image."""
    img_list = os.listdir(bg_path)
    randomImgNumber = random.randint(0, len(img_list)-1)
    bg_img = img_list[randomImgNumber]
    return bg_img


def add_shader_on_world():
    """needed for Environment Map Background."""
    bpy.data.worlds['World'].use_nodes = True
    env_node = bpy.data.worlds['World'].node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    emission_node = bpy.data.worlds['World'].node_tree.nodes.new(type='ShaderNodeEmission')
    world_node = bpy.data.worlds['World'].node_tree.nodes['World Output']

    # connect env node with emission node
    bpy.data.worlds['World'].node_tree.links.new(env_node.outputs['Color'], emission_node.inputs['Color'])
    # connect emission node with world node
    bpy.data.worlds['World'].node_tree.links.new(emission_node.outputs['Emission'], world_node.inputs['Surface'])



def scene_cfg(camera, i):
    """configure the blender scene with random distributions."""
    scene = bpy.data.scenes['Scene']
    if (not cfg.use_environment_maps):
        setup_light(scene)  # light source not needed for HDR Maps

    # background
    if (cfg.use_environment_maps):
        # set HDRI Environment texture
        bg_img = get_bg_image(cfg.environment_path)
        bpy.data.images.load(os.path.join(cfg.environment_path, bg_img))
        bpy.data.worlds['World'].node_tree.nodes['Environment Texture'].image = bpy.data.images[bg_img]

        # set Emission Node Strength
        bpy.data.worlds['World'].node_tree.nodes['Emission'].inputs[1].default_value = random.uniform(cfg.emission_min, cfg.emission_max)

    elif (cfg.use_bg_image):
        bg_img = get_bg_image(cfg.bg_path)
        # set camera background image
        img = bpy.data.images.load(os.path.join(cfg.bg_path, bg_img))
        #camera.data.show_background_images = True
        #bg = camera.data.background_images.new()
        #bg.image = img
        # set render background image
        tree = bpy.context.scene.node_tree
        image_node = tree.nodes.get("Image")
        image_node.image = img

    repeat = True

    while (repeat):
        # random camera position
        camera=place_camera(camera, radius=random.uniform(cfg.cam_rmin, cfg.cam_rmax), inclination=random.uniform(cfg.cam_incmin, cfg.cam_incmax), azimuth=random.uniform(cfg.cam_azimin, cfg.cam_azimax))
        #camera.location.x = random.uniform(cfg.cam_xmin, cfg.cam_xmax)
        #camera.location.y = random.uniform(cfg.cam_ymin, cfg.cam_ymax)
        #camera.location.z = random.uniform(cfg.cam_zmin, cfg.cam_zmax)

        #obj = bpy.data.objects['Object']
        obj_list = bpy.context.selectable_objects # camera, objects

        for j in range(1, cfg.NumberOfObjects+1):
            obj = obj_list[j]
            if j!=1:
                obj1 = obj_list[j-1]
                obj.location = obj1.location
                obj.rotation_euler = obj1.rotation_euler
                obj.location.z += 2.1
            else:
                # random object pose
                obj.location.x = random.uniform(cfg.obj_location_xmin, cfg.obj_location_xmax)
                obj.location.y = random.uniform(cfg.obj_location_ymin, cfg.obj_location_ymax)
                obj.location.z = random.uniform(cfg.obj_location_zmin, cfg.obj_location_zmax)
                obj.rotation_euler = (random.uniform(cfg.obj_rotation_xmin*2*math.pi/360, cfg.obj_rotation_xmax*2*math.pi/360),
		    			random.uniform(cfg.obj_rotation_ymin*2*math.pi/360, cfg.obj_rotation_ymax*2*math.pi/360),
					random.uniform(cfg.obj_rotation_zmin*2*math.pi/360, cfg.obj_rotation_zmax*2*math.pi/360))

            # background objects location
            #_loc = obj.location + Vector(get_sphere_coordinates(radius=random.uniform(0.3,1.5), inclination=random.uniform(0,math.pi), azimuth=random.uniform(0, 2*math.pi)))
            #bpy.data.objects[clutter_obj].location = _loc

            # update blender object world_matrices!
            bpy.context.view_layer.update()

            # Some point in 3D you want to project
            #v = obj.location
            # Projecting v with the camera
            #K, RT = get_camera_KRT(camera)
            #p = K @ (RT @ v)
            #p /= p[2]
            #p[1] = 512 - p[1]  # openCV frame

            center = project_by_object_utils(camera, obj.location)  # object 2D center

            # set background image
            #if (cfg.use_bg_image):
            #    bg_img = get_bg_image()
            #else:
            #    bg_img = None

            class_ = 0  # class label for object
            labels = [class_]
            labels.append(center[0])  # center x coordinate in image space
            labels.append(center[1])  # center y coordinate in image space
            corners = orderCorners(obj.bound_box)  # change order from blender to SSD paper

            kps = []
            repeat = False
            for corner in corners:
                p = obj.matrix_world @ Vector(corner)  # object space to world space
                p = project_by_object_utils(camera, p)  # world space to image space
                labels.append(p[0])
                labels.append(p[1])
                if (p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1):
                    v = 1 # v=1: labeled but not visible
                else:
                    v = 2 # v=2: labeled and visible
                kps.append([p[0]*cfg.resolution_x, p[1]*cfg.resolution_y, v]) # 8 bounding box keypoints

                # filter out objects outside of the image view
                if (p[0] < -cfg.max_boundingbox or p[0] > (1+cfg.max_boundingbox) or
                    p[1] < -cfg.max_boundingbox or p[1] > (1+cfg.max_boundingbox)):
                    repeat = True
                    print('Repeating this Scene CFG')
                    print(p)



        min_x = np.min([labels[3], labels[5], labels[7], labels[9], labels[11], labels[13], labels[15], labels[17]])
        max_x = np.max([labels[3], labels[5], labels[7], labels[9], labels[11], labels[13], labels[15], labels[17]])

        min_y = np.min([labels[4], labels[6], labels[8], labels[10], labels[12], labels[14], labels[16], labels[18]])
        max_y = np.max([labels[4], labels[6], labels[8], labels[10], labels[12], labels[14], labels[16], labels[18]])

        # save labels in txt file
        x_range = max_x - min_x
        y_range = max_y - min_y
        labels.append(x_range)
        labels.append(y_range)

        # fix center point
        labels[1] = (max_x + min_x)/2
        labels[2] = (max_y + min_y)/2

        kps.insert(0, [cfg.resolution_x*(max_x + min_x)/2, cfg.resolution_y*(max_y + min_y)/2, 2]) # center is the 1st keypoint

        if cfg.NumberOfObjects == 1:
            mode = 'wb'
        else:
            mode = 'ab'
        if(not repeat):
            fname = "DATASET/object/labels/{:06}.txt".format(i)
            f = open(fname, mode)
            np.savetxt(f,labels, newline=' ', fmt='%1.6f')
            f.close()


            # COCO
            image = {
                "id":           i,
                "file_name":    "object/images/{:06}".format(i) + '.png',
                "height":       cfg.resolution_y,
                "width":        cfg.resolution_x,
                }
            annotation = {
                "id":              i,
                "image_id":        i,
                "bbox":            [min_x*cfg.resolution_x, min_y*cfg.resolution_y, x_range*cfg.resolution_x, y_range*cfg.resolution_y],
                "category_id":     0,
                "segmentation":    [],
                "iscrowd":          0,
                "area":             x_range*cfg.resolution_x*y_range*cfg.resolution_y,
                "keypoints":        kps,
                "num_keypoints":    9
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
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    #bpy.context.scene.render.image_settings.compression = 15

    # constrain camera to look at blenders (0,0,0) scene origin
    cam_constraint = camera.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    empty_obj = bpy.data.objects.new("empty", None)
    cam_constraint.target = empty_obj

    # composite node
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new(type="CompositorNodeRLayers")

    if(cfg.use_bg_image):
        setup_bg_image_nodes(rl)


    # camera distortion node
    #distortion_node = tree.nodes.new(type="CompositorNodeLensdist")
    #distortion_node.inputs[1].default_value = cfg.cam_distort
    #links.new(rl.outputs['Image'], distortion_node.inputs[0])
    #rgb_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    #links.new(distortion_node.outputs[0], rgb_file_output.inputs[0])

    # save depth output file?
    if (cfg.output_depth):
        depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.base_path = ''
        depth_file_output.format.file_format = 'PNG'  # 'OPEN_EXR'
        depth_file_output.format.color_depth = '16'  # cfg.depth_color_depth
        depth_file_output.format.color_mode='BW'

        map_node = tree.nodes.new(type="CompositorNodeMapRange")
        map_node.inputs[1].default_value = 0  # From Min
        map_node.inputs[2].default_value = 20  # From Max
        map_node.inputs[3].default_value = 0  # To Min
        map_node.inputs[4].default_value = 1  # To Max
        links.new(rl.outputs['Depth'], map_node.inputs[0])
        links.new(map_node.outputs[0], depth_file_output.inputs[0])
    else:
        depth_file_output = None

    bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True

    #  delete Cube from default blender scene
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    #obj = bpy.data.objects['Cube']

    #  import Model Object
    for i in range(cfg.NumberOfObjects):
        if (cfg.model_path[-4:] == '.obj' or cfg.model_path[-4:] == '.OBJ'):
            obj = importOBJobject(filepath=cfg.model_path)
        elif (cfg.model_path[-4:] == '.ply' or cfg.model_path[-4:] == '.PLY'):
            obj = importPLYobject(filepath=cfg.model_path, scale=cfg.model_scale)

    #  save Model real world Bounding Box for PnP algorithm
    np.savetxt("model_bounding_box.txt", orderCorners(obj.bound_box))

    #  import Environment Object (e.g. working desk)
    #if(not cfg.use_bg_image):
    #    bpy.ops.import_scene.fbx(filepath='/home/leon/Downloads/workplace/source/my_workplace2.fbx')
    #    obj_objects = bpy.context.selected_objects[:]
    #    ctx = bpy.context.copy()
    #    ctx['active_object'] = obj_objects[0]
    #    ctx['selected_objects'] = obj_objects
    #    bpy.ops.object.join(ctx)
    #    obj_objects[0].name = "Environment"
    #    env = bpy.data.objects['Environment']
    #    env.location = Vector((0, 0, 0))
    if (cfg.use_environment_maps):
        add_shader_on_world()  # used for HDR background image

    return camera, depth_file_output


def render_cfg():
    if(cfg.use_cycles):
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = cfg.samples
        bpy.context.scene.cycles.max_bounces = 8
        bpy.context.scene.cycles.use_denoising = cfg.use_cycles_denoising
        if (cfg.use_cycles_denoising and cfg.use_GPU):
            bpy.context.scene.cycles.denoiser = 'OPTIX' # Optix AI denoiser on NVIDIA GPU
        elif(cfg.use_cycles_denoising and not cfg.use_GPU):
            bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE' # Intel OpenImage AI denoiser on CPU
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.eevee.taa_render_samples = cfg.samples
    if(cfg.use_GPU):
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = "CUDA"


    # https://docs.blender.org/manual/en/latest/files/media/image_formats.html
    # set image width and height
    bpy.context.scene.render.resolution_x = cfg.resolution_x
    bpy.context.scene.render.resolution_y = cfg.resolution_y
    # set tile size??


def render(camera, depth_file_output, test_flag):
    render_cfg()  # setup render config once
    annotations = []
    images = []
    labels = []

    #  render loop
    if (test_flag):
        cfg.numberOfRenders=1
    for i in range(cfg.numberOfRenders):
        bpy.context.scene.render.filepath = './DATASET/object/images/{:06}.png'.format(i)
        bg_img, image, annotation = scene_cfg(camera, i)
        images.append(image)
        annotations.append(annotation)

        if (cfg.output_depth):
            depth_file_output.file_slots[0].path = bpy.context.scene.render.filepath + '_depth'
        bpy.ops.render.render(write_still=True, use_viewport=False)  # render current scene
        #if (cfg.use_bg_image):
        #    bpy.data.images.remove(bpy.data.images[bg_img])
    print('finished rendering')
    return images, annotations


def main():
    """
    call this script with 'blender --background --python main.py'

    edit the config.py file to change configuration parameters

    """
    #cfg = cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--python")
    parser.add_argument("--background", action="store_true") #  run blender in the background
    parser.add_argument("--test", help="plot bounding box with OpenCV after rendering",
                        action="store_true")
    args = parser.parse_args()


    camera, depth_file_output = setup()  # setup once

    images, annotations = render(camera, depth_file_output, args.test)  # render loop
    K, RT = get_camera_KRT(bpy.data.objects['Camera'])
    Kdict = save_camera_matrix(K)  # save Camera Matrix to K.txt
    bpy.ops.wm.save_as_mainfile(filepath="./scene.blend", check_existing=False)  # save current scene as .blend file

    saveCOCOlabel(images, annotations, Kdict)

    ##########################
    # test stuff at the end

    if args.test:
        import cv2
        image = cv2.imread('./DATASET/object/images/{:06}.png'.format(cfg.numberOfRenders-1))

        # load data
        BB = np.loadtxt("model_bounding_box.txt")
        label = np.loadtxt("./DATASET/object/labels/{:06}.txt".format(cfg.numberOfRenders-1))
        center = np.reshape(label[1:3], (1,2))
        center[:,0] *= cfg.resolution_x
        center[:,1] *= cfg.resolution_y
        label = np.reshape(label[3:19], (8,2))
        label[:,0] *= cfg.resolution_x
        label[:,1] *= cfg.resolution_y

        K = np.loadtxt("K.txt")


        # draw stuff into image with OpenCV
        center = (round(center[0,0]), round(center[0,1]))
        if(True):
            #cv2.circle(image, center, 1, (255, 0, 0), 3)  # center point in blue
            for i in range(8):
                x = (round(label[i,0]), round(label[i,1]))
                cv2.circle(image, x, 1, (0, 0, 255), 2)  # bounding box pixel-points

            # draw 3d bbox lines
            p0 = (round(label[0,0]), round(label[0,1]))
            p1 = (round(label[1,0]), round(label[1,1]))
            p2 = (round(label[2,0]), round(label[2,1]))
            p3 = (round(label[4,0]), round(label[4,1]))

            p4 = (round(label[7,0]), round(label[7,1]))
            p5 = (round(label[6,0]), round(label[6,1]))
            p6 = (round(label[5,0]), round(label[5,1]))
            p7 = (round(label[3,0]), round(label[3,1]))


            cv2.line(image, p0, p1, (0,0,255))
            cv2.line(image, p0, p2, (0,0,255))
            cv2.line(image, p0, p3, (0,0,255))

            cv2.line(image, p4, p5, (0,0,255))
            cv2.line(image, p4, p6, (0,0,255))
            cv2.line(image, p4, p7, (0,0,255))

            cv2.line(image, p1, p7, (0,0,255))
            cv2.line(image, p1, p6, (0,0,255))

            cv2.line(image, p3, p5, (0,0,255))
            cv2.line(image, p2, p5, (0,0,255))

            cv2.line(image, p2, p7, (0,0,255))

            cv2.line(image, p3, p6, (0,0,255))

        axis1 = np.float32([[0.1,0,0], [0,0.1,0], [0,0,-0.1]]).reshape(-1,3)
        axis0 = np.float32([[0,0,0], [0,0,0], [0,0,0]]).reshape(-1,3)

        # PnP
        #retval, rvec, tvec = cv2.solvePnP(BB, label, K, None)
        #imgpts1, jac = cv2.projectPoints(axis1, rvec, tvec, K, None)
        #imgpts0, jac = cv2.projectPoints(axis0, rvec, tvec, K, None)

        #cv2.line(image, tuple(imgpts0[0][0]), tuple(imgpts1[0][0]),(0, 0, 255) ,2)
        #cv2.line(image, tuple(imgpts0[1][0]), tuple(imgpts1[1][0]),(0, 255, 0) ,2)
        #cv2.line(image, tuple(imgpts0[2][0]), tuple(imgpts1[2][0]),(255, 0, 0) ,2)

        # 2d bbox
        if(False):
            label = np.loadtxt("./DATASET/object/labels/{:06}.txt".format(cfg.numberOfRenders-1))
            pt1 = (round(center[0] - label[-2]*cfg.resolution_x/2), round(center[1] - label[-1]*cfg.resolution_y/2)) # (center - x_range/2, center - y_range/2)
            pt2 = (round(center[0] + label[-2]*cfg.resolution_x/2), round(center[1] + label[-1]*cfg.resolution_y/2)) # (center + x_range/2, center + y_range/2)
            cv2.rectangle(image, pt1, pt2, (255, 0, 0))

        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyWindow("image")

    return True


if __name__ == '__main__':
    main()
