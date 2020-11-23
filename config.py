#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:35:21 2020

@author: leon
"""

from easydict import EasyDict

cfg = EasyDict()

#  PATHS
#cfg.bg_path = '/home/leon/git/python-blender-datagen/bg/'
#cfg.bg_path = '/home/leon/datasets/indoorCVPR_09/Images/warehouse/' #Indoor Scene Recognition DB
cfg.bg_path = './bg'
cfg.environment_path = './environment'
#cfg.bg_path = '/home/leon/tubCloud/InternalShare'
#cfg.random_obj_path = '/home/leon/datasets/lm_models/models/'  # linemod
#cfg.random_obj_path = '/home/leon/datasets/tless_models/models_cad/'  # t less
cfg.model_path = './models/3dbox0922marker_new.ply'  # filepath to object
cfg.NumberOfObjects = 1

cfg.model_scale = 1E-2  # model scale for PLY objects

# DEPTH OUTPUT
cfg.output_depth = False
cfg.depth_color_depth = '16'

# AUGMENTATION
cfg.use_bg_image = False  # use Background Images
cfg.use_environment_maps = not cfg.use_bg_image  # use 360° HDRI Panoramas



# OBJECT COLOR
cfg.hsv_hue = 0.5 # changes hue of Hue Saturation Value Node, default 0.5
cfg.hsv_saturation = 1 # changes saturation of Hue Saturation Value Node, default 1
cfg.hsv_value = 0.35 # changes value of Hue Saturation Value Node, default 1
cfg.roughness  = 0.1 # Object Material Roughness (0=Mirror, 1=No Reflections)

#  CAMERA
cfg.clip_end = 50
cfg.clip_start = 0.01
#cfg.camera_min_r = 4  # minimum camera radius
#cfg.camera_max_r = 5  # maximum camera radius

cfg.cam_xmin = -12
cfg.cam_xmax = 12
cfg.cam_ymin = 7
cfg.cam_ymax = 9
cfg.cam_zmin = -12
cfg.cam_zmax = 12

#  OBJECT POSITION
cfg.obj_location_xmin = 0 # translation in meters
cfg.obj_location_xmax = 0
cfg.obj_location_ymin = -3
cfg.obj_location_ymax = 3
cfg.obj_location_zmin = 0
cfg.obj_location_zmax = 0

cfg.obj_rotation_xmin = 90 # euler rotation in degrees
cfg.obj_rotation_xmax = 90
cfg.obj_rotation_ymin = 0
cfg.obj_rotation_ymax = 0
cfg.obj_rotation_zmin = 0
cfg.obj_rotation_zmax = 0

cfg.max_boundingbox = 0.2 # filter out objects with bbox < -x or > 1+x

# ZED 2
cfg.cam_lens = 2.69 # Camera lens value in mm
cfg.cam_sensor_height = (2E-6 * 1520)*1000 #mm
cfg.cam_sensor_width = (2E-6 * 2688)*1000 # mm
#cfg.cam_distort = -0.001

#  RENDERING
cfg.use_GPU = True
cfg.use_cycles = True  # cycles or eevee
cfg.use_cycles_denoising = False
cfg.resolution_x = 640  # pixel resolution
cfg.resolution_y = 480
cfg.samples = 256  # render engine samples


#  OUTPUT
cfg.numberOfRenders = 1 # how many rendered examples

