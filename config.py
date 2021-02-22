#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:35:21 2020

@author: leon
"""

#from easydict import EasyDict
from math import pi

#self = EasyDict()

class cfg:
  def __init__(self):
    #  PATHS
    self.bg_path = './bg/coco/train2017'
    self.environment_path = './environment'
    #self.random_obj_path = '/home/leon/datasets/tless_models/models_cad/'  # t less
    #self.model_paths = ['./models/H8000.obj', './models/schaufel.obj'] #3dbox0922marker_new.ply'  # list of filepath to objects
    #self.model_paths = ['./models/H8000.obj', './models/4000F_1.obj', './models/4000F_2.obj'] #3dbox0922marker_new.ply'  # list of filepath to objects
    self.model_paths = ['./models/3dbox0922marker_new.ply']  # filepath to object

    self.NumberOfObjects = 1

    self.model_scale = 1  # model scale for PLY objects

    # DEPTH OUTPUT
    self.output_depth = False
    self.depth_color_depth = '16'

    # AUGMENTATION
    self.use_bg_image = False # use Background Images
    self.use_environment_maps = True  # use 360° HDRI Panoramas
    self.emission_min = 0.5 # only for environment maps
    self.emission_max = 5 # only for environment maps
    self.light_number = 2 # only for background images
    self.light_energymin = 200 # only for background images
    self.light_energymax = 1000 # only for background images
    self.random_hsv_value = False # randomize the value of HSV color space of the object with p=0.5
    self.random_metallic_value = False # randomize the metallic object value with p=0.5
    self.random_roughness_value = False # randomize the roughness object value with p=0.5

    # OBJECT COLOR (for PLY Files)
    self.hsv_hue = 0.5 # changes hue of Hue Saturation Value Node, default 0.5
    self.hsv_saturation = 1 # changes saturation of Hue Saturation Value Node, default 1
    self.hsv_value = 1#0.35 # changes value of Hue Saturation Value Node, default 1
    #self.roughness  = 0.3#0.1 # Object Material Roughness (0=Mirror, 1=No Reflections)

    #  CAMERA
    self.clip_end = 50
    self.clip_start = 0.01
    # camera sphere coordinates
    self.cam_rmin = 0.3  # minimum camera distance
    self.cam_rmax = 1.1  # maximum camera distance
    self.cam_incmin = 0
    self.cam_incmax = pi*2/3
    self.cam_azimin = 0
    self.cam_azimax = 2*pi

    # camera cartesian coordinates
    #self.cam_xmin = -0.6
    #self.cam_xmax = 0.6
    #self.cam_ymin = -0.4
    #self.cam_ymax = 0.4
    #self.cam_zmin = 1.1
    #self.cam_zmax = 1.3

    #  OBJECT POSITION
    self.obj_location_xmin = 0.1 # translation in meters
    self.obj_location_xmax = -0.1
    self.obj_location_ymin = -0.1
    self.obj_location_ymax = 0.1
    self.obj_location_zmin = -0.1
    self.obj_location_zmax = 0.1

    self.obj_rotation_xmin = 0 # euler rotation in degrees
    self.obj_rotation_xmax = 0
    self.obj_rotation_ymin = 0
    self.obj_rotation_ymax = 0
    self.obj_rotation_zmin = 0
    self.obj_rotation_zmax = 0

    self.max_boundingbox = 0.05 # filter out objects with bbox < -x or > 1+x

    # camera intrinsics
    self.cam_lens_unit = 'FOV' # Choose 'FOV' or 'MILLIMETERS'
    self.cam_lens = 4.7 # Camera lens value in mm
    self.cam_fov = (59+90)/2 # camera field of view in degrees
    self.cam_sensor_height = 3.84  # mm
    self.cam_sensor_width = 5.11  # mm

    #  RENDERING
    self.use_GPU = True
    self.use_cycles = True  # cycles or eevee
    self.use_cycles_denoising = False
    self.use_adaptive_sampling = True
    self.resolution_x = 640  # pixel resolution
    self.resolution_y = 480
    self.samples = 512  # render engine samples

    #  OUTPUT
    self.numberOfRenders = 3 # how many rendered examples

    # temporary variables (dont change anything here)
    self.metallic = []
    self.roughness = []
