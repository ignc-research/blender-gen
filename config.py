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
    self.model_path = './models/H8000.obj' #3dbox0922marker_new.ply'  # filepath to object
    #self.model_path = './models/3dbox0922marker_new.ply'  # filepath to object

    self.NumberOfObjects = 1

    self.model_scale = 0.5E-3  # model scale for PLY objects

    # DEPTH OUTPUT
    self.output_depth = False
    self.depth_color_depth = '16'

    # AUGMENTATION
    self.use_bg_image = False # use Background Images
    self.use_environment_maps = not self.use_bg_image  # use 360° HDRI Panoramas
    self.emission_min = 1 # only for environment maps
    self.emission_max = 5 # only for environment maps
    self.light_number = 1 # only for background images
    self.light_energymin = 60 # only for background images
    self.light_energymax = 1000 # only for background images
    self.random_hsv_value = True # randomize the value of HSV color space of the object with p=0.5
    self.random_metallic_value = False # randomize the metallic object value with p=0.5



    # OBJECT COLOR (for PLY Files)
    self.hsv_hue = 0.5 # changes hue of Hue Saturation Value Node, default 0.5
    self.hsv_saturation = 1 # changes saturation of Hue Saturation Value Node, default 1
    self.hsv_value = 1#0.35 # changes value of Hue Saturation Value Node, default 1
    self.roughness  = 0.3#0.1 # Object Material Roughness (0=Mirror, 1=No Reflections)


    #  CAMERA
    self.clip_end = 50
    self.clip_start = 0.01
    # camera sphere coordinates
    self.cam_rmin = 1  # minimum camera distance
    self.cam_rmax = 1.3  # maximum camera distance
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
    self.obj_location_xmin = -0.075 # translation in meters
    self.obj_location_xmax = -0.075
    self.obj_location_ymin = -0.075
    self.obj_location_ymax = 0.075
    self.obj_location_zmin = -0.075
    self.obj_location_zmax = 0.075

    self.obj_rotation_xmin = 0 # euler rotation in degrees
    self.obj_rotation_xmax = 0
    self.obj_rotation_ymin = 0
    self.obj_rotation_ymax = 0
    self.obj_rotation_zmin = 0
    self.obj_rotation_zmax = 0

    self.max_boundingbox = 0.05 # filter out objects with bbox < -x or > 1+x

    # camera intrinsics
    self.cam_lens = 4.7 # Camera lens value in mm
    self.cam_sensor_height = (2E-6 * 1520)*1000 #mm
    self.cam_sensor_width = (2E-6 * 2688)*1000 # mm
    #self.cam_distort = -0.001

    #  RENDERING
    self.use_GPU = True
    self.use_cycles = True  # cycles or eevee
    self.use_cycles_denoising = False
    self.resolution_x = 640  # pixel resolution
    self.resolution_y = 480
    self.samples = 256  # render engine samples

    #  OUTPUT
    self.numberOfRenders = 1 # how many rendered examples



    # temporary variables (dont change anything here)
    self.metallic = []
