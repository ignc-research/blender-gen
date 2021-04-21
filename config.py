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
    self.bg_paths = ['./bg']
    self.environment_paths = ['./environment']
    #self.model_paths = ['./models/H8000.obj', './models/4000F_2.obj'] #3dbox0922marker_new.ply'  # list of filepath to objects
    #self.model_paths = ['./models/3dbox0922marker_new.ply']  # filepath to object
    self.model_paths = ['./models/3dbox0922marker_new.ply']
    #self.model_paths = ['./models/Pepper_Robot_Custom_Nonglossy_Nonshiny.obj']
    #self.distractor_paths = ['./distractors/048_hammer', './distractors/035_power_drill', './distractors/037_scissors', 
    #'./distractors/043_phillips_screwdriver', './distractors/025_mug', './distractors/036_wood_block', './distractors/044_flat_screwdriver']
    self.distractor_paths = []

    #self.NumberOfObjects = 1


    # DEPTH OUTPUT
    self.output_depth = False
    self.depth_color_depth = '16'

    # AUGMENTATION
    self.use_bg_image = True # use Background Images
    self.use_environment_maps = True  # use 360° HDRI Panoramas
    self.emission_min = 0.5 # only for environment maps
    self.emission_max = 2 # only for environment maps
    self.light_number_min = 1 # only for background images
    self.light_number_max = 3 # only for background images
    self.light_energymin = 20 # only for background images
    self.light_energymax = 100 # only for background images
    self.random_hsv_value = False # randomize the value of HSV color space of the object with p=0.5
    self.random_metallic_value = False # randomize the metallic object value with p=0.5
    self.random_roughness_value = False # randomize the roughness object value with p=0.5
    self.random_projector_colors = False # random projector augmentation with p=0.5 (point light with random color)

    # OBJECT COLOR (for PLY Files)
    self.model_scale = 1  # model scale for PLY objects
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
    self.cam_incmax = pi/2#pi*2/3
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
    self.obj_location_xmin = -0.1 # translation in meters
    self.obj_location_xmax = 0.1
    self.obj_location_ymin = -0.1
    self.obj_location_ymax = 0.1
    self.obj_location_zmin = -0.1
    self.obj_location_zmax = 0.1
    self.cam_rotation_min = 0
    self.cam_rotation_max = 2*pi

    self.max_distractor_objects = 0

    #self.obj_rotation_xmin = 0#-90 # euler rotation in degrees
    #self.obj_rotation_xmax = 0#90
    #self.obj_rotation_ymin = 0#-90
    #self.obj_rotation_ymax = 0#90
    #self.obj_rotation_zmin = 0
    #self.obj_rotation_zmax = 0


    self.max_boundingbox = 0.1 # filter out objects with bbox < -x or > 1+x

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
    self.resolution_y = 360
    self.samples = 512  # render engine samples

    #  OUTPUT
    self.numberOfRenders = 10# how many rendered examples

    # temporary variables (dont change anything here)
    self.metallic = []
    self.roughness = []
