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
        self.seed = 1  # random seed for image generation. use None or an integer
        self.test = False

        #  PATHS
        self.out_folder = 'dataset'  # render images will be saved to DATASET/out_folder
        #self.bg_paths = ['./bg/real', './bg/coco/train2017', './bg/coco/train2017']
        self.bg_paths = ['./bg']
        self.environment_paths = ['./environment']
        # self.model_paths = ['./models/H8000.obj', './models/4000F_2.obj'] #3dbox0922marker_new.ply'  # list of filepath to objects
        # self.model_paths = ['./models/3dbox0922marker_new.ply']  # filepath to object
        self.model_paths = ['/data/object.ply']
        self.compute_bbox = 'fast'  # choose 'tight' or 'fast' (tight uses all vertices to compute a tight bbox but it is slower)
        #self.distractor_paths = ['./distractors/048_hammer', './distractors/035_power_drill', './distractors/037_scissors',
        # './distractors/043_phillips_screwdriver', './distractors/025_mug', './distractors/036_wood_block', './distractors/044_flat_screwdriver']
        self.distractor_paths = ['./distractors/Cube', './distractors/Cube', './distractors/Cube']
        self.max_distractor_objects = 3

        self.object_texture_path = './object_textures' #'./environment'# './textures_realistic' #'./textures'
        self.distractor_texture_path = './distractor_textures'#'./bg/coco/train2017'

        #self.NumberOfObjects = 1
        self.use_fps_keypoints = False # experimental feature for 6d pose estimation


        # DEPTH OUTPUT (not tested)
        self.output_depth = False
        self.depth_color_depth = '16'

        # AUGMENTATION
        self.use_bg_image = True  # use Background Images
        self.use_environment_maps = True  # use 360° HDRI Panoramas
        self.emission_min = 1  # only for environment maps
        self.emission_max = 8  # only for environment maps
        self.light_number_min = 1  # only for background images
        self.light_number_max = 3  # only for background images
        self.light_energymin = 20  # only for background images
        self.light_energymax = 80  # only for background images
        self.random_hsv_value = False  # randomize the value of HSV color space of the object with p=0.5
        self.random_metallic_value = False  # randomize the metallic object value with p=0.5
        self.random_roughness_value = False  # randomize the roughness object value with p=0.5
        # self.random_projector_colors = False # random projector augmentation with p=0.5 (point light with random color)

        self.random_color = "None"  # choose "None", "temperature", "projector"

        # OBJECT COLOR (for PLY Files)
        self.model_scale = 0.5E-3  # model scale for PLY objects
        self.hsv_hue = 0.5  # changes hue of Hue Saturation Value Node, default 0.5
        self.hsv_saturation = 1  # changes saturation of Hue Saturation Value Node, default 1
        self.hsv_value = 1  # 0.35 # changes value of Hue Saturation Value Node, default 1
        # self.roughness  = 0.3#0.1 # Object Material Roughness (0=Mirror, 1=No Reflections)

        # camera sphere coordinates
        self.cam_rmin = 0.3  # minimum camera distance
        self.cam_rmax = 1.1  # maximum camera distance
        self.cam_incmin = 0
        self.cam_incmax = pi/2  # pi*2/3
        self.cam_azimin = 0
        self.cam_azimax = 2*pi


        #  OBJECT POSITION
        self.obj_location_xmin = -0.2  # translation in meters
        self.obj_location_xmax = 0.2
        self.obj_location_ymin = -0.2
        self.obj_location_ymax = 0.2
        self.obj_location_zmin = -0.2
        self.obj_location_zmax = 0.2
        self.cam_rotation_min = 0
        self.cam_rotation_max = 2*pi

  

        self.max_boundingbox = 0.2  # filter out objects with bbox < -x or > 1+x (a value of 0.1 means max. 10% occlusion)

        # Camera
        self.cam_lens_unit = 'FOV'  # Choose 'FOV' or 'MILLIMETERS'
        self.cam_lens = 4.7  # Camera lens value in mm
        self.cam_fov = (59+90)/2  # camera field of view in degrees
        self.cam_sensor_height = 3.84  # mm
        self.cam_sensor_width = 5.11  # mm

        self.clip_end = 50
        self.clip_start = 0.01

        #  RENDERING CONFIG
        self.use_GPU = True
        self.use_cycles = True  # cycles or eevee
        self.use_cycles_denoising = False
        self.use_adaptive_sampling = True
        self.resolution_x = 640  # pixel resolution
        self.resolution_y = 360
        self.samples = 256  # render engine samples

        #  OUTPUT
        self.numberOfRenders = 2  # how many rendered examples

        # temporary variables (dont change anything here)
        self.metallic = []
        self.roughness = []
