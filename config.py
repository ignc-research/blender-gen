#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi

class cfg:
    def __init__(self):
        self.seed = 1  # random seed for image generation. use None or an integer
        self.test = False

        #  PATHS
        self.out_folder = 'Suzanne'  # render images will be saved to DATASET/out_folder
        self.bg_paths = ['./bg']
        self.environment_paths = ['./environment']
        self.model_paths = ['./models/Suzanne.obj']
        self.compute_bbox = 'tight'  # choose 'tight' or 'fast' (tight uses all vertices to compute a tight bbox but it is slower)
        self.distractor_paths = ['./distractors/Cube', './distractors/Cube', './distractors/Cube']
        self.max_distractor_objects = 3

        self.object_texture_path = './object_textures'
        self.distractor_texture_path = './distractor_textures'

        # DEPTH OUTPUT (not tested)
        self.output_depth = False
        self.depth_color_depth = '16'

        # AUGMENTATION
        self.use_bg_image = True  # use Background Images
        self.use_environment_maps = True  # use 360° HDRI Panoramas
        self.emission_min = 1  # only for environment maps
        self.emission_max = 5  # only for environment maps
        self.light_number_min = 1  # only for background images
        self.light_number_max = 3  # only for background images
        self.light_energymin = 20  # only for background images
        self.light_energymax = 80  # only for background images
        self.random_metallic_value = False  # randomize the metallic object value with p=0.5
        self.random_roughness_value = False  # randomize the roughness object value with p=0.5
        self.random_color = "None"  # choose "None", "temperature", "projector"

        # OBJECT COLOR (for PLY Files)
        self.model_scale = 1  # model scale for PLY objects
        self.hsv_hue = 0.5  # changes hue of Hue Saturation Value Node, default 0.5
        self.hsv_saturation = 1  # changes saturation of Hue Saturation Value Node, default 1
        self.hsv_value = 1  # 0.35 # changes value of Hue Saturation Value Node, default 1

        # camera sphere coordinates
        self.cam_rmin = 0.3  # minimum camera distance
        self.cam_rmax = 1.1  # maximum camera distance
        self.cam_incmin = 0
        self.cam_incmax = pi/2
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
        self.max_boundingbox = 0.1  # filter out objects with bbox < -x or > 1+x (a value of 0.1 means max. 10% occlusion)

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
        self.samples = 512  # render engine samples

        #  OUTPUT
        self.number_of_renders = 10  # how many rendered examples
