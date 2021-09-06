# python-blender-datagen
![Screenshot](/example.png)

## installation
Download and unpack Blender in a folder /path/to/blender/blender-2.xx.x-linux64/ from https://www.blender.org/download/
To bind the 'blender' command to the blender application execute the following command in the terminal:
```
sudo ln -s /full/path/to/blender/blender-2.xx.x-linux64/blender /usr/local/bin/blender
```

### background images
If you want to use random COCO background images, download the COCO dataset (http://images.cocodataset.org/zips/train2017.zip) and unzip the images into the folder ./bg/coco
If you want to use HDRI 360° environment maps, you can download them (e.g. from https://polyhaven.com/hdris) and put them in the ./environment folder
If you want to use real deployment background images, put them in the ./bg/real folder
If you want to use random textures (e.g. from https://polyhaven.com/textures), put the images in the ./distractor_textures and ./object_textures folders


## usage

### render images
execute the following command in the terminal:
```
blender --background --python main.py
```

### show annotations (bounding box)
After rendering images, execute the following command in the terminal:
```
python show_annotations.py
```

### render image and open blender scene
To check the Blender scene setup, especially to configure the relationship between camera and object it is helpful to open the Blender scene after rendering.
1. set the test flag in the config.py file to True
1. start blender with the command line:
```
blender --python main.py
```


## config.py
This python file contains a simple configuration class to configure the Blender generation script. The following parameters must be adapted to your specific application.

Parameter | Description
--------- | -----------
seed | Initialize the random number generator. Set to an integer or None.
test | Boolean test flag. If you set this to True, Blender will only render one image and not delete light data and background image data. Use this together with blender _--python main.py_ to see the Blender scene setup.
out_folder | Output folder. Rendered images will be saved to DATASET/out_folder
bg_paths | List of paths to background images. Use multiple paths to mix different datasets in different ratios.
environment_paths | List of paths to environment images (360° HDRIs). Use multiple paths to mix different datasets in different ratios.
model_paths | List of paths to 3D CAD models.
compute_bbox | Choose _'tight'_ or _'fast'_. _Tight_ uses all vertices to compute a tight bbox but it is slower. _Fast_ uses only the 3D Bounding Box corners.
distractor_paths | List of paths to distracting foreground objects
max_distractor_objects | Integer. Maximum number of  distracting foreground objects
distractor_texture_path | String pointing to the textures folder for distracting foreground objects
object_texture_path | String pointing to the textures folder for the 3D model that we want to detect
use_bg_image | Boolean. Use background images (and not HDRI images) from the bg_paths folder
use_environment_maps | Boolean. Use 360° HDRI images from the environment_paths folder. If use_bg_image is also True, only the HDRI lighting will be used.
emission_min | HDRI minimum emission strength
emission_max | HDRI maximum emission strength
light_number_min | Minimum number of Point Lights
light_number_max | Maximum number of Point Lights
light_energymin | Minimum Energy of Point Lights [W]
light_energymax | Maximum Energy of Point Lights [W]
random_color | Choose either "None" or "temperature" for Point Lights. _None_ uses only white light, where _temperature_ will use the temperature colors from _util.py_
cam_rmin | minimum camera radial distance in spherical coordinate system
cam_rmax | maximum camera radial distance in spherical coordinate system
cam_incmin | minimum camera inclination in spherical coordinate system
cam_incmax | maximum camera inclination in spherical coordinate system
cam_azimin | minimum camera azimuth in spherical coordinate system
cam_azimax | maximum camera azimuth in spherical coordinate system
obj_location_xmin | minimum object (3D CAD model) offset in the x-axis in cartesian coordinate system
obj_location_xmax | maximum object (3D CAD model) offset in the x-axis in cartesian coordinate system
obj_location_ymin | minimum object (3D CAD model) offset in the y-axis in cartesian coordinate system
obj_location_ymax | maximum object (3D CAD model) offset in the y-axis in cartesian coordinate system
obj_location_zmin | minimum object (3D CAD model) offset in the z-axis in cartesian coordinate system
obj_location_zmax | maximum object (3D CAD model) offset in the z-axis in cartesian coordinate system
cam_rotation_min | minimum XYZ euler rotation angle of the constrained camera in radians
cam_rotation_max | maximum XYZ euler rotation angle of the constrained camera in radians
max_boundingbox | filters out blender scenes where the bbox of the 3D CAD Model is outside of the image to a certain threshold. A value of 0.1 means max. 10% occlusion
cam_lens_unit | Choose either 'FOV' or 'MILLIMETERS' (https://docs.blender.org/api/current/bpy.types.Camera.html#bpy.types.Camera.lens_unit)
cam_lens | Camera lens value in mm. This is used when 'MILLIMETERS' is the lens unit. https://docs.blender.org/manual/en/latest/render/cameras.html
cam_fov | Camera field of view in degrees. This is used when 'FOV' is the lens unit. https://docs.blender.org/manual/en/latest/render/cameras.html
cam_sensor_height | Vertical size of the image sensor area in millimeters (https://docs.blender.org/api/current/bpy.types.Camera.html)
cam_sensor_width | Horizontal size of the image sensor area in millimeters (https://docs.blender.org/api/current/bpy.types.Camera.html)
clip_end | Camera far clipping distance (https://docs.blender.org/api/current/bpy.types.Camera.html)
clip_start | Camera near clipping distance (https://docs.blender.org/api/current/bpy.types.Camera.html)
use_GPU | Boolean. If True, the GPU will be used for rendering
use_cycles | Boolean. If True, cycles will be used as rendering engine. If False, Eevee will be used
use_cycles_denoising | Boolean. If True, the rendered images are denoised afterwards (https://docs.blender.org/manual/en/latest/render/cycles/render_settings/sampling.html#denoising)
use_adaptive_sampling | Boolean. If True, adaptive sampling is used (https://docs.blender.org/manual/en/latest/render/cycles/render_settings/sampling.html#adaptive-sampling)
resolution_x | Pixel resolution of the output image (width)
resolution_y | Pixel resolution of the output image (height)
samples | Render engine number of samples (sets cycles.samples)
numberOfRenders | Number of rendered images



### Getting Started with your own data
If you want to use the default settings that were used in the paper, you can only change the following parameters:
1. Place your 3D CAD model as an *.OBJ file with the material *.mtl file in the ./models folder. Blender can be used to convert to obj-format and create a mtl file. Make sure the model_paths parameter in the config file points to your object file.
1. Place random background images in the ./bg folder.
1. Place random HDRI environment images in the ./environment folder.
1. Place random texture images in the ./distractor_textures and ./object_textures folders.
1. If you don't have a compatible GPU, set use_GPU to False in the config.py file.
1. Set the test parameter in the config.py file to True, render one image and inspect the Blender scene.
1. Change the camera parameters cam_* as needed in the config.py file.
1. Set the test parameter to False and set numberOfRenders to the desired number of images. Start the rendering process.




