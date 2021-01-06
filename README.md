# python-blender-datagen

## installation
Download and unpack Blender in a folder /path/to/blender/blender-2.xx.x-linux64/ from https://www.blender.org/download/
To bind the 'blender' command to the blender application execute the command 
```
sudo ln -s /full/path/to/blender/blender-2.xx.x-linux64/blender /usr/local/bin/blender
```
### installation of additional pip packages
To install pip for Blenders python environment (https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py):

```
cd ~
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
cd path/to/blender/blender-2.xx.x-linux64/2.90/python/bin/
./python3.7m ~/get-pip.py
```

Then install the necessary pip packages. OpenCV is only required with the --test command to view bounding boxes.
In the python/bin directory execute the following commands:
```
./python3.7m -m pip install opencv-python
```

### background images
If you want to use random COCO background images, download the COCO dataset (e.g. http://images.cocodataset.org/zips/train2017.zip) and unzip the images into the folder ./bg
If you want to use HDRI 360° environment maps, you can download them e.g. from https://hdrihaven.com/hdris/ and put them in the ./environment folder


## usage

### render images
blender --background --python main.py

### render images and open blender scene
blender --python main.py

### render image and view bounding box with OpenCV
blender --background --python main.py --test

## example:
![Screenshot](/example.png)

## labels
* COCO labels: https://cocodataset.org/#format-data
* 6D Pose labels: https://github.com/ignc-research/Industrial6DPoseEstimation and https://github.com/microsoft/singleshotpose
