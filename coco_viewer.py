#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:29:04 2020

@author: leon
"""

from pycocotools.coco import COCO # pip install pycocotools
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
coco=COCO('DATASET/Frames_rgb/annotation_coco_650.json')
#coco=COCO('DATASET/annotation_coco.json')
catIds = 0
imgIds = coco.getImgIds(catIds=catIds)
idx = imgIds[np.random.randint(0,len(imgIds))]
img = coco.loadImgs(idx)[0]
print(img)
#I = io.imread('DATASET/'+img['file_name']) # img_prefix = DATASET/
I = io.imread('DATASET/Frames_rgb/'+img['file_name']) # img_prefix = DATASET/Frames_rgb
fig,ax = plt.subplots(1)
ax.imshow(I)
plt.axis('off')

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
anns = coco.loadAnns(annIds)

for i in anns:
    [x,y,w,h] = i['bbox']
    # Create a Rectangle patch
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    #keypoints = i['keypoints']
    #for k in keypoints:
    #    circle = patches.Circle((k[0],k[1]),5,edgecolor='None',facecolor='r')
    #    ax.add_patch(circle)
    # Add the patch to the Axes
    ax.add_patch(rect)
#coco.showAnns(anns)
plt.show()
