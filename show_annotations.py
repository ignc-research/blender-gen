#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import random
import json

with open('DATASET/object/annotations/instances_default.json') as f:
  data = json.load(f)

images = data['images']
labels = data['annotations']
while(True):
    idx = random.randint(0,len(images)-1)
    img_name = images[idx]['file_name']
    print('showing annotation for img: ' + img_name)
    print('close the window to see the next label')

    bbox = labels[idx]['bbox']

    I = mpimg.imread('DATASET/object/images/'+img_name)  # load rendered image

    fig,ax = plt.subplots(1)
    ax.imshow(I)
    plt.axis('off')
    [x,y,w,h] = bbox
    rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='g',facecolor='none')  # add bounding box annotation
    ax.add_patch(rect)
    
    vis_keypoints = False
    if (vis_keypoints):
      keypoints = labels[idx]['keypoints']
      for kp in keypoints:
        x = kp[0]
        y = kp[1]
        circle = patches.Circle((x,y), radius=3)
        ax.add_patch(circle)
    plt.show()
    

