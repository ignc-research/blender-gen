#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:17:43 2020

@author: leon
"""


import os
import json
import numpy as np
from config import cfg


def main(obj='phone'):
    
    path = './DATASET/{}/images'.format(obj)
    files = os.listdir(path)
    files.sort()

    # split data into test and train
    n = len(files)  # number of images
    start = 0
    end = round(n * 0.8)  # 80% 20% split
    
    train_range = np.zeros((end,1), dtype=int)
    
    # remove train and test files if they exist
    train_path = "./DATASET/{}/".format(obj) + "/train.txt"
    test_path = "./DATASET/{}/".format(obj) + "/test.txt"
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(test_path):
        os.remove(test_path)     
        
    # create training_range.txt and train.txt    
    for i in range(start, end):
        fname = files[i]
        train_range[i] = os.path.splitext(fname)[0]  # remove file extension
        with open(train_path, "a") as text_file:
            text_file.write(path[2:]+'/'+fname + '\n')
    np.savetxt('./DATASET/{}/'.format(obj) +"training_range.txt", train_range, fmt='%d')
    
    # create test.txt
    for i in range(end, n):
        fname = files[i]
        with open(test_path, "a") as text_file:
            text_file.write(path[2:]+'/'+fname + '\n')
            
            
    # create cfg/obj.data
    fpath = './cfg/{}.data'.format(obj)
    if os.path.exists(fpath):
        os.remove(fpath)
    with open(fpath, "a") as text_file:
        text_file.write('train = DATASET/{}/train.txt'.format(obj) + '\n')
        text_file.write('valid = DATASET/{}/test.txt'.format(obj) + '\n')
        text_file.write('backup = backup/{}'.format(obj)  + '\n')
        text_file.write('mesh = DATASET/{}/{}.ply'.format(obj,obj) + '\n')
        text_file.write('tr_range = DATASET/{}/training_range.txt'.format(obj) + '\n')
        text_file.write('name = {}'.format(obj) + '\n')
        text_file.write('diam = 0' + '\n')
        text_file.write('gpus = 0' + '\n')
        text_file.write('width = ' + str(cfg.resolution_x) + '\n')
        text_file.write('height = ' + str(cfg.resolution_y) + '\n')
        
        with open("camera_intrinsic.json","r") as f:
            cam = f.read()
        cam = json.loads(cam)
        
        text_file.write('fx = ' + str(cam['fx']) + '\n')
        text_file.write('fy = ' + str(cam['fy']) + '\n')
        text_file.write('u0 = ' + str(cam['cx']) + '\n')
        text_file.write('v0 = ' + str(cam['cy']) + '\n')






        
    
    
    
    
if __name__ == '__main__':
    main()