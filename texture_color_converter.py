import argparse
import os
import cv2
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='path to folder with textures')
    parser.add_argument('--output_folder', help='path to output folder with new textures')
    args = parser.parse_args()

    # Parameters
    H = [220, 260]  # hue range
    V = -70         # value offset
    MAX_V = 128     # maximum value
    MIN_V = 0       # minimum value

    dir_ = os.listdir(args.input_folder)

    for file in dir_:
        print(args.input_folder+'/'+file)
        image = cv2.imread(args.input_folder+'/'+file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h,w,c = image.shape
        image[:,:,0] = np.random.randint(H[0], H[1], size=(h,w))/2
        image[:,:,1] = 255  # constant saturation
        image[:,:,2] = image[:,:,2] + V  # increase or decrease pixel-wise value
        image[image[:,:,2] < MIN_V, 2]  = MIN_V  # set minimum value
        image[image[:,:,2] > MAX_V, 2]  = MAX_V  # set maximum value
        image[image > 255] = 255
        image[image < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(args.output_folder+'/'+file, image)


if __name__ == '__main__':
    main()
