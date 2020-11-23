import os
import random
from PIL import Image
import numpy as np
from image import *


hue = 0.025
saturation = 1.5
exposure = 1.5

factor = 1



counter = len(os.listdir('./JPEGImages/'))
for file in sorted(os.listdir('./JPEGImages/')):
	for i in range(factor):
		jitter = (random.random() * 0.6) - 0.4
		# Get the data augmented image and their corresponding labels
		img, p = load_data_detection('./JPEGImages/' + file, (640,480), jitter, hue, saturation, exposure)
		img.save('./JPEGImages/' + format(counter, '06') + '.png', "PNG")
		f = open(os.path.join('./labels/' + format(counter, '06') + '.txt'), "w+")
		f.write("0 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f" % (p[1],p[2],       p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20]))
		f.close()
		print(p)
		counter+=1