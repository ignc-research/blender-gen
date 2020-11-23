# Script to convert labels generate from annotate.photo to SSPE label format

import glob, os, sys
import json


def createLabelContent():
	print('creating labels and deleting pics where the obj is partly outside of frame')
	with open('annotation-results.6dan.json.json', 'r') as json_file:
		data = json.load(json_file)
		for lbl in data:
			p = lbl['sspd']
			f = open(os.path.join('./labels',lbl['image'].split('.')[0] + '.txt'), "w+")
			f.write("0 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f" % (p[1],p[2],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19]))
			f.close()


def createTestAndTrainFiles(counter):
	print('creating test files')

	f_test = open(os.path.join('./', 'test.txt'), "w+")
	for i in range(counter):
		img_type = ".jpg"
		f_test.write('./JPEGImages/' + format(i, '06') + img_type + " \n")
	
	f_test.close()








	
if __name__ == "__main__":
	createLabelContent()
	createTestAndTrainFiles(len(os.listdir('./labels')))