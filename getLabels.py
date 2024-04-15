#!/usr/bin/env python3

###############################################################
##Get the labels for k means regression 		           ####
##Uses parts of the original script getLabels.py by ebellis####
###############################################################

import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

###Main part of the code

expFile = sys.argv[1]
imagePath = sys.argv[2]
labelsPath = sys.argv[3]

### get a list of genes of interest
imageList = glob.glob(imagePath + "*.npy")
geneList = []
for image in imageList:
	gene = image.replace(imagePath, "")   # remove path info
	gene = gene.replace("_",".").replace(".npy","")
	geneList.append(gene) 
print(geneList[0:5])
print(len(geneList))

### read in DEG file and add information to the dictionary
geneFile = open(expFile, 'r')

Col_0 = {}
for line in geneFile:
	line2 = line.split(',')
	expr = line2[2]
	gene = line2[0]
	try:
		if expr == 'up\n':
			Col_0[gene] = 0
		elif expr == 'down\n':
			Col_0[gene] = 1
		elif expr == 'no\n':
			Col_0[gene] = 2
	except KeyError:
		print(gene)
geneFile.close()
print(len(geneList))
print(len(Col_0))

### traverse goi list,
### look up label for gene in dictionary
### print to outfile (np array)
for gene in geneList:
	label = np.array([Col_0[gene]])
	file_name = f"{gene.replace('.','_')}.npy"
	np.save(labelsPath + file_name, label)
	
sys.exit("Live long and prosper!")