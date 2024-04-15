#!/usr/bin/env python3

import tensorflow as tf
import glob
import numpy as np
import math
import random
import itertools
import tensorflow.keras as keras
from keras.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Reshape, Conv2D, Activation, MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score , precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from keras.utils.vis_utils import plot_model
from keras import backend as K 
from keras.models import Sequential,Model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from random import choices
import time
import sys
import gc
import os
import multiprocessing 


start_time = time.time()

#########################################################################
#####Test hyperparameters to build a model                           ####
#####Reuses parts from trainModel.py & oneHotEncoding.py by ebellis  ####
#########################################################################

######Functions######
#function to build a model with specific hyperparameters.
#modified from Washburn et al 2020 paper/code.
def build(DNA_length,hp):
	hp1= []
	for x in hp:
		try:
			hp1.append(int(x))
		except ValueError:
			try:
				hp1.append(float(x))
			except ValueError:	
				hp1.append(None)
	conv1_filters,conv2_filters,conv3_filters,conv_width,pool_width,pool_stride,dropout,dense1_units,dense2_units,conv_layers,dense_layers=hp1
	model=Sequential()

	model.add(Conv2D(conv1_filters,kernel_size=(4,conv_width),padding='valid',input_shape=[4,DNA_length,1]))
	model.add(Activation('sigmoid'))
	model.add(Conv2D(conv1_filters,kernel_size=(1,conv_width),padding='same'))
	model.add(Activation('sigmoid'))
	model.add(MaxPooling2D(pool_size=(1,pool_width),strides=(1,pool_stride),padding='same'))
	model.add(Dropout(dropout))

	if conv_layers>=2:
		model.add(Conv2D(conv2_filters,kernel_size=(1,conv_width),padding='same'))
		model.add(Activation('sigmoid'))
		model.add(Conv2D(conv2_filters,kernel_size=(1,conv_width),padding='same'))
		model.add(Activation('sigmoid'))
		model.add(MaxPooling2D(pool_size=(1,pool_width),strides=(1,pool_stride),padding='same'))
		model.add(Dropout(dropout))

	elif conv_layers>=3:
		model.add(Conv2D(conv3_filters,kernel_size=(1,conv_width),padding='same'))
		model.add(Activation('sigmoid'))
		model.add(Conv2D(conv3_filters,kernel_size=(1,conv_width),padding='same'))
		model.add(Activation('sigmoid'))
		model.add(MaxPooling2D(pool_size=(1,pool_width),strides=(1,pool_stride),padding='same'))
		model.add(Dropout(dropout))

	model.add(Flatten())
	model.add(Dense(dense1_units))
	model.add(Activation('sigmoid'))
	model.add(Dropout(dropout))

	if dense_layers>=3:
		model.add(Dense(dense2_units))
		model.add(Activation('sigmoid'))
	
	model.add(Dense(2))
	model.add(Activation('sigmoid'))
	return model

## need to convert np array from 3002 bp long x 1 x 5 different bases to 4 bases x Y bp long x 1 
# input: .npy file name including path
# returns: a 4 x Y x 1 array or as long as necessary the number of snps
def convert_array(in_arr):
	inArr = np.load(in_arr)
	inArr = inArr[500:2500] #keep only the -1000bp and +1000bp 
	inArr = np.swapaxes(inArr[:,0:4],0,1)
	#inArr = np.swapaxes(np.delete(inArr, 4, 1),0,1)
	inArr = inArr[:,:,np.newaxis]
	return inArr

###Function to plot ROC for each class
def plot_roc_curve(fpr,tpr, figname): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.savefig(figname)    
  
 
#train models for different combinations of parameter
def compare_models(length, hp, seqs, seqs_te, labs, labs_te, c1, c2 , shuffle, rate, itera):
	os.mkdir("/storage/home/mzt5590/scratch/GxEmodels/models/model{}".format("_".join([str(x) for x in hp])))
	callback = tf.keras.callbacks.ModelCheckpoint(
		filepath="/storage/home/mzt5590/scratch/GxEmodels/models/model{}".format("_".join([str(x) for x in hp])),
		save_weights_only=False,
		monitor='loss',
		mode='min',
		save_best_only=True)

	model=build(length, hp)
	#compile the model
	optimiser = keras.optimizers.SGD(learning_rate=rate)
	model.compile(optimizer=optimiser, loss=keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.AUC(curve='PR')])

	#fit the model
	#default batch size is 258, using the recommendation of the keras guideline of bigger one. Increases the chance of having both classes in the batch
	history = model.fit(x = seqs, y = labs, batch_size = 2048, epochs = 15, verbose = 2, validation_split = 0.15, callbacks=callback)
	
	#plot the loss
	##Plot the history
	plt.plot(history.history['loss'], label='loss')
	plt.plot(history.history['val_loss'], label='val_loss')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend()
	plt.grid(True)
	plt.savefig("/storage/home/mzt5590/GxEmodels/scratch/figures/train_model_oversampling_{}.png".format("_".join([str(x) for x in hp])))
	
	#reload the best saved model
	model = keras.models.load_model("/storage/home/mzt5590/scratch/GxEmodels/models/model{}".format("_".join([str(x) for x in hp])))
	
	##predictions
	predictions = model.predict(seqs_te)
	results = model.evaluate(seqs_te, labs_te, batch_size=256)
	
	##Get the most probable class. Based only on the accuracy not sth elses
	pred = np.argmax(predictions, axis=1)
	real=np.argmax(labs_te,axis=1)
	deg_preds = predictions[:,0]
	no_preds = predictions[:,1] #pick the probs for each class seperately

	###AUC STUFF HERE
	##Compute the TPR and FPR for thresholds. 
	#functions takes the actual values (labels) and the predictions for the labels)
	#see also https://sinyi-chou.github.io/python-sklearn-precision-recall/

	precision, recall, thresholds = roc_curve(real, deg_preds)
	plot_roc_curve(precision, recall)

	precision, recall, thresholds = roc_curve(real, no_preds)
	plot_roc_curve(precision, recall, "/storage/home/mzt5590/scratch/GxEmodels/aurocs/roc_class_{}.pdf".format("_".join([str(x) for x in hp])))
	
	##Estimate the AUC of ROC PR
	#auc_roc = roc_auc_score(precision, recall)
	auc_roc = roc_auc_score(labs_te, predictions, multi_class="ovr", average="micro")
	print("AUC of precision ROC: ", auc_roc)
	
	#plot the AUC ROC One vs All
	RocCurveDisplay.from_predictions(
		labs_te.ravel(),
		predictions.ravel(),
		name="micro-average OvR",
		color="darkorange",
	)
	plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
	plt.axis("square")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
	plt.legend()
	plt.savefig("/storage/home/mzt5590/scratch/GxEmodels/aurocs/roc_{}.pdf".format("_".join([str(x) for x in hp])))
	
	#Estimate the percentage of correct assignments per class
	i = 0 
	c01 = 0
	c11 = 0
	while i < len(pred):
		if pred[i] == real[i]:
			if pred[i] == 0: #correct cluster 0
				c01 = c01 + 1 
			elif pred[i] == 1: #correct cluster 1
				c11 = c11 + 1
		i = i + 1
	cR_c = c01/cR
	cN_c = c11/cN
	print(cR_c, cN_c)
	
	#write to file
	with open('/storage/home/mzt5590/scratch/GxEmodels/compare_models_{}'.format(typ), "a") as result_file:
		result_file.write("\t".join([str(x) for x in hp]))
		result_file.write("\t")
		result_file.write("\t".join([shuffle,str(rate),str(results[0]),str(results[1]), str(cR_c), str(cN_c), str(auc_roc), str(history['auc'][-1]), str(history['loss'][-1]), str(history['auc_val'][-1]), str(history['loss_val'][-1]), str(len(history['auc'][-1])), str(history['loss'].index(min(history['loss'])) + 1)]))
		result_file.write("\n")
	del model
	K.clear_session()

####Main Part of the code
typ = sys.argv[1] #whether it is up vs no or down vs no
hyper = sys.argv[2] #give a table with the hyperparameters. 
itera = sys.argv[3] #give which one is actually loaded 

##Check how many processes are used
N_CPU = multiprocessing.cpu_count()
N_GPU = tf.config.experimental.list_physical_devices('GPU')
print('CPUs number:', N_CPU)
print('GPUs number:', N_GPU)
print(tf.__version__)

##load files, data etc
## get lists of x and y values for the training set
xList = glob.glob('/storage/home/mzt5590/scratch/GxEmodels/Encoded/*.npy')
yList = glob.glob('/storage/home/mzt5590/scratch/GxEmodels/Labels/*.npy')
seqs2 = []
ylabs = []
ylabs2 = []
gr1 = []
gr2 = []

print("Reading in data...")
if len(xList) == len(yList):
	# for images
	for gene in xList:
		seqs2.append(convert_array(gene))

	print("Seqs done:",len(seqs2))
	# for labels
	ind = []
	i = 0
	if typ == 'up': #Pick which classes I want to compare from the labels
		for exp in yList:
			x = int(np.load(exp))
			if x == 0: #upDE genes
				ylabs.append([1,0]) #for classification it needs one hot encoding of the values
				ylabs2.append(0)
				gr1.append(i) #to use for oversampling
			elif x == 1: #downDE
				ylabs.append([0,0])
				ylabs2.append(2)
				ind.append(i)
			elif x == 2: #noDE genes
				ylabs.append([0,1]) 
				ylabs2.append(1)
				gr2.append(i) #collect the indexes for rearranging the seqs when I oversample and randomize
			i = i + 1
			#print(x)
	if typ == 'down':
		for exp in yList:
			x = int(np.load(exp))
			if x == 1: #downDE
				ylabs.append([1,0]) 
				ylabs2.append(0)
				gr1.append(i) 
			elif x == 0: #upDE
				ylabs.append([0,0])
				ylabs2.append(2)
				ind.append(i)
			elif x == 2: #noDE genes
				ylabs.append([0,1])
				ylabs2.append(1)
				gr2.append(i)
			i = i + 1 
	if typ == 'both':
		for exp in yList:
			x = int(np.load(exp))
			if x == 1: #noDE use them here 
				ylabs.append([1,0]) 
				ylabs2.append(0)
				gr1.append(i) 
			elif x == 0: #upDE use them also here
				ylabs.append([1,0])
				ylabs2.append(0)
				gr1.append(i) 
				ind.append(i)
			elif x == 2: 
				ylabs.append([0,1])
				ylabs2.append(1)
				gr2.append(i)
			i = i + 1 

else:
	sys.exit("Error: number of image files does not match number of label files.")

print("All data are read in. Oversampling...")	

##Oversampling the indexes of the 'up/down/both' group! 
overin = choices(gr1, k=int(len(gr2))) #get the new group randomising with replacing, so that it is same length as the noDE group
newlabs = overin + gr2 
print("Number of labels in original upDE, noDE and total after oversampling:", len(gr1), len(gr2), len(newlabs))

#Randomize the list above
random.shuffle(newlabs) #will use the randomized indexes of the labels/seq as read in to then put them in the correct order

##make the new labels and order the sequences based on where the original labels are::
labs = []
seqs = []
labs2 = []
for ind in newlabs:
	labs2.append(ylabs2[ind])
	seqs.append(seqs2[ind])
	labs.append(ylabs[ind])

print("Train labels:",len(labs))
print("Train seqs:", len(seqs))

seqs = np.stack(seqs, axis=0)    # samples x 4 x 3002 x 1
labs2 = np.stack(labs2, axis = 0).squeeze()   # samples x 1

del ylabs2
del ylabs
del xList
del yList

##Open the test set data
xList = glob.glob('/storage/home/mzt5590/scratch/GxEmodels/Test/Encoded/*.npy')
yList = glob.glob('/storage/home/mzt5590/scratch/GxEmodels/Test/Labels/*.npy')
seqs_te = []
labs_te = [] #for one hot encoding

print("Reading in testing data...")
if len(xList) == len(yList):
	# for images
	for gene in xList:
		seqs_te.append(convert_array(gene))

	# for labels
	cR = 0 #count how many genes in category of regulation
	cN = 0 #count how many genes in no regulation
	ind = []
	i = 0
	if typ == 'up':
		for exp in yList:
			x = int(np.load(exp))
			if x == 0:
				cR = cR + 1
				labs_te.append([1,0]) 
			elif x == 1:
				ind.append(i)
			elif x == 2:
				cN = cN + 1
				labs_te.append([0,1])
			i = i + 1 
	if typ == 'down':
		for exp in yList:
			x = int(np.load(exp))
			if x == 1:
				cR = cR + 1 
				labs_te.append([1,0]) 
			elif x == 0:
				ind.append(i)
			elif x == 2:
				cN = cN + 1
				labs_te.append([0,1])
			i = i + 1 

	labs_te = np.stack(labs_te, axis = 0).squeeze()   # samples x 1
	
else:
	print("Error: number of image files does not match number of label files.") 

##Delete the seqs from the labels that are not in the groups we trained for.
if len(ind) > 0:
	for index in sorted(ind, reverse=True):
		del seqs_te[index]	

print("Test labels:",len(labs_te))
print("Test seqs:",len(seqs_te))

seqs_te = np.stack(seqs_te, axis=0)    # samples x 4 x 3002 x 1

#Class weights:
print("No class weights implimented.")

##List of hyperparameters and layers to test
#check if a list of hyperparameters has already been tested and remove it from the list
priors = []
try:
	with open("/storage/home/mzt5590/scratch/GxEmodels/compare_models_{}".format(typ), 'r') as priort:
		for line in priort:
			if line[0] == 'c':
				pass
			else:
				line2 = line.split('\t')
				line2 = line2[0:11]
				priors.append(line2)
except FileNotFoundError:
	print("No prior grid search result file found.")
	
# hyperparameters to test from a file
hyper = open('/storage/home/mzt5590/scratch/GxEmodels/{}'.format(hyper),'r')
hps = []
for line in hyper:
	if line[0] == 'c': #this is the header
		pass 
	else:
		line2 = line.split(",")
		x = line2[-1]
		line2[-1] = x[0:-1]
		if len(priors) > 0:
			if line2 in priors:
				pass
			else:
				hps.append(line2)
		else:
			hps.append(line2)

print("Hyperparameter combinations to test: {}".format(len(hps)))

###Build and train models
# Write the header line to file
try:
	with open("/storage/home/mzt5590/scratch/GxEmodels/compare_models_{}".format(typ), 'r') as priort:
		print("File to write exists.")
except FileNotFoundError:
	with open('/storage/home/mzt5590/scratch/GxEmodels/compare_models_{}'.format(typ), "a") as result_file:
				result_file.write("\t".join(['conv1_filters','conv2_filters','conv3_filters','conv_width','pool_width','pool_stride','dropout','dense1_units','dense2_units','conv_layers','dense_layers']))
				result_file.write("\t")
				result_file.write("\t".join(['shuffle','learning_rate','loss','accuracy','cluster0', 'cluster1', 'auc_test', 'auc_model', 'loss_model', 'auc_val_model', 'loss_val_model', 'epochs', 'best_epoch']))
				result_file.write("\n")

# Start training models
print("Start evaluation of {} models".format(len(hps)))
for hp in hps:
	print("Evaluating: {}".format("_".join([str(x) for x in hp])))
	compare_models(2000, hp, seqs, seqs_te, labs, labs_te, cR, cN , 'None', 0.0001, itera)
	
K.clear_session()
gc.collect()
sys.exit("--- %s seconds ---" % (time.time() - start_time))
