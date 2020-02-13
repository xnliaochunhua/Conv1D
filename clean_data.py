from spectral import envi
import numpy as np
from random import shuffle
import os, sys 
from sklearn.preprocessing import MinMaxScaler
def clean_data (inputfile,trainingdata,validationdata,testingdata):
	"""Loading features and labels"""
	training = envi.open(trainingdata)
	training = training.open_memmap(writeable = True)
	training = training[:,:,0]
	testing = envi.open(testingdata)
	testing = testing.open_memmap(writeable = True)
	testing = testing[:,:,0]
	validation = envi.open(validationdata)
	validation = validation.open_memmap(writeable = True)
	validation = validation[:,:,0]	
	labels_training= training
	labels_testing= testing
	labels_validation= validation


	features = envi.open(inputfile+'.hdr')
	features = features.open_memmap(writeable = True)
	features= np.array(features)


	"""Flattening of the rows x columns x bands features array to a 2D rows x columns x row array. Flattening of rows x columns labels array to a 1D rows x columns array."""
	features =features.reshape((-1,features.shape[2]))  	
	
	labels_training = labels_training.reshape(-1)
	labels_testing = labels_testing.reshape(-1)
	labels_validation = labels_validation.reshape(-1)

	"""Remove all non-crop from features and labels_training. Turn to numpy arrays, delete existing array"""
	features_training_NN=[]
	labels_training_NN=[]
	for i, label in enumerate(labels_training):
		if label!=0:
			features_training_NN.append(features[i])
			labels_training_NN.append(label)
	features_training_NN=np.array(features_training_NN)
	labels_training_NN=np.array(labels_training_NN)
	

	"""Remove all non-crop from features and labels_testing. Turn to numpy arrays, delete existing array"""
	features_testing_NN=[]
	labels_testing_NN=[]
	for i, label in enumerate(labels_testing):
		if label!=0:
			features_testing_NN.append(features[i])
			labels_testing_NN.append(label)
	features_testing_NN=np.array(features_testing_NN)
	labels_testing_NN=np.array(labels_testing_NN)

	
	"""Remove all non-crop from features and labels_validation. Turn to numpy arrays, delete existing array"""
	features_validation_NN=[]
	labels_validation_NN=[]
	for i, label in enumerate(labels_validation):
		if label!=0:
			features_validation_NN.append(features[i])
			labels_validation_NN.append(label)
	features_validation_NN=np.array(features_validation_NN)
	labels_validation_NN=np.array(labels_validation_NN)


	"""Done to prevent lag"""
	del(features)
	del(labels_training)



	"""Must be typed manually in interpreter to save the data files"""

	np.save(inputfile+'_features_training', features_training_NN)
	np.save(inputfile+'_labels_training', labels_training_NN)
	np.save(inputfile+'_features_testing', features_testing_NN)
	np.save(inputfile+'_labels_testing', labels_testing_NN)
	np.save(inputfile+'_features_validation', features_validation_NN)
	np.save(inputfile+'_labels_validation', labels_validation_NN)
	
	





