import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
import csv
from clean_data import clean_data
from MLP import MLP
from CNN1d import CNN1d
from rf import RF
from XGBoost import XGBoost
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# features=['Venus2','RS2','Venus2-RS2','Venus2-RS2-MNF']
# features=['FD', 'CP', 'Venus-Pauli2', 'Venus-Pauli2-MNF']
features=['Venus2-RS2']
for feature in features:
	# trainingdatalist=['data/Training1-3.hdr','data/Training2-4.hdr','data/Training3-5.hdr','data/Training4-1.hdr','data/Training5-2.hdr']
	trainingdatalist=['data/Training1-3.hdr']
	for trainingdata in trainingdatalist:
		if trainingdata=='data/Training1-3.hdr':
			validationdata='data/Testing4.hdr'
			testingdata='data/Testing5.hdr'	
		elif trainingdata=='data/Training2-4.hdr':
			validationdata='data/Testing5.hdr'
			testingdata='data/Testing1.hdr'	
		elif trainingdata=='data/Training3-5.hdr':
			validationdata='data/Testing1.hdr'
			testingdata='data/Testing2.hdr'	
		elif trainingdata=='data/Training4-1.hdr':
			validationdata='data/Testing2.hdr'
			testingdata='data/Testing3.hdr'	
		elif trainingdata=='data/Training5-2.hdr':
			validationdata='data/Testing3.hdr'
			testingdata='data/Testing4.hdr'	
		print ("Clean data...")

		"""clean data"""
		inputfile='data/'+feature
	
		clean_data(inputfile,trainingdata,validationdata,testingdata)
		modelname=feature+'_crossvalidation5-test'
		# modelname='Venus2-RS2-MNF-test'


		"""Loading data"""


		features_training=np.load(inputfile+'_features_training'+".npy")
		labels_training=np.load(inputfile+'_labels_training'+".npy")
		features_testing=np.load(inputfile+'_features_testing'+".npy")
		labels_testing=np.load(inputfile+'_labels_testing'+".npy")
		features_validation=np.load(inputfile+'_features_validation'+".npy")
		labels_validation=np.load(inputfile+'_labels_validation'+".npy")




		"""Shuffling training"""
		indices = np.arange(features_training.shape[0])
		np.random.shuffle(indices)
		features_training = features_training[indices]
		labels_training = labels_training[indices]

		"""Shuffling testing"""
		indices = np.arange(features_testing.shape[0])
		np.random.shuffle(indices)
		features_testing = features_testing[indices]
		labels_testing = labels_testing[indices]
		
		# model=MLP(features_training,labels_training,features_testing,labels_testing, features_validation,labels_validation, modelname)
		model=CNN1d(features_training,labels_training,features_testing,labels_testing,features_validation,labels_validation,modelname)
		# model=RF(features_training,labels_training,features_testing,labels_testing,modelname)
		# model=XGBoost(features_training,labels_training,features_testing,labels_testing,modelname)


	