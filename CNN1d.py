import tensorflow as tf
import numpy as np
from numpy import newaxis
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling2D, MaxPooling1D, ZeroPadding1D
from tensorflow.keras.models import Sequential

import csv
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import normalize
from osgeo import gdal, osr
from spectral import envi
import os, sys 
import time


def CNN1d(features_training,labels_training,features_testing,labels_testing,features_validation,labels_validation,modelname):
	"""convert the 2D numpy arrary to 3D arrary"""
	features_training = features_training.reshape(features_training.shape + (1,)) 
	features_testing = features_testing.reshape(features_testing.shape + (1,))
	features_validation = features_validation.reshape(features_validation.shape + (1,))
	nfeature=features_testing.shape[1]
	
	# """Empty the results file whenever we run TF"""
	# with open('CNN_accuracy_results.csv', 'w', newline='') as writeFile:
		# writer = csv.writer(writeFile)
		# writer.writerow(['Epoch', 'Testing Accuracy'])

	"""Check testing accuracy after each epoch, does this by looking at accuracy of first 20000 shuffled testing features and testing labels"""
	class MyCallBack(Callback):
		def on_epoch_end(self, epoch, logs=None):
			acc=accuracy_score(labels_testing[:20000], model.predict_classes(features_testing[:20000]))
			print("Testing accuracy:", acc)
			
			# with open('CNN_accuracy_results.csv', 'a', newline='') as writeFile:
				# writer = csv.writer(writeFile)
				# try:
					# writer.writerow([self.model.history.epoch[-1]+2,acc])
				# except:
					# writer.writerow([1, acc])
			# writeFile.close()
	cbk=MyCallBack()

	"""CNN model code"""
	print ("Training data...")
	
	model = tf.keras.models.Sequential()
	# model.add(ZeroPadding1D(1,input_shape=(nfeature,1)))
	# model.add(Conv1D(filters=64, kernel_size=3, activation='relu',data_format='channels_last'))
	# model.add(ZeroPadding1D(1))
	# model.add(Conv1D(64, 3, activation='relu',data_format='channels_last'))
	# model.add(MaxPooling1D(pool_size=2, strides=2))

	# model.add(ZeroPadding1D(1))
	# model.add(Conv1D(128, 3, activation='relu',data_format='channels_last'))
	# model.add(ZeroPadding1D(1))
	# model.add(Conv1D(128,3, activation='relu',data_format='channels_last'))
	# model.add(MaxPooling1D(pool_size=2, strides=2))

	model.add(ZeroPadding1D(1,input_shape=(nfeature,1)))
	model.add(Conv1D(512, 3, activation='relu',data_format='channels_last'))
	model.add(ZeroPadding1D(1))
	model.add(Conv1D(512,3, activation='relu',data_format='channels_last'))
	model.add(ZeroPadding1D(1))
	model.add(Conv1D(512,3, activation='relu',data_format='channels_last'))
	# model.add(MaxPooling1D(pool_size=2, strides=2))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(8, activation='softmax'))

	start=time.time()
				  
	model.compile(optimizer=Adam(lr=1.5e-4,),  # Good default optimizer to start with
				  loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
				  metrics=['accuracy'])  # what to track
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
	
	model.fit(features_training, labels_training,
					  epochs=20,
					  batch_size=3200,
					  # validation_split=0.2,	
					  validation_data=(features_validation,labels_validation),					  
					  callbacks=[es],
					  shuffle=True)  # train the model
	# # summarize history for accuracy
	# plt.plot(history.history['accuracy'])
	# plt.plot(history.history['val_accuracy'])
	# plt.title('model accuracy')
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'validation'], loc='upper left')
	# plt.show()
	# # summarize history for loss
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title('model loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'validation'], loc='upper left')
	# plt.show()
					  
	OA=accuracy_score(labels_testing, model.predict_classes(features_testing))
	Kappa=cohen_kappa_score(labels_testing, model.predict_classes(features_testing))
	array=confusion_matrix(labels_testing, model.predict_classes(features_testing))
	print ("Test Accuracy ", OA)
	print ("Confusion matrix ", array)
	model.save ('CNN1d_model_'+modelname+'.h5')
	end=time.time()
	print (end-start)
	t=end-start
	with open('CNN1d_accuracy_results_'+modelname+'.csv', 'a', newline='') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow([OA,Kappa,t])
		writeFile.close()
	"""Visualization of results ~ Confusion matrix ~ Labels_validation X Features_validation"""
	# array=normalize(confusion_matrix(model.history.validation_data[1], model.predict_classes(model.history.validation_data[0])))	
	# df_cm = pd.DataFrame(array, range(1,8),
					  # range(1,8))
	# sn.set(font_scale=1.4)  #for label size
	# sn.heatmap(df_cm, annot=True,annot_kws={"size": 10}, cmap='Blues')# font size
	# plt.title('Validation accuracy')
	# plt.xlabel('Predicted label', fontsize=16)
	# plt.ylabel('True label', fontsize=16)
	# plt.show()

	"""Visualization of results ~ Confusion matrix ~ Labels_testing X Features_testing"""

	
	df_cm = pd.DataFrame(array, range(1,8), range(1,8))
	df_cm.to_csv('CNN1d_ConfusionMatrix_'+modelname+'.csv')
	
	with open('CNN1d_accuracy_results_'+modelname+'.csv', 'a', newline='') as targetcsv:                
		writer = csv.writer(targetcsv)
		with open('CNN1d_ConfusionMatrix_'+modelname+'.csv', 'r') as sourcecsv:
			reader = csv.reader(sourcecsv)
			for row in reader:
				writer.writerow(row)
		targetcsv.close()				
	# array=normalize(confusion_matrix(labels_testing, model.predict_classes(features_testing)))

	# df_cm = pd.DataFrame(array, range(1,9),
					  # range(1,9))
	# sn.set(font_scale=1.4)  #for label size
	# sn.heatmap(df_cm, annot=True,annot_kws={"size": 10}, cmap='Blues')# font size
	# plt.title('Testing accuracy')
	# plt.xlabel('Predicted label', fontsize=16)
	# plt.ylabel('True label', fontsize=16)
	# plt.show()

# print ("Train Accuracy :: ", accuracy_score(labels_training, model.predict(features_training)))
# print ("Test Accuracy  :: ", accuracy_score(labels_testing, predictions))


