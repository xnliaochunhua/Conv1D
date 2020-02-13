import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tensorflow.keras.layers import Dense, Dropout
import csv
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from clean_data import clean_data
import time


def MLP(features_training,labels_training,features_testing,labels_testing,features_validation,labels_validation,modelname):

	"""Empty the results file whenever we run TF"""
	# with open('TF_accuracy_results.csv', 'w', newline='') as writeFile:
		# writer = csv.writer(writeFile)
		# writer.writerow(['Epoch', 'Testing Accuracy'])

	"""Check testing accuracy after each epoch, does this by looking at accuracy of first 20000 shuffled testing features and testing labels"""
	class MyCallBack(Callback):
		def on_epoch_end(self, epoch, logs=None):
			acc=accuracy_score(labels_testing[:20000], model.predict_classes(features_testing[:20000]))
			print("Testing accuracy:", acc)

			# with open('TF_accuracy_results.csv', 'a', newline='') as writeFile:
				# writer = csv.writer(writeFile)
				# try:
					# writer.writerow([self.model.history.epoch[-1]+2,acc])
				# except:
					# writer.writerow([1, acc])
			# writeFile.close()
	cbk=MyCallBack()

	print ("Training data...")

	"""Tensorflow model code"""
	model = tf.keras.models.Sequential()  # a basic feed-forward model
	model.add(tf.keras.layers.Dense(512, activation='relu'))  # a simple fully-connected layer, 512 units, relu activation
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(512, activation='relu'))  # a simple fully-connected layer, 512 units, relu activation
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(512, activation='relu'))  # a simple fully-connected layer, 512 units, relu activation
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(512, activation='relu'))  # a simple fully-connected layer, 512 units, relu activation
	model.add(tf.keras.layers.Dropout(0.5))

	model.add(tf.keras.layers.Dense(8, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

	start=time.time()
	model.compile(optimizer=Adam(lr=1.5e-4,),  # Good default optimizer to start with
				  loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
				  metrics=['accuracy'])  # what to track
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
	
	history=model.fit(features_training, labels_training,
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
	
	model.save('MLP_model_'+modelname+'.h5')
	OA=accuracy_score(labels_testing, model.predict_classes(features_testing))
	Kappa=cohen_kappa_score(labels_testing, model.predict_classes(features_testing))
	array=confusion_matrix(labels_testing, model.predict_classes(features_testing))
	print ("Test Accuracy ", OA)
	print ("Confusion matrix ", array)	

	end=time.time()
	print (end-start)
	t=end-start
	with open('MLP_accuracy_results_'+modelname+'.csv', 'a', newline='') as writeFile:
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
	df_cm.to_csv('MLP_ConfusionMatrix_'+modelname+'.csv')
	
	with open('MLP_accuracy_results_'+modelname+'.csv', 'a', newline='') as targetcsv:                
		writer = csv.writer(targetcsv)
		with open('MLP_ConfusionMatrix_'+modelname+'.csv', 'r') as sourcecsv:
			reader = csv.reader(sourcecsv)
			for row in reader:
				writer.writerow(row)  
		targetcsv.close()				
	# sn.set(font_scale=1.4)  #for label size
	# sn.heatmap(df_cm, annot=True,annot_kws={"size": 10}, cmap='Blues')# font size
	# plt.title('Testing accuracy')
	# plt.xlabel('Predicted label', fontsize=16)
	# plt.ylabel('True label', fontsize=16)
	# plt.show()


