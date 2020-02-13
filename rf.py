from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pickle  #save and load model
from osgeo import gdal, osr
from spectral import envi
import os, sys 
import time
import csv
# """clean data"""
# inputfile='data/Venus2-RS2-MNF'
# clean_data(inputfile)
print ("Loading data...")

"""Loading data"""

def RF(features_training,labels_training,features_testing,labels_testing,modelname):
	print("Training model...")

	"""Random forest"""
	start=time.time()
	model = RandomForestClassifier(n_estimators=100)
	model.fit(features_training,labels_training)
	predictions=model.predict(features_testing)
	end=time.time()
	print(end-start)
	t=end-start

	# # load the model from disk
	# loaded_model = pickle.load(open(filename, 'rb'))
	# result = loaded_model.score(X_test, Y_test)
	# load_model.predict(X_test)
	OA=accuracy_score(labels_testing, predictions)
	Kappa=cohen_kappa_score(labels_testing, predictions)
	array=confusion_matrix(labels_testing, predictions)
	print ("Test Accuracy ", OA)
	print ("Confusion matrix ", array)		
	with open('RF_accuracy_results_'+modelname+'.csv', 'a', newline='') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow([OA,Kappa,t])
		writeFile.close()

	""" Save to file in the current working directory"""
	filename = 'RF_model_'+modelname+'.sav'
	pickle.dump(model, open(filename, 'wb'))

	"""Visualization of results ~ Confusion matrix ~ Labels_testing X Features_testing"""

	df_cm = pd.DataFrame(array, range(1,8), range(1,8))
	df_cm.to_csv('RF_ConfusionMatrix_'+modelname+'.csv')
	
	with open('RF_accuracy_results_'+modelname+'.csv', 'a', newline='') as targetcsv:                
		writer = csv.writer(targetcsv)
		with open('RF_ConfusionMatrix_'+modelname+'.csv', 'r') as sourcecsv:
			reader = csv.reader(sourcecsv)
			for row in reader:
				writer.writerow(row)
		targetcsv.close()	
	
