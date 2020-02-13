from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle 
from osgeo import gdal, osr
from spectral import envi
import os, sys 
import time
import csv

def XGBoost (features_training,labels_training,features_testing,labels_testing,modelname):
	"""XGBoost"""
	start=time.time()
	model = XGBClassifier()
	model.fit(features_training,labels_training)
	end=time.time()
	print(end-start)
	t=end-start
	# print(model)

	"""make predictions for test data"""
	predictions = model.predict(features_testing)
	# predictions = [round(value) for value in y_pred]

	OA=accuracy_score(labels_testing, predictions)
	Kappa=cohen_kappa_score(labels_testing, predictions)
	array=confusion_matrix(labels_testing, predictions)
	print ("Test Accuracy ", OA)
	print ("Confusion matrix ", array)
	with open('XGBoost_accuracy_results_'+modelname+'.csv', 'a', newline='') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow([OA,Kappa,t])
		writeFile.close()

	""" Save to file in the current working directory"""
	filename = 'XGBoost_model_'+modelname+'.sav'
	pickle.dump(model, open(filename, 'wb'))

	"""Visualization of results ~ Confusion matrix ~ Labels_testing X Features_testing"""

	df_cm = pd.DataFrame(array, range(1,8), range(1,8))
	df_cm.to_csv('XGBoost_ConfusionMatrix_'+modelname+'.csv')
	
	with open('XGBoost_accuracy_results_'+modelname+'.csv', 'a', newline='') as targetcsv:                
		writer = csv.writer(targetcsv)
		with open('XGBoost_ConfusionMatrix_'+modelname+'.csv', 'r') as sourcecsv:
			reader = csv.reader(sourcecsv)
			for row in reader:
				writer.writerow(row)
		targetcsv.close()


