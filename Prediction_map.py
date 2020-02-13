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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from osgeo import gdal, osr
from spectral import envi
import os, sys 
import time
import pickle  #save and load model

# def Prediction_map(features_training,labels_training,features_testing,labels_testing,modelname):

print("Making predictions...")

featureds=gdal.Open('D:/Chunhua/crop_classifier_2018/2018Script/data/Venus-Pauli2-MNF')
rows = featureds.RasterYSize
cols = featureds.RasterXSize
features = envi.open('data/Venus-Pauli2-MNF.hdr')
features = features.open_memmap(writeable = True)
features= np.array(features)
features =features.reshape((-1,features.shape[2]))  
print(features.shape)
# features = features.reshape(features.shape + (1,)) 
# print(features.shape)

# # load the model from disk MLP and CNN
# loaded_model =tf.keras.models.load_model('D:/Chunhua/crop_classifier_2018/2018Script/XGBoost_model_Venus-Pauli2-MNF_crossvalidation5.h5')
# # result = loaded_model.score(X_test, Y_test)
# a=loaded_model.predict_classes(features)

# load the model from disk RF and XGBoost
loaded_model = pickle.load(open('RF_model_Venus-Pauli2-MNF_crossvalidation5.sav', 'rb'))
# result = loaded_model.score(X_test, Y_test)
a=loaded_model.predict(features)


# a= model.predict(features) # Input: 1D Array
a=a.reshape((1495,1729))
print("Prediction obtained")

# save files
driver = gdal.GetDriverByName('GTiff')
newRasterfn= os.path.join('D:/Chunhua/crop_classifier_2018/2018Script/data','RF_Venus-Pauli2-MNF-v5.tif')
outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
outRaster.SetGeoTransform(featureds.GetGeoTransform())
outband = outRaster.GetRasterBand(1)
outband.WriteArray(a)
outRasterSRS = osr.SpatialReference()
outRasterSRS.ImportFromWkt(featureds.GetProjection())
outRaster.SetProjection(outRasterSRS.ExportToWkt())
outband.FlushCache()
