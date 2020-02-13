
# summarize filters in each convolutional layer
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from spectral import envi
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
# load the model
# model = VGG16()

model =tf.keras.models.load_model('D:/Chunhua/crop_classifier_2018/2018Script/CNN1d_model_Venus-Pauli2-MNF_crossvalidation5.h5')
#Trained_CNN_1d_Venus2-RS2.h5


""" summarize filter shapes"""

for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
model2 = Model(inputs=model.inputs, outputs=model.layers[3].output)
	
# """ retrieve weights from the second hidden layer"""
# filters, biases = model.layers[1].get_weights()
# print (biases)
# # normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)

# """ plot first few filters"""
# n_filters, ix = 6, 1

# for i in range(n_filters):
	# # get the filter
	# f = filters[:, :, i]
	# # plot each channel separately
	
	# # specify subplot and turn of axis
	# ax = pyplot.subplot(n_filters, 1, ix)
	# ax.set_xticks([])
	# ax.set_yticks([])
	# # plot filter channel in grayscale
	# pyplot.imshow(f[:], cmap='gray')
	# ix += 1
# # show the figure
# pyplot.show()

#load the data
features = envi.open('data/Corn_Venus-Pauli2-MNF.hdr')
features = features.open_memmap(writeable = True)
features= np.array(features)
features =features.reshape((-1,features.shape[2]))  
print(features.shape)
features = features.reshape(features.shape + (1,)) 
print(features.shape)

# scaler = MinMaxScaler(feature_range=(0, 1))
# features = scaler.fit_transform(features)	
# print(features.shape)

# min=np.min(features)
# max=np.max(features)
# print(min,max)
# features=(features-min)/(max-min)
# features=preprocess_input(features)

feature_maps = model2.predict(features)
print (np.shape(feature_maps))

# plot all 64 maps in an 8x8 squares
square = 4
ix = 1
x=list(range(54))
y=features[0,:]
pyplot.plot(x,y)
for _ in range(square):
	for _ in range(square):
		x=list(range(54))
		y=feature_maps[0,:,ix-1]
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.plot(x,y)
		ix += 1
# show the figure
pyplot.show()