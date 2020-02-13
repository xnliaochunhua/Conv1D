from spectral import envi
import numpy as np
from random import shuffle
import os, sys 
from osgeo import gdal, osr

images=['data/fold_1.hdr','data/fold_2.hdr','data/fold_3.hdr','data/fold_4.hdr','data/fold_5.hdr']
for image in images:
	training = envi.open(image)
	# training = training.open_memmap(writeable = True)
	training = training[:,:,0]
	training = training.reshape(-1)
	print (training.shape)
	n1=0
	n2=0
	n3=0
	n4=0
	n5=0
	n6=0
	n7=0
	n8=0
	for i, label in enumerate(training):

		if label==1.0:
			n1=n1+1

			if n1>20000:
				training[i]=0.0

		elif label==2.0:
			n2=n2+1
			if n2>22000:
				training[i]=0.0			
		elif label==3.0:
			n3=n3+1	
			if n3>3800:
				training[i]=0.0
		elif label==4.0:
			n4=n4+1	
			if n4>3000:
				training[i]=0.0			
		elif label==5.0:
			n5=n5+1	
			if n5>2500:
				training[i]=0.0			
		elif label==6.0:
			n6=n6+1		
			if n6>480:
				training[i]=0.0
		elif label==7.0:
			n7=n7+1		
			if n7>370:
				training[i]=0.0


	
	lable_training=training.reshape(1495,1729)

	print(lable_training.shape)
	featureds=gdal.Open('data/fold_1')
	rows = featureds.RasterYSize
	cols = featureds.RasterXSize
	driver = gdal.GetDriverByName('GTiff')
	newRasterfn= image+'.tif'
	outRaster = driver.Create(newRasterfn,  cols, rows, 1, gdal.GDT_Float32)
	outRaster.SetGeoTransform(featureds.GetGeoTransform())
	outband = outRaster.GetRasterBand(1)
	outband.WriteArray(lable_training)
	outRasterSRS = osr.SpatialReference()
	outRasterSRS.ImportFromWkt(featureds.GetProjection())
	outRaster.SetProjection(outRasterSRS.ExportToWkt())
	outband.FlushCache()


	
	





