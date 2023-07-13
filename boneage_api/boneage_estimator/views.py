from django.shortcuts import render
from .forms import PostForm
from .models import Post
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import redirect, get_object_or_404
from django.core.files import File
from django.conf import settings

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib.request as ur
import json
import cv2
import os
import math
import imutils
import scipy as sp
import scipy.ndimage
from sympy import Point, Ellipse
from keras.optimizers import Adam
import random
from numpy import *
from sklearn import metrics
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K 
from boneage_api.settings import STATIC_DIR
from PIL import Image

baseDir = settings.BASE_DIR

@csrf_exempt
def post_new(request):
	if request.method == "POST":
		form = PostForm(request.POST or None, request.FILES or None)
		if form.is_valid():
			post = form.save(commit = False)
			post.save()
			print("Post saved")
			return redirect('post_detail')
	else:
		form = PostForm()
	return render(request, 'boneage_estimator/post_edit.html', {'form': form}) 

@csrf_exempt
def post_detail(request):
	post = Post.objects.latest('userId')
	imageUrl = post.image.url
	gender = post.gender
	print("Gender = ", gender)
	print("imageUrl", baseDir+imageUrl)

	image = cv2.imread(baseDir+imageUrl, 0)
	preprocessed_img, prediction = PerformBAA.doBAA(image, gender)
	path = r'D:/Web API/boneage_api/media/processed/preprocessed_img.png'
	cv2.imwrite(path, preprocessed_img)
	
	post.prediction = int(prediction)
	post.save()
	
	with open(path, 'rb') as f:
		data = File(f)
		post.carpal_image.save('carpal.png', data, True)

	stringPrediction = str(prediction)
	headPred, sep, tail = stringPrediction.partition('.')

	context = {
		"age": post.age,
		"original_image": post.image,
		"processed_image": post.carpal_image,
		"prediction": headPred
	}
	return render(request, 'boneage_estimator/result.html', context) 

@csrf_exempt
def estimate(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
 
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])
			gender = request.POST.get("gender")
			print("Gender:", gender)
 
		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)
 
			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)
 
			# load the image and convert
			image = _grab_image(url=url)


		img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		preprocessed_img, prediction = PerformBAA.doBAA(img_gray, gender)

		cv2.imwrite("preprocessed_img.png", preprocessed_img)
 
		# update the data dictionary with the faces detected
		data.update({"prediction": prediction, "success": True})
 
	# return a JSON response
	return JsonResponse(data)

 
def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
 
	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			base_url = "http://127.0.0.1:8000"
			resp = ur.urlopen(base_url + url)
			data = resp.read()
 
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
 
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image




class SmallerVGGNet: 
	@staticmethod
	def build(width, height, depth):
		#initialize the model, input shape, and channels dim
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		
		#CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		model.add(MaxPooling2D(pool_size=(3,3)))
		model.add(Dropout(0.5))
		
		#(CONV => RELU) => POOL
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
		
		#(CONV => RELU) => POOL
		model.add(Conv2D(128, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))
		
		#first and only FC =>  RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		
		#output layer
		model.add(Dense(1, activation='sigmoid'))
		# return the constructed network architecture
		return model


# coding: utf-8

# In[ ]:

class Cropping:
	@staticmethod
	def crop_image(image):
		first_index = 4 * int(image.shape[0]/8)
		second_index = 7 * int(image.shape[0]/8)
		cropped_img = image[first_index:second_index]
		return cropped_img
	
	@staticmethod
	def strip_black_sides(image):
		first_index = 2 * int(image.shape[1]/8)
		second_index = 7 * int(image.shape[1]/10)
		cropped_img = image[:, first_index:second_index]
		return cropped_img

class Segmentation:
	@staticmethod
	def verticalProjection(img):
		"Return a list containing the sum of the pixels in each column"
		(h, w) = img.shape[:2]
		sumCols = []
		for j in range(w):
			col = img[0:h, j:j+1] # y1:y2, x1:x2
			sumCols.append(np.sum(col))
		return sumCols
	
	@staticmethod
	def normalize(values, bounds):
		return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]
	
	@staticmethod
	def vertical_bounds(v_sum_norm):
		flag = True
		for i in v_sum_norm:
			if i >= 0.70 and flag == True:
				low_bound = v_sum_norm.index(i)
				flag = False
			elif i >= 0.70:
				upper_bound = v_sum_norm.index(i)

		low_bound = low_bound - 15        
		if low_bound < 0:
			low_bound = 0

		return low_bound , upper_bound+30
	
	@staticmethod
	def horizontalProjection(img):
		"Return a list containing the sum of the pixels in each column"
		(h, w) = img.shape[:2]
		sumCols = []
		for j in range(h):
			col = img[j:j+1, 0:w] # y1:y2, x1:x2
			sumCols.append(np.sum(col))
		return sumCols
	
	@staticmethod
	def horizontal_bounds(h_sum_norm):
		new_list = h_sum_norm[50:]
		val, idx = min((val, idx) for (idx, val) in enumerate(new_list))
		upper_bound = idx + 50

		val2, idx2 = max((val2, idx2) for (idx2, val2) in enumerate(h_sum_norm[0:upper_bound+1]))
		low_bound = idx2 - 40

		if low_bound < 0:
			low_bound = 0

		return low_bound, upper_bound + 30

class ObjectsRefinement:
	def morpho(image):
		kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
		return closing 
	
	@staticmethod
	def get_contours(img):
		#img = morpho(img)
		x,y = img.shape
		ret, contours, hierarchy  = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:12]
		c = np.size(contours[:])
		Blank_window=np.zeros([x,y,3])
		Blank_window=np.uint8(Blank_window)

		peri_list = []
		ecc_list = []
		cnt = []

		for u in contours:
			if (u.size > 6) and len(u) >= 5: 
				cnt.append(u)

		cv2.drawContours(Blank_window, cnt, -1, (255,255,255), 1)
		final_image  = cv2.cvtColor(Blank_window,cv2.COLOR_BGR2GRAY)
		return final_image
	
	@staticmethod
	def get_contours2(img):
		#img = morpho(img)
		x,y = img.shape
		ret, contours, hierarchy  = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)[:12]
		c = np.size(contours[:])
		Blank_window=np.zeros([x,y,3])
		Blank_window=np.uint8(Blank_window)

		peri_list = []
		ecc_list = []
		cnt = []

		for u in contours:
			if (u.size > 6) and len(u) >= 5: 
				ellipse = cv2.fitEllipse(u)
				(center,axes,orientation) = ellipse
				majoraxis_length = max(axes)
				minoraxis_length = min(axes)
				eccentricity = (np.sqrt(1-(minoraxis_length/majoraxis_length)**2))
				ecc_list.append(eccentricity)
				if (eccentricity >= 0.1 and eccentricity <= 0.9):
					area = cv2.contourArea(u)
					hull = cv2.convexHull(u)
					hull_area = cv2.contourArea(hull)
					solidity = float(area)/hull_area

					if solidity >= 0.5:
						cnt.append(u)

		cv2.drawContours(Blank_window, cnt, -1, (255,255,255), 1)
		final_image  = cv2.cvtColor(Blank_window,cv2.COLOR_BGR2GRAY)
		return final_image
	
	@staticmethod
	def flood_fill(test_array,h_max=255):
		input_array = np.copy(test_array) 
		el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
		inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
		output_array = np.copy(input_array)
		output_array[inside_mask]=h_max
		output_old_array = np.copy(input_array)
		output_old_array.fill(0)   
		el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
		while not np.array_equal(output_old_array, output_array):
			output_old_array = np.copy(output_array)
			output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
		#output_array = np.array(output_array, dtype=np.uint8)
		return output_array

	@staticmethod
	def get_square(image,square_size):
		height,width = image.shape
		if(height>width):
			differ=height
		else:
			differ=width
		differ+=4
		mask = np.zeros((differ,differ), dtype="uint8")   
		x_pos=int((differ-width)/2)
		y_pos=int((differ-height)/2)
		mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
		mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)
		return mask 

def denormalize(normalized):
	min =  0
	max = 66
	denormalized = (normalized * (max - min) + min);
	denormalized = round(denormalized)
	return denormalized

def model_predict(x_test, gender):
	seed = 7
	np.random.seed(seed)

	EPOCHS = 200
	INIT_LR = 1e-3
	BS = 32
	IMAGE_DIMS = (96, 96, 3)

	x_test = x_test.astype('float32') /255.0
  
	model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height = IMAGE_DIMS[0], depth=IMAGE_DIMS[2])
	opt = Adam(lr=INIT_LR, decay = INIT_LR/EPOCHS)
	
	# Compile model
	if gender == 'Female':
		weights_path = os.path.join(STATIC_DIR, "girls.weights.best.hdf5")
	elif gender == 'Male':
		weights_path = os.path.join(STATIC_DIR, "boys.weights.best.hdf5")

	model.load_weights(weights_path)
	model.compile(loss="mse", optimizer=opt, metrics=["mae"])

	#evaluation
	x_test = cv2.cvtColor(x_test,cv2.COLOR_GRAY2RGB)
	x_test= np.expand_dims(x_test, axis=0)
	y_pred = model.predict(x_test, verbose=0)
	print("normalized: ", y_pred)
	
	y_pred_inv = []
	for sublist in y_pred:
		for item in sublist:
			y_pred_inv.append(item)
	
	denormalized = denormalize(y_pred_inv[0])
	print("Prediction: ", denormalized)
	K.clear_session()
	return(denormalized)

class PerformBAA: 
	@staticmethod
	def doBAA(img_gray, gender):
		width=512
		height = 600
		#img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
		resized = imutils.resize(img_gray, width=width, height=height)

		crop_img = Cropping.crop_image(resized)
		img = Cropping.strip_black_sides(crop_img)

		v_sum = Segmentation.verticalProjection(img)
		v_sum_norm = Segmentation.normalize( v_sum, {'actual': {'lower': 0.0, 'upper': max(v_sum)}, 'desired': {'lower': 0.0, 'upper': 1.0}})    
		(v_lower, v_upper) = Segmentation.vertical_bounds(v_sum_norm)    
		h_sum = Segmentation.horizontalProjection(img)
		h_sum_norm = Segmentation.normalize(h_sum, {'actual': {'lower': 0.0, 'upper': max(h_sum)}, 'desired': {'lower': 0.0, 'upper': 1.0}}) 
		(h_lower, h_upper) = Segmentation.horizontal_bounds(h_sum_norm)
		roi = img[h_lower:h_upper+1, v_lower:v_upper+1]

		blurred = cv2.medianBlur(roi,3)
		ret, th1 = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
		lowThresh = 0.5 * ret
		canny = cv2.Canny(blurred, lowThresh, ret)

		new_img = ObjectsRefinement.get_contours(canny)
		refined = ObjectsRefinement.get_contours2(new_img)
		filled2 = ObjectsRefinement.flood_fill(refined)
		masked_data = cv2.bitwise_and(roi, roi, mask = filled2) 
		masked_data =  ObjectsRefinement.get_square(masked_data, 96)

		prediction = model_predict(masked_data, gender)

		return masked_data, prediction


