import pandas as pd
import numpy as np
import cv2
import os
import keras
from keras import backend as K
import tensorflow as tf
from datetime import datetime
from keras import regularizers, optimizers
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Dropout, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, f1_score
import dataset as dataset

class MyModel:
    
	def __init__(self, model_pretrained):
		K.clear_session()
		base_model = keras.models.load_model(model_pretrained)
		self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense_1').output)
		self.model._make_predict_function()
		self.graph = tf.get_default_graph()
		self.model.summary()
	
	def extract_features(self, data):
		with self.graph.as_default():
			data_features = self.model.predict(data)
			return data_features
