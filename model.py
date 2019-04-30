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
		self.model = self.load_model(self,data, model_pretrained=model_pretrained, 
								target_size=target_size, n_classes=n_classes, batch=batch, epochs=epochs, 
								baseMapNum=baseMapNum, weight_decay=weight_decay, decay=decay, lr=lr, loss = loss)
		self.model = keras.models.load_model(model_pretrained)
		self.model._make_predict_function()
		self.graph = tf.get_default_graph()
		self.model.summary()
		
	def get_model(self):
		return self.model
	
	def load_model(self, data, load_pretrained=True, model_pretrained=None, target_size=(128, 128, 3), n_classes=100, 
						batch=64, epochs=50, baseMapNum=32, weight_decay=1e-4, decay=1e-6, lr=0.001, 
						loss = 'categorical_crossentropy'):
		model = None
		if load_pretrained:
			model = keras.models.load_model(model_pretrained)
			model.summary()
		else:
			model = self.train_model(self, data, target_size=target_size, n_classes=n_classes, 
						batch=batch, epochs=epochs, baseMapNum=baseMapNum, weight_decay=weight_decay, 
                        decay=decay, lr=lr, loss = loss)
		return model

	def create_model(self, target_size=(128, 128, 3), n_classes=100,baseMapNum=32, weight_decay=1e-4):
		
		model = Sequential()
		model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=target_size))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.2))
		
		model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.3))

		model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		#model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.4))

		model.add(Conv2D(8*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(8*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		#model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))

		model.add(Conv2D(16*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(16*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.6))

		model.add(Flatten())
		#model.add(GlobalAveragePooling2D())
		#model.add(Dropout(0.6))
		#model.add(Dense(100, activation='relu'))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(n_classes, activation='softmax'))
		
		model.summary()
		
		return model

	def train_model(self, data, target_size=(128, 128, 3), n_classes=100, batch=64, epochs=50, baseMapNum=32, 
						weight_decay=1e-4, decay=1e-6, lr=0.001, loss = 'categorical_crossentropy'):
		
		model = self.create_model(self, target_size, n_classes, baseMapNum, weight_decay)
		
		X_train, X_test, y_train, y_test = data
		
		datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=False)
		datagen.fit(X_train)
		
		opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
		model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
		model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
								steps_per_epoch=X_train.shape[0] // batch_size, 
								epochs=1*epochs,verbose=1, validation_data=(X_test,y_test))
							
		save_trained_model = 'CBIR' + '_ep' + epochs + '_bt' + batch + '_lr' + lr + '_' + datetime.now().utcnow()
		model.save(save_trained_model)
		
		y_pred = (model.predict(X_test) > 0.5)
		
		print(classification_report(y_test, y_pred))
		print(f1_score(y_test, y_pred, average='weighted'))
		return model


	def create_feature_model(self, input_model, output_layer):
		self.model = Model(inputs=input_model.input, outputs=input_model.get_layer(output_layer).output)
		self.model.summary()
		self.model._make_predict_function()
		self.graph = tf.get_default_graph()
		return self.model

	def predict_model(self, data):
		with self.graph.as_default():
			data_features = self.model.predict(data)
			return data_features

	def extract_features(self, data, save_features=True):
		data_features = self.predict_model(self, data)
		if save_features:
			np.save('features_' + datetime.now().utcnow() + '.npy', data_features)
		return data_features

	def load_features(self, features_path):
		return np.load(features_path)

	def top_search(self, trained_features, query_features, image_data_category, top_count = 10):
		dists = np.linalg.norm(trained_features - query_features, axis=1)
		ids = np.argsort(dists)[:top_count]
		scores = [(dists[id], dataset.base_directory() + 'dataset/img_128_128/' + image_data_category[id][0]) for id in ids]
		return scores

