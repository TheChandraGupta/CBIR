# -*- coding: utf-8 -*-
"""
Object Recognition Problem Statement Checkpoint - 02
Date of Submission : 10-Feb-2019
"""

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
#from keras.applications.xception import Xception
#from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense,GlobalAveragePooling2D, Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model, load_model
from keras.optimizers import Adam

from sklearn.metrics import classification_report

model = load_model('model')

#print(model.summary())

model.load_weights('weight.h5')

# Load Image Data Set Using Keras
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.15)
    
training_set = train_datagen.flow_from_directory('dataset/training_set_test',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 subset = 'training')
    
validation_set = train_datagen.flow_from_directory('dataset/training_set_test',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 subset = 'validation')

model.evaluate_generator(training_set, 49)
model.evaluate_generator(validation_set, 49)

