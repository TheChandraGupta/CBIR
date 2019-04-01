# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:58:13 2019

@author: GUPTA
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

import cv2
"""
import glob

folders = glob.glob('dataset/*')
imagenames_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.jpg'):
        imagenames_list.append(f)
"""


annotation = pd.read_csv('dataset/annotation.txt', delimiter='\t') 
annotation = annotation[['filename', 'color', 'micro_category', 'macro_category', 'macro_category(english)']]
data_category_01 = annotation[['filename', 'macro_category(english)']]
data_category_02 = annotation[['filename', 'macro_category']]
data_category_03 = annotation[['filename', 'micro_category']]

count = 0
for name, groups in data_category_03.groupby('micro_category'):
    print(name)
    count += 1

print(count)

count = 0
for name, groups in data_category_02.groupby('macro_category'):
    print(name)
    count += 1

print(count)

count = 0
for name, groups in data_category_01.groupby('macro_category(english)'):
    print(name)
    count += 1

print(count)

read_images = []

for filename in data_category_01.values:
    image_path = 'dataset/images/' + filename[0].split('.')[0] + '_resized.' + filename[0].split('.')[1]
    print(image_path)
    read_images.append(cv2.imread(image_path))
    
    
image_array = np.ndarray(shape=(len(data_category_01), 128, 128, 3), dtype=np.float32)
i=0
count = 0
for filename in data_category_01.values:
    try:
        image_path = 'dataset/images/' + filename[0].split('.')[0] + '_resized.' + filename[0].split('.')[1]
        img = img_to_array(load_img(image_path, target_size=(128,128,3)))
        img = img / 255
        image_array[i] = img
        #np.append(image_array, img)
        i += 1
        print(image_array.shape)
    except:
        print('File Not Found ' + filename)
        count += 1
    
np.save('image_dataset_array.npy', image_array)
image_array_loaded = np.load('image_dataset_array.npy')







    
    
    