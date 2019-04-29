import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array


def base_directory():
	return ''

def load_dataset_label():
	annotation = pd.read_csv(base_directory() + 'dataset/annotation.txt', delimiter='\t')
	annotation = annotation[['filename', 'color', 'micro_category', 'macro_category', 'macro_category(english)']]
	data_category = annotation[['filename', 'macro_category(english)']]
	count = 0
	for name, groups in data_category.groupby('macro_category(english)'):
		print(name)
		count += 1
	print(count)
	return data_category

def label_encoding(label_dataset):
	data_category = label_dataset.iloc[:, 1].values
	data_category = data_category.reshape(-1, 1)
	ohe = OneHotEncoder(sparse=False)
	data_category_ohe = ohe.fit_transform(data_category)
	return data_category_ohe

def load_dataset_image(data_category, target_size = (128, 128, 3)):
	image_array = np.ndarray(shape=(len(data_category), 64, 64, 3), dtype=float)
	i=0
	count = 0
	for filename in data_category.values:
		try:
			image_load_path = base_directory() + 'dataset/img_128_128/' + filename[0]
			img = img_to_array(load_img(image_load_path, target_size=target_size))
			img = img / 255
			image_array[i] = img
			i += 1
			if i % 10000 == 0:
				print(i)
		except:
			print('File Not Found ' + filename[0])
			count += 1
		
	print('Count:' + str(count))
	return image_array
	
def split_dataset(image_array, data_category_ohe):
	X_train, X_test, y_train, y_test = train_test_split(image_array, data_category_ohe, test_size=0.15)
	return (X_train, X_test, y_train, y_test)
	
def fetch_image(image_paths, target_size = (128, 128, 3)):
	image_array = np.ndarray(shape=(1, 64, 64, 3), dtype=float)
	img = img_to_array(load_img(image_paths, target_size=target_size))
	img = img / 255
	image_array[0] = img
	return image_array
	
def load_features(features_path):
		return np.load(features_path)
		
def top_search(trained_features, query_features, image_data_category, top_count = 10):
		dists = np.linalg.norm(trained_features - query_features, axis=1)
		ids = np.argsort(dists)[:top_count]
		scores = [(dists[id], base_directory() + 'dataset/img_128_128/' + image_data_category.values[id][0]) for id in ids]
		return scores
