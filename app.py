import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from flask_cors import CORS
from my_model import MyModel
import dataset as dataset

app = Flask(__name__, static_url_path = "/dataset", static_folder = "dataset")
#app.static_url_path = '/dataset'
#CORS(app)

target_size = (64, 64, 3)

myModel = MyModel(model_pretrained='CBIR_Model_01')
data_category = dataset.load_dataset_label()
data_category_ohe = dataset.label_encoding(label_dataset=data_category)
image_array = dataset.load_dataset_image(data_category, target_size)
data_features = dataset.load_features(features_path='X_features_01.npy')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_image']
        
        #img = Image.open(file.stream)
        uploaded_img_path = 'dataset/uploaded/' + file.filename
        print(uploaded_img_path)
        file.save(uploaded_img_path)
        
        query_image = dataset.fetch_image(uploaded_img_path, target_size)
        query_image_features = myModel.extract_features(query_image)
        scores = dataset.top_search(trained_features=data_features, query_features=query_image_features, 
                                  image_data_category=data_category, top_count=5)

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")
