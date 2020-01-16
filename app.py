from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import PIL
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import keras.models
from keras.models import model_from_json
#from scipy.misc import imread, imresize,imshow
import tensorflow as tf

# Define a flask app
app = Flask(__name__)

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

#Reading JSON file
with open('./models/model.json','r') as f:
    model_json = f.read()
    f.close()

session = tf.Session(config=config)
keras.backend.set_session(session)

#loading saved model
model = model_from_json(model_json)
model.load_weights('./models/model.h5')
model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
# import pickle

print('Model loaded. Check http://127.0.0.1:5000/')
#print(model.summary())

def load_image(img_file, target_size=(224,224)):
    X = np.zeros((1, *target_size, 3))
    X[0, ] = np.asarray(tf.keras.preprocessing.image.load_img(
        img_file, 
        target_size=target_size)
    )
    X = tf.keras.applications.mobilenet.preprocess_input(X)
    return X

def model_predict(img_path, model):
    try:
        with session.as_default():
            with session.graph.as_default():
                image_batch = load_image(img_path)
                predicted_batch = model.predict(image_batch)
                return predicted_batch 
    except Exception as ex:
        log.log('Prediction Error', ex, ex.__traceback__.tb_lineno)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #file_path = os.path.join(
           # basepath, 'uploads', secure_filename(f.filename))

        file_path = "./uploads/" + secure_filename(f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path,model)

        print(preds)

        the_pred_class = np.argmax(preds)
        print(the_pred_class)

        if the_pred_class == 1:
            return "Maruti Suzuki Swift"
        else:
            return "BMW Z4"
    return None

if __name__ == '__main__':
    app.run(debug=True)
