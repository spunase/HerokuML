
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import PIL
import numpy as np
import os
import json

app = Flask(__name__)

model = load_model('model.h5')
# import os
# basedir = os.path.abspath(os.path.dirname(__file__))
# class Config(object):
#     UPLOAD_FOLDER = os.getcwd() + '/images/'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])  
def predict():
    '''
    For rendering results on HTML GUI
    '''
    PATH = os.getcwd()
    img_folder_path = os.path.join(PATH, "images")


    f = request.files['file']
    # filename = secure_filename(f.filename)
    img_url = os.path.join(img_folder_path, f.filename)
    f.save(img_url)
   
    image_predict = image.load_img(img_url, target_size=(64,64))
    image_predict = image.img_to_array(image_predict)
    image_predict = np.expand_dims(image_predict, axis=0)

    y_prob = model.predict(image_predict) 
    y_classes = int(y_prob.argmax(axis=-1))


    # CATEGORY = os.listdir(img_folder_path)
    with open('category.json') as json_file:
          CATEGORY = json.load(json_file)
    output = CATEGORY[y_classes]

    return render_template('index.html', prediction_text='The Image belongs to the classification of {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
