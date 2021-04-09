import flask
import tensorflow as tf
import time
from flask import request
from flask import Flask,jsonify
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import os
import numpy as np


app=Flask(__name__)


def model(base_model):
    
    last_layer=base_model.output
    x=Flatten()(last_layer)
    x=Dense(256,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(1,activation='sigmoid')(x)

    model=Model(inputs=base_model.input,outputs=x)

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['Accuracy'])

    return model

base_model=tf.keras.applications.VGG16(input_shape=(224,224,3),include_top=False)

for layer in base_model.layers:
    layer.trainable=False

train_model=model(base_model)
train_model.load_weights('mask_detect.h5')



@app.route("/predict",methods=["GET"])
def predict():
    file_path=os.path.join('face dataset','Aaron_Eckhart_0001.jpg')
    image_input=cv2.imread(file_path)
    image_input=cv2.resize(image_input,(224,224))
    prediction=train_model.predict(np.array([image_input]))
    response={}
    response["response"] =  "No mask on the face" if int(prediction[0][0])==1 else "Person is wearing a mask"
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)



