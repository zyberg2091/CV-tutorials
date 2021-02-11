import cv2

import numpy as np
from PIL import Image

facemodel=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



   
def detection(image,gray):
    # for face detection
    faces=facemodel.detectMultiScale(gray, 1.1, 5)
    if(len(faces)>0):
        print("Found {} faces".format(str(len(faces))))
    for (x, y, w, h )in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
        # # for eye detection not implemented
        # roi_gimage=gray[x:x+w,y:y+h]
        # roi_image=image[x:x+w,y:y+h]
    return image


image=cv2.imread('face dataset/Aaron_Eckhart_0001.jpg')
gimage= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

canvas=detection(image,gimage)

while True:
    cv2.imshow('images',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

