import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import cv2
import StagImages as si
from tensorflow.keras.models import load_model

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

def calcScore(i):
    score_i = (float) (100 * scores[i])
    return "  {0}%".format(round(score_i))


### SETTINGS
# Fill user names
class_names = ['User1', 'User2', 'User3', 'User4', 'User5']

#Model input image size
img_height = 128
img_width = 128

isGrayImage = False
saved_model = "trained_models/model_color_"

### /SETTINGS


saved_model += str(img_height) + "_" + str(img_width)
print(saved_model)

#Loading Model
model = load_model(saved_model)

faceCascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

#Camera Loop
while True:

    ret, img = cam.read()

    if isGrayImage:
        #If you want gray photo capturing, Array is resizing (640,480,1) ==> (640,480,3) for model input
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        #
    else:
        #Colored photo to model predict
        gray = img

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        grayForPredict = cv2.resize(gray, (img_height, img_width))
        img_array = tf.expand_dims(grayForPredict, 0)  # Create a batch
        #tenserflow predict the image
        predictions = model.predict(img_array)
        #

        scores = tf.nn.softmax(predictions[0])
        scoreMax = 100 * np.max(scores)


        strList = ""
        for i in range(0,len(class_names)):
            strList+=  (str(class_names[i]) + "-" + calcScore(i)+ ", ")
        # you can see all of the users score in the console
        print(strList)

        if (scoreMax > 50):
            foundName = class_names[np.argmax(scores)]
            calculatedScore = "  {0}%".format(round(scoreMax))
        else:
            printid = "unknown"
            printScore = "  {0}%".format(round(scoreMax))

        cv2.putText(img, str(foundName), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(calculatedScore), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    cv2.waitKey(100)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()



