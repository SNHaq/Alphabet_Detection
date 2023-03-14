import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#setting and https context to fetch data from openml
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)): 
  ssl._create_default_https_context = ssl._create_unverified_context

#This line imports the library which stores the handwriting.
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"
           "T", "U", "V", "W", "X", "Y", "Z"]
nClasses = len(classes)
print(nClasses)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
X_train_scale = X_train/255.0
X_test_scale = X_test/255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(X_train_scale, Y_train)

#Here, we are turning the camera on.
cap = cv2.VideoCapture(0)

while True:
  #capture frame by frame
  try: 
    ret, frame = cap.read()
    #our operations on the frame (our photo/video) come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #drawing a focus box in the center of the video
    height, width = gray.shape
    upperLeft = (int(width/2-56), int(height/2-56))
    bottomRight = (int(width/2+56), int(height/2+56))
    cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 2), 2)
    
    #to consider the area inside the box for detecting the digit, we are going 
    #to create ROI = Region of Interest
    roi = gray[upperLeft[1]:bottomRight[1], upperLeft[0]:bottomRight[0]]
    #we are converting the cv2 image to PIL (Python Imaging Library) format
    impil = Image.fromarray(roi)
    imgbw = impil.convert("L")
    imgbwresized = imgbw.resize((28, 28), Image.ANTIALIAS)
    imgbw_resized_inverted = PIL.ImageOps.invert(imgbwresized)

    pixelFilter = 20
    minimumPixel = np.percentile(imgbw_resized_inverted, pixelFilter)
    imgbw_resized_inverted_scaled = np.clip(imgbw_resized_inverted - minimumPixel, 0, 255)

    maximumPixel = np.max(imgbw_resized_inverted)
    imgbw_resized_inverted_scaled = np.asarray(imgbw_resized_inverted_scaled)/maximumPixel

    testSample = np.array(imgbw_resized_inverted_scaled).reshape(1, 784)
    testPredict = clf.predict(testSample)
    print("Our Prediction: ", testPredict)
    #here we are displaying the resultting frame
    cv2.imshow("Frame", gray)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
    
  #the except block deletes all the small errors
  except Exception as e:
    pass

#when everything is done, close the camera
cap.release()
cv2.destroyAllWindows()