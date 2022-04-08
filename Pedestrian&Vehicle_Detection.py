# -*- coding: utf-8 -*-
"""
Created on Thur Apr  8 15:09:22 2022
@author: Mohammed Abdul Hafeez Khan
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


!wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/videos.zip
!wget https://moderncomputervision.s3.eu-west-2.amazonaws.com/haarcascades.zip

!unzip -qq haarcascades.zip
!unzip -qq videos.zip

"""#### **Testing on a Single Frame from our Video**"""

capture = cv2.VideoCapture('walking.mp4')


body_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')


ret_bool, frame = capture.read()

if ret_bool: 


  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  bodies_detected = body_classifier.detectMultiScale(gray, 1.2, 3)

 
  for (x,y,w,h) in bodies_detected:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
  

capture.release()   
imshow("Pedestrian Detector", frame)

capture = cv2.VideoCapture('walking.mp4')


w = int(capture.get(3))
h = int(capture.get(4))

output = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

body_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')

while(True):

  ret_bool, frame = capture.read()
  if ret_bool: 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
    bodies_detected = body_detector.detectMultiScale(gray, 1.2, 3)

   
    for (x,y,w,h) in bodies_detected:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
   
    output.write(frame)
  else:
      break

capture.release()
output.release()

!ffmpeg -i /content/walking_output.avi walking_output.mp4 -y

from IPython.display import HTML
from base64 import b64encode

mp4 = open('walking_output.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

capture = cv2.VideoCapture('cars.mp4')

vehicle_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')

ret_bool, frame = capture.read()

if ret_bool:
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  vehicles = vehicle_detector.detectMultiScale(gray, 1.3, 2)

  for (x,y,w,h) in vehicles:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
capture.release()
imshow('Vehicle Detector', frame)

capture = cv2.VideoCapture('cars.mp4')

vehicle_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')

width = int(capture.get(3))
height = int(capture.get(4))

out = cv2.VideoWriter('cars_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

while(True):
  ret_bool, frame = capture.read()

  if ret_bool:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    vehicles = vehicle_detector.detectMultiScale(gray, 1.2, 3)

    for (x,y,width,height)in vehicles:
      cv2.rectangle(frame, (x,y), (x+width, y+height), (0,255,255), 2)

    out.write(frame)
  else:
    break 

capture.release()
out.release()

!ffmpeg -i /content/cars_output.avi cars_output.mp4 -y

from IPython.display import HTML
from base64 import b64encode

mp4 = open('cars_output.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

HTML("""
<video controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

