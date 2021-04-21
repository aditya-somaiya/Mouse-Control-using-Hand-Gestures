# Mouse-Control-using-Hand-Gestures
## Synopsis

The goal is to innovate a new Human-Computer Interaction System which facilitates the use of natural and more intuitive hand gestures rather than an external mechanical device such as a mouse.

Hand Gestures provide a better way of interaction because they are intuitive i.e. a user can interact with the system more freely with much more flexibility.

## Unique Selling Points: 
1. Pre trained model available, easy to build a custom dataset
2. Use of masking in image processing to eliminate background (within region of interest)
3. Training model code comes with great visualization tools (accuracy & loss curves, confusion matrix, classification report)

## Functions used
## Phase 1: Taking Input 

cv2.VideoCapture()  -  This method is used to start the webcam and access the video.

cap.read() - This is used to read individual frames from the             webcam continuously

cv2.resize() - This method is used to section off a region of interest (ROI) from the captured frames. This is used here to create an area where hand gestures are recognized

## Phase 2: Background Subtraction 

cv2.erode  -  This method is used for background subtraction, it erodes the foreground from the static background.

## Phase 3: Image Processing

cv2.COLOR_BGR2GRAY() -  This is used to convert a normal BGR image to Grayscale

cv2.GaussianBlur() – The Gaussian Blur algorithm suppresses and reduces noise and also smoothens the image by reducing details.

cv2.threshold() - Converts the inputted grayscale image to black and white thresholded image. This is the final processing done to the image before the image is sent to the CNN

## Phase 4: Creating Dataset and Training CNN 

classifier.compile() :
It is used to compile a CNN models and decide key features about it.
It takes input parameters in the form of choice of Optimizers, Loss algorithm and Performance Metrics.
In this case we used the ‘adam’ optimizer and ‘categorical cross entropy’ algorithm to find loss and our performance metric is accuracy

## Phase 5:  Gesture Recognition and Mouse Actions

pyAutoGUI() -  
PyAutoGUI is a cross-platform GUI automation Python module for human beings. Used to programmatically control the mouse & keyboard. It is used to execute mouse functions upon given an input, in this case hand gestures

## Installation
Only requirement, installing these packages :)

from keras.models import model_from_json
import pyautogui as pai
import operator
import cv2
import mouse_position as mp
import numpy as np

## Tests

![image](https://user-images.githubusercontent.com/56780496/115606646-9aa91880-a301-11eb-945b-314ef7b43ecb.png)
![image](https://user-images.githubusercontent.com/56780496/115607334-7b5ebb00-a302-11eb-9dd4-62077aa85b5c.png)
![image](https://user-images.githubusercontent.com/56780496/115607451-a2b58800-a302-11eb-81cb-3293922625b6.png)
![image](https://user-images.githubusercontent.com/56780496/115607515-b4972b00-a302-11eb-9d01-7f69155577e9.png)
![image](https://user-images.githubusercontent.com/56780496/115607586-c7a9fb00-a302-11eb-8d8a-488af592ba08.png)
![image](https://user-images.githubusercontent.com/56780496/115607672-dee8e880-a302-11eb-9c84-f910eb442a2e.png)

