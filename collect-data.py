import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("data"):
    
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    
    os.makedirs("data/train/index")
    os.makedirs("data/train/right-click")
    os.makedirs("data/train/left-click")

    os.makedirs("data/test/index")
    os.makedirs("data/test/right-click")
    os.makedirs("data/test/left-click")
    

# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'index': len(os.listdir(directory+"/index")),
             'right-click': len(os.listdir(directory+"/right-click")),
             'left-click': len(os.listdir(directory+"/left-click"))}
    
    # Printing the count in each set to the screen
    cv2.putText(frame, "Mode: "+mode, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "No. of images", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Index: "+str(count['index']), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Right click: "+str(count['right-click']), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Left click: "+str(count['left-click']), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (128,128)) 
 
    cv2.imshow("Frame", frame)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    roi = cv2.medianBlur(roi, 5)
    
    #_, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'index/'+str(count['index'])+'.jpg', roi)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'right-click/'+str(count['right-click'])+'.jpg', roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'left-click/'+str(count['left-click'])+'.jpg', roi)
    
cap.release()
cv2.destroyAllWindows()
"""
d = "old-data/test/0"
newd = "data/test/0"
for walk in os.walk(d):
    for file in walk[2]:
        roi = cv2.imread(d+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)     
"""# -*- coding: utf-8 -*-

