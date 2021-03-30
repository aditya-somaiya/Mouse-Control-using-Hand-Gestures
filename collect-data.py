import cv2
import numpy as np
import math
import os

threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

# Create the directory structure
if not os.path.exists("data"):
    
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    
    os.makedirs("data/train/index")
    os.makedirs("data/train/right-click")
    os.makedirs("data/train/left-click")
    os.makedirs("data/train/scroll-up")
    os.makedirs("data/train/scroll-down")

    os.makedirs("data/test/index")
    os.makedirs("data/test/right-click")
    os.makedirs("data/test/left-click")
    os.makedirs("data/test/scroll-up")
    os.makedirs("data/test/scroll-down")
   

# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'index': len(os.listdir(directory+"/index")),
             'right-click': len(os.listdir(directory+"/right-click")),
             'left-click': len(os.listdir(directory+"/left-click")),
             'scroll-up': len(os.listdir(directory+"/scroll-up")),
             'scroll-down': len(os.listdir(directory+"/scroll-down")),
             'double': len(os.listdir(directory+"/double")),}

    # Printing the count in each set to the screen
    cv2.putText(frame, "Mode: "+mode, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Index: "+str(count['index']), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Right click: "+str(count['right-click']), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Left click: "+str(count['left-click']), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Scroll Up: "+str(count['scroll-up']), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Scroll Down: "+str(count['scroll-down']), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.putText(frame, "Double: "+str(count['double']), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    
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
    #roi = cv2.resize(roi, (64,64)) 
 
    cv2.imshow("Frame", frame)
    
    if isBgCaptured == 1:
    
        img = removeBG(roi)
        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        #kernel = np.ones((1, 1), np.uint8)
        #img = cv2.dilate(mask, kernel, iterations=1)
        #img = cv2.erode(mask, kernel, iterations=1)
        # do the processing after capturing the image!
        #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #_, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        #th = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        cv2.imshow("ROI", thresh)
        
        '''# get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print (cnt)
                    #app('System Events').keystroke(' ')  # simulate pressing blank space
                    

        cv2.imshow('output', drawing)'''
        
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    elif interrupt & 0xFF == ord('i'):
        roi = cv2.resize(thresh, (64, 64))
        cv2.imwrite(directory+'index/'+str(count['index'])+'.jpg', roi)
    elif interrupt & 0xFF == ord('r'):
        roi = cv2.resize(thresh, (64, 64))
        cv2.imwrite(directory+'right-click/'+str(count['right-click'])+'.jpg', roi)
    elif interrupt & 0xFF == ord('u'):
        roi = cv2.resize(thresh, (64, 64))
        cv2.imwrite(directory+'scroll-up/'+str(count['scroll-up'])+'.jpg', roi)
    elif interrupt & 0xFF == ord('d'):
        roi = cv2.resize(thresh, (64, 64))
        cv2.imwrite(directory+'scroll-down/'+str(count['scroll-down'])+'.jpg', roi)
    elif interrupt & 0xFF == ord('2'):
        roi = cv2.resize(thresh, (64, 64))
        cv2.imwrite(directory+'double/'+str(count['double'])+'.jpg', roi)
    elif interrupt & 0xFF == ord('l'):
        roi = cv2.resize(thresh, (64, 64))
        cv2.imwrite(directory+'left-click/'+str(count['left-click'])+'.jpg', roi)
        
    elif interrupt == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif interrupt == ord('z'):  # press 'z' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif interrupt == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
    
cap.release()
cv2.destroyAllWindows()

