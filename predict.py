from keras.models import model_from_json
import pyautogui as pai
import operator
import cv2
import mouse_position as mp
import numpy as np
import handDetection as hde

threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
#to save the prediction result for a few frames
fList=[]

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Constants for finding range of skin color in YCrCb
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

nof = 1
while cap.isOpened():
    nof = nof+1
    _, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
      

    # Same actions as collect-data.py for the CNN predictions and display
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
    
    #tmep = hde.detect_hand(roi, min_YCrCb, max_YCrCb)
    # Resizing the ROI so it can be fed to the model for prediction
    '''roi = cv2.resize(roi, (64, 64))     
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.medianBlur(roi, 5)
    
    test_image = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow("test", test_image)'''
    cv2.imshow("Frame", frame)
    if isBgCaptured == 1:
    
        img = removeBG(roi)
        cv2.imshow('Remove BG', img)
        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('Grayscale', blur)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ROI', thresh)
        #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # Batch of 1
        test_image = cv2.resize(thresh, (64, 64))
        result = loaded_model.predict(test_image.reshape(1 ,64, 64, 1))
        
        #tmep = hde.detect_hand(roi, min_YCrCb, max_YCrCb)
        x, y, w, h = cv2.boundingRect(thresh)
        top = (np.argmax(thresh[y, :]), y)
        cv2.circle(roi, top, 8, (255, 50, 0), -1)

        prediction = {'double': result[0][0],
                      'index': result[0][1], 
                  'left-click': result[0][2], 
                  'right-click': result[0][3],
                  'scroll-down': result[0][4],
                  'scroll-up': result[0][5]}
    
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        fList.append(str(prediction[0][0]))
        
        if prediction[0][0] == "inde":
            mp.moveCursor(roi, top)
    
        if len(fList) >= 1:
            repeated = max(set(fList), key = fList.count)
            fList.clear()
            print(repeated)
            # Displaying the predictions
            cv2.putText(frame, repeated, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)    
            cv2.imshow("Frame", frame)

            if top is None:
                pass
            else:
                '''if repeated == "index":
                    mp.moveCursor(roi, top)'''
                if repeated == "double":
                    pai.doubleClick()
                elif repeated == "left-clic":
                    pai.click(button='left')
                elif repeated == "scroll-u":
                    pai.scroll(100)
                elif repeated == "scroll-dow":
                    pai.scroll(-100)
                elif repeated == "right-clic":
                    pai.click(button='right')
                else:
                    pass       
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
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
print("Total number of frames = "+str(nof))
cap.release()
cv2.destroyAllWindows()
