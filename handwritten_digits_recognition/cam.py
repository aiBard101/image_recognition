import cv2
import math
import numpy as np

width = 1280
height = 720
fps = 30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

# refining each digit
def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows,cols = gray.shape
    
    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))        
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
    
    #get padding 
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))
    
    #apply apdding 
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    return gray

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    x1, y1 = int(width/2)-100, int(height/2)-100
    x2, y2 = int(width/2)+200, int(height/2)+200
    f2 = frame.copy()
    
    roi = f2[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # ret, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
    
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # for j, cnt in enumerate(contours):
    #     epsilon = 0.01 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
        
    #     hull = cv2.convexHull(cnt)
    #     k = cv2.isContourConvex(cnt)
    #     x, y, w, h = cv2.boundingRect(cnt)
        
    #     if w > 15 and h > 15:  #hierarchy[0][j][3] != -1 and 
    #         cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow('Digit Recognition', frame)
    roi2 = cv2.bitwise_not(blurred)
    ri = image_refiner(roi2)
    ret, thresh =cv2.threshold(ri,100,255,cv2.THRESH_BINARY)
    cv2.imshow('ROI Window', thresh)
    cv2.imshow('ROI Window1', ri)
    cv2.imshow('ROI Window2', roi2)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
