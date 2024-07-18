import numpy as np
import cv2
import tensorflow as tf
import math

# Load the trained model
model = tf.keras.models.load_model('model/digit_recognition_model.keras')

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    inverse = cv2.bitwise_not(blurred)
    ri = image_refiner(inverse)
    ret, thresh =cv2.threshold(ri,100,255,cv2.THRESH_BINARY)
    return thresh

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

def predict_digit(img):
    img = img/255
    test_image = img.reshape(-1,28,28,1)
    return np.argmax(model.predict(test_image))

width = 1280
height = 720
fps = 30

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

d = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    x1, y1 = int(width/2)-100, int(height/2)-100
    x2, y2 = int(width/2)+200, int(height/2)+200
    f2 = frame.copy()
    
    roi = f2[y1:y2, x1:x2]
    
    if cv2.waitKey(1) & 0xFF == ord('c'):
        ri = preprocess_image(roi)
        digit = predict_digit(ri)
        d = digit
    cv2.putText(frame, f'Predicted: {d}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(roi, f'Predicted: {d}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow('Digit Recognition', frame)
    cv2.imshow('ROI Window', roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
