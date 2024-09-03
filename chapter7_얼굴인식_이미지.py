import cv2

faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

img = cv2.imread('data/faces.png')
imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 흑백 변환

faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4)
eyes = eyeCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2) #bgr
for (x,y,w,h) in eyes:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #bgr
cv2.imshow('result',img)
cv2.waitKey(0)
