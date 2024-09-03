import cv2
import numpy as np

img = cv2.imread("data/lambo.png")
img2 = cv2.imread("data/fff.png")

# 크기확인 (세로, 가로, 채널수)
print(img.shape) # (462,623,3)
print(img2.shape) # (462,623,3)

# 이미지 리사이즈

imgResize = cv2.resize(img,(300,200))
imgResize2 = cv2.resize(img,(800,600))

#특정위치 이미지 자르기
# crop은 opencv를 사용하지 않고 배열로 시작과 긑을 배열로 끝점 크기를 정해서 자르기만 하면됨
imgCropped = img[250:319,100:400] # img[y:y2,x:x2] # 이미지 자르기


print(imgResize.shape) # (462,623,3)

cv2.imshow("Output",img)
cv2.imshow("Resize", imgResize)
cv2.imshow("Resize2", imgResize2)
cv2.imshow("Output",img2)
cv2.imshow("Cropped",imgCropped)
cv2.waitKey(0)

