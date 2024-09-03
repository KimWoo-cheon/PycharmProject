# # 이미지 불러오기
# import cv2
# import numpy as np
# img = cv2.imread('data/lena.png')
# cv2.imshow("ouput",img)
# cv2.waitKey(0)

# 동영상 불러오기
# import cv2
# frameWidth = 640 #동영상 창 크기 설정(가로사이즈)
# frameHeight = 480 # 동영상 창 크기 설정(세로 사이즈)
# cap = cv2.VideoCapture("data/test.mp4")
#
# while True:
#     success, img = cap.read()
#     img = cv2.resize(img, (frameWidth, frameHeight))
#     cv2.imshow("Video", img)
#     if cv2.waitKey(30) & 0xFF == ord("q"): # 영상이 실행중에 'q'를 누르면
#         # waitKey(1) -> 0.001초
#         # 0xFF 아스키 키값
#         break # 종료

# 웹캠 불러오기
# webcam : 0, 내장 카메라 : 1
import cv2
frameWidth = 640  #동영상 창 크기 설정(가로 사이즈)
frameHeight = 480 #동영상 창 크기 설정(세로 사이즈)

cap = cv2.VideoCapture(0) # webcam : 0, 내장 카메라 : 1
if not cap.isOpened():
    print("Error : cannot open Camera")
    exit()
while True:
    success, img = cap.read()
    #동영상이 정상적으로 열리지 않았을 때
    if not success:
        print('영상을 불러오는데 실패 했습니다.')
        break

    img = cv2.resize(img, (frameWidth, frameHeight))
    cv2.imshow('Result', img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

#루프를 종료할 때 사용한 리소스를 정리해주세요
cap.release()   # 루프를 종료할 때 사용한 리소스를 정리해주세요
cv2.destroyAllWindows()