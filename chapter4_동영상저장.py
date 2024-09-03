import cv2

# 웹캠 캡쳐 객체 생성

cap= cv2.VideoCapture(1)

# 코덱종류
# MPEG-4.2 : 'PIM1'
# MPEG-4.3 : 'PIM'
# MPEG-4.10/H.264 : 'H264'
# XVID : 'XVID'
# MJPEG : 'MJPG'
# MPEG : 'MPG'

#동영상 저장을 위한 설정
# fourcc = cv2.VideoWriter_fourcc(*'XVID') #코덱을 설정하는 함수
fourcc = cv2.VideoWriter_fourcc(*'mp4v') #코덱을 설정하는 함수
fps = 20.0
width = int(cap.get(3)) # get(3) 넓이
height = int(cap.get(4)) # get(4) 높이

#  출력파일 생성
# out = cv2.VideoWriter('capture/captured_video.avi', fourcc, fps, (width, height),isColor=True)
out = cv2.VideoWriter('capture/captured_video.mp4', fourcc, fps, (width, height),isColor=True)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    # 화면에 프레임 읽기
    cv2.imshow("Web cam Capture",frame)

    # 프레임 저장
    out.write(frame)

    # 'q' 키를 누르면 종료

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 루프를 종료할 때 사용한 리소스를 정리해주세요
cap.release() # while문 안으로 들어가면 안됨..
out.release()
cv2.destroyAllWindows()


