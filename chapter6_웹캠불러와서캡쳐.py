import cv2
import os
# 웹캠 객체 생성

cap = cv2.VideoCapture(1) # 0: 웹캠 1: 내장 카메라
cnt = 0

folder_path = 'capture'

while True :
    ref,frame = cap.read()
    if not ref :
        print('카메라 구동 실패')
        break
    # f'cap_img{cnt}.jpeg'
    # 's'누르면 저장
    cv2.imshow('Webcam Capture',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        file_name = f'cap_img{cnt}.png'
        # 이미지 저장 전체경로
        out_path = os.path.join(folder_path, file_name)  # 폴더에 저장할 때
        cv2.imwrite(out_path, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

        print('프레임 저장 완료')
        cnt += 1 # cnt = cnt + 1

# 루프 종료시 사용한 리소스 정리
cap.release()
cv2.destroyAllWindows()


