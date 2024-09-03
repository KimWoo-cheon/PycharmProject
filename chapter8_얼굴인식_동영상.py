import cv2

# Haar Cascade 파일 경로 설정
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 웹캠 객체 생성
cap = cv2.VideoCapture(1)  # 0: 기본 웹캠, 1: 내장 카메라 등

while True:
    ref, frame = cap.read()
    if not ref:
        print('카메라 구동 실패')
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환

    # 얼굴과 눈 검출
    faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4)
    eyes = eyeCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        # 얼굴 영역 ROI 추출
        face_roi = imgGray[y:y+h, x:x+w]
        # 얼굴 외곽선 검출
        contours, _ = cv2.findContours(face_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 얼굴 면적 계산 (외곽선 영역의 실제 면적)
        face_area = sum(cv2.contourArea(contour) for contour in contours)
        # 얼굴 영역 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 얼굴 면적 텍스트로 표시
        cv2.putText(frame, f'Face Area: {face_area:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for (x, y, w, h) in eyes:
        # 눈 영역 ROI 추출
        eye_roi = imgGray[y:y+h, x:x+w]
        # 눈 외곽선 검출
        contours, _ = cv2.findContours(eye_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 눈 면적 계산 (외곽선 영역의 실제 면적)
        eye_area = sum(cv2.contourArea(contour) for contour in contours)
        # 눈 영역 표시
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 눈 면적 텍스트로 표시
        cv2.putText(frame, f'Eye Area: {eye_area:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 결과 영상 출력
    cv2.imshow('Webcam Capture', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 정리
cap.release()
cv2.destroyAllWindows()


