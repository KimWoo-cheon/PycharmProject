# import cv2
# import mysql.connector
#
# # MySQL 데이터베이스 연결
# conn = mysql.connector.connect(
#     host='localhost',  # MySQL 서버 호스트 이름
#     user='your_username',  # MySQL 사용자 이름
#     password='your_password',  # MySQL 비밀번호
#     database='your_database'  # 사용할 데이터베이스 이름
# )
# cursor = conn.cursor()
#
# # 테이블 생성 (이미 생성되었으면 생략 가능)
# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS Detections (
#         id INT AUTO_INCREMENT PRIMARY KEY,
#         object_number INT,
#         x1 FLOAT,
#         y1 FLOAT,
#         x2 FLOAT,
#         y2 FLOAT,
#         face_area FLOAT,
#         eye_area_ratio FLOAT
#     )
# ''')
# conn.commit()
#
# # Haar Cascade 파일 경로 설정
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#
# # 웹캠 객체 생성
# cap = cv2.VideoCapture(0)  # 0: 기본 웹캠, 1: 내장 카메라 등
# object_number = 0
#
# while True:
#     ref, frame = cap.read()
#     if not ref:
#         print('카메라 구동 실패')
#         break
#
#     frame = cv2.flip(frame, 1)  # 좌우 반전
#     imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환
#
#     # 얼굴 검출
#     faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4)
#
#     for (x, y, w, h) in faces:
#         object_number += 1
#         # 얼굴 영역 ROI 추출
#         face_roi = imgGray[y:y + h, x:x + w]
#         # 얼굴 외곽선 검출
#         contours, _ = cv2.findContours(face_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         # 얼굴 면적 계산 (외곽선 영역의 실제 면적)
#         face_area = sum(cv2.contourArea(contour) for contour in contours)
#
#         # 눈 검출
#         eyes = eyeCascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=4)
#         eye_area = 0
#         for (ex, ey, ew, eh) in eyes:
#             eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
#             eye_contours, _ = cv2.findContours(eye_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             eye_area += sum(cv2.contourArea(contour) for contour in eye_contours)
#
#         # 눈 면적/얼굴 면적 비율 계산
#         eye_area_ratio = eye_area / face_area if face_area > 0 else 0
#
#         # 얼굴 영역 표시
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         # 얼굴과 눈 면적 비율 텍스트로 표시
#         cv2.putText(frame, f'Face {object_number}: Area {face_area:.2f}, Eye Ratio {eye_area_ratio:.2f}',
#                     (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#         # MySQL에 얼굴 정보 저장
#         cursor.execute('''
#             INSERT INTO Detections (object_number, x1, y1, x2, y2, face_area, eye_area_ratio)
#             VALUES (%s, %s, %s, %s, %s, %s, %s)
#         ''', (object_number, x, y, x + w, y + h, face_area, eye_area_ratio))
#         conn.commit()
#
#     # 결과 영상 출력
#     cv2.imshow('Webcam Capture', frame)
#
#     # 'q' 키를 누르면 루프 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 리소스 정리
# cap.release()
# cv2.destroyAllWindows()
#
# # MySQL 연결 종료
# cursor.close()
# conn.close()


import cv2

# Haar Cascade 파일 경로 설정
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 웹캠 객체 생성
cap = cv2.VideoCapture(0)  # 0: 기본 웹캠, 1: 내장 카메라 등
object_number = 0

while True:
    ref, frame = cap.read()
    if not ref:
        print('카메라 구동 실패')
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환

    # 얼굴 검출
    faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        object_number += 1
        # 얼굴 영역 ROI 추출
        face_roi = imgGray[y:y + h, x:x + w]

        # Canny 에지 검출 추가
        edges = cv2.Canny(face_roi, 100, 200)

        # 얼굴 외곽선 검출
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 얼굴 면적 계산 (외곽선 영역의 실제 면적)
        face_area = sum(cv2.contourArea(contour) for contour in contours)

        # 눈 검출
        eyes = eyeCascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=4)
        eye_area = 0
        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
            eye_edges = cv2.Canny(eye_roi, 100, 200)
            eye_contours, _ = cv2.findContours(eye_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            eye_area += sum(cv2.contourArea(contour) for contour in eye_contours)

            # 눈 윤곽선을 얼굴 ROI 내부에 그리기 (전체 이미지의 좌표로 변환)
            cv2.drawContours(frame, eye_contours, -1, (0, 0, 255), 3, offset=(x + ex, y + ey))

        # 눈 면적/얼굴 면적 비율 계산
        eye_area_ratio = eye_area / face_area if face_area > 0 else 0

        # 얼굴 윤곽선 표시
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1, offset=(x, y))
        # 얼굴과 눈 면적 비율 텍스트로 표시
        cv2.putText(frame, f'Face {object_number}: Area {face_area:.2f}, Eye Ratio {eye_area_ratio:.2f}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 결과 영상 출력
    cv2.imshow('Webcam Capture', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 정리
cap.release()
cv2.destroyAllWindows()


