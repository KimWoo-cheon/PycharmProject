import cv2
import numpy as np
import torch

# 1. YOLOv5 모델을 ONNX로 변환 (한 번만 실행)
def convert_model_to_onnx():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(model, dummy_input, "yolov5s.onnx", verbose=True, opset_version=12)

# 2. ONNX 모델 불러오기
def load_model():
    net = cv2.dnn.readNetFromONNX('yolov5s.onnx')
    return net

# 3. 이미지에서 바운딩 박스 추출
def get_bounding_box(outputs, image_shape, conf_threshold=0.5):
    h, w = image_shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    for detection in outputs[0, 0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            box = detection[0:4] * np.array([w, h, w, h])
            centerX, centerY, width, height = box.astype("int")

            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    return boxes, confidences, class_ids

# 4. 패널을 바운딩 박스 크기로 리사이즈
def resize_panel_to_bbox(image, box):
    x, y, w, h = box
    cropped_panel = image[y:y+h, x:x+w]
    resized_panel = cv2.resize(cropped_panel, (w, h))
    return resized_panel

# 5. 패널을 바운딩 박스에 투영
def project_panel_to_bbox(image, box):
    x, y, w, h = box
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
    dst_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype='float32')
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_panel = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return warped_panel

# 6. 바운딩 박스에 패널 배치
def place_panel_on_image(image, box, panel):
    x, y, w, h = box
    image[y:y+h, x:x+w] = panel
    return image

# 7. 전체 이미지 처리
def process_image_with_panel_detection(image, net, conf_threshold=0.5):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()

    boxes, confidences, class_ids = get_bounding_box(outputs, image.shape, conf_threshold)

    for box in boxes:
        resized_panel = resize_panel_to_bbox(image, box)  # 리사이즈 또는 project_panel_to_bbox(image, box) 사용 가능
        image = place_panel_on_image(image, box, resized_panel)

    return image

# 8. 실행 예제
if __name__ == "__main__":
    # 모델 ONNX로 변환 (이미 변환된 경우 이 부분은 주석 처리)
    # convert_model_to_onnx()

    # 모델 불러오기
    net = load_model()

    # 이미지 불러오기
    image = cv2.imread('input_image.jpg')

    # 이미지 처리
    processed_image = process_image_with_panel_detection(image, net)

    # 결과 저장 및 표시
    cv2.imwrite('output_image_with_panels.jpg', processed_image)
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
