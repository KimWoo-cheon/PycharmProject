import os
import torch
import cv2

# 1. 경로 설정을 명확히 하기
yolov5_repo_path = r"C:\Users\smhrd1\PycharmProjects\pythonProject_001\yolov5"
custom_model_path = r"C:\Users\smhrd1\Desktop\실전프로젝트\model\best.pt"


# 2. YOLOv5 모델을 로드하는 함수
def load_yolov5_model(yolov5_repo_path, custom_model_path):
    # 경로를 명확히 지정해 줍니다
    model = torch.hub.load(yolov5_repo_path, 'custom', path=custom_model_path, source='local', force_reload=True)
    return model


# 3. 모델을 ONNX로 변환하는 함수
def convert_model_to_onnx(model, output_onnx_path='best.onnx'):
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(model, dummy_input, output_onnx_path, verbose=True, opset_version=12)


if __name__ == "__main__":
    # 모델 로드
    model = load_yolov5_model(yolov5_repo_path, custom_model_path)

    # ONNX로 변환
    convert_model_to_onnx(model, output_onnx_path='custom_best.onnx')

