import os
import cv2
import json
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


def load_model():
    # Detectron2 모델 구성 및 로드
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "/content/Fashion-U-Want-Virtual-Try-On/model/model_final_5ad38f.pkl"
    cfg.MODEL.DEVICE = "cuda"  # GPU 사용
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 객체 탐지 임계값 설정
    predictor = DefaultPredictor(cfg)
    return predictor


def inference_and_save(image_path, predictor, output_json_path):
    oriImg = cv2.imread(image_path)
    if oriImg is None:
        print(f"Error: Could not read image from path {image_path}")
        return

    # Detectron2로 예측 수행
    outputs = predictor(oriImg)

    # 신체 관절 예측 결과
    keypoints = outputs["instances"].pred_keypoints.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()

    # JSON 저장 경로와 이름 설정
    json_path = os.path.join(output_json_path, "00001_00_keypoints.json")

    # Prepare JSON output
    json_data = {"version": 1.3, "people": []}
    for idx, kp in enumerate(keypoints):
        person_data = {
            "person_id": [idx],
            "pose_keypoints_2d": [],
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        }
        for keypoint in kp:
            person_data["pose_keypoints_2d"].extend([float(keypoint[0]), float(keypoint[1]), float(keypoint[2])])

        json_data["people"].append(person_data)

    os.makedirs(output_json_path, exist_ok=True)
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON saved at {json_path}")


if __name__ == "__main__":
    # Load model
    predictor = load_model()

    # 입력 및 출력 경로 설정
    input_path = './input/model.jpg'  # 단일 이미지 파일 경로
    output_json_path = './HR-VITON/test/test/openpose_json'  # JSON 저장 경로

    # Ensure output directory exists
    os.makedirs(output_json_path, exist_ok=True)

    # 단일 이미지 처리
    if not input_path.endswith(('.jpg', '.png')):
        raise ValueError(f"Unsupported file format: {input_path}")

    print(f'Processing: {input_path}')
    inference_and_save(input_path, predictor, output_json_path)