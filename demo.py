import os
import cv2
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode


def load_model():
    # 모델 구성 및 로드
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "model/model_final_5ad38f.pkl"  # 모델 가중치 경로 설정
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 임계값 설정
    cfg.MODEL.DEVICE = "cuda"  # GPU 사용
    predictor = DefaultPredictor(cfg)
    return predictor


def generate_keypoints_json(image_path, predictor, output_dir):
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Detectron2를 통해 키포인트 예측
    outputs = predictor(img)

    # 키포인트 데이터 추출
    keypoints = outputs["instances"].pred_keypoints.cpu().numpy()

    # JSON 데이터 구조 생성
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
        for x, y, conf in kp:
            person_data["pose_keypoints_2d"].extend([float(x), float(y), float(conf)])
        json_data["people"].append(person_data)

    # JSON 파일 저장
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "00001_00_keypoints.json")
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"Keypoints JSON saved at {json_path}")


if __name__ == "__main__":
    # 모델 로드
    predictor = load_model()

    # 입력 이미지와 출력 경로 설정
    input_image_path = "input/model.jpg"  # 입력 이미지 경로
    output_directory = "HR-VITON/test/test/openpose_json"  # 출력 디렉토리 경로

    # JSON 생성
    generate_keypoints_json(input_image_path, predictor, output_directory)