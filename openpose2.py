import cv2
import os
import json
import numpy as np
from src import util
from src.body import Body
from src.hand import Hand


def load_model(use_hand=False):
    model_path = './model/pose_iter_584000.caffemodel.pt'
    body_estimation = Body(model_path, 'body25')
    hand_estimation = Hand('./model/hand_pose_model.pth') if use_hand else None
    return body_estimation, hand_estimation


def adjust_precision(keypoints, precision=3):
    """좌표 및 confidence 값 정밀도 조정"""
    return [round(value, precision) if isinstance(value, float) else value for value in keypoints]


def map_openpose_to_hrviton(candidate, subset, all_hand_peaks):
    """OpenPose의 Body25와 Hand Keypoints를 HR-VITON JSON 형식으로 변환"""
    json_data = {"version": 1.3, "people": []}
    for person in subset:
        person_data = {
            "person_id": [-1],
            "pose_keypoints_2d": [],
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        }

        # Body Keypoints 매핑
        for idx in range(25):  # Body25 기준
            if idx in person[:len(candidate)]:
                keypoint = candidate[int(idx)]
                person_data["pose_keypoints_2d"].extend([float(keypoint[0]), float(keypoint[1]), float(keypoint[2])])
            else:
                person_data["pose_keypoints_2d"].extend([0.0, 0.0, 0.0])  # Default for missing points

        # Hand Keypoints 추가
        if len(all_hand_peaks) > 0:
            # 왼손
            if len(all_hand_peaks) > 0:
                for peak in all_hand_peaks[0]:
                    person_data["hand_left_keypoints_2d"].extend(
                        [float(peak[0]), float(peak[1]), 1.0 if peak[0] > 0 else 0.0]
                    )
            else:
                person_data["hand_left_keypoints_2d"].extend([0.0] * 63)  # Default

            # 오른손
            if len(all_hand_peaks) > 1:
                for peak in all_hand_peaks[1]:
                    person_data["hand_right_keypoints_2d"].extend(
                        [float(peak[0]), float(peak[1]), 1.0 if peak[0] > 0 else 0.0]
                    )
            else:
                person_data["hand_right_keypoints_2d"].extend([0.0] * 63)  # Default

        # 얼굴 Keypoints는 OpenPose Face 모델 필요
        person_data["face_keypoints_2d"] = [0.0] * 70 * 3  # Placeholder

        # 정밀도 조정
        person_data["pose_keypoints_2d"] = adjust_precision(person_data["pose_keypoints_2d"])
        person_data["hand_left_keypoints_2d"] = adjust_precision(person_data["hand_left_keypoints_2d"])
        person_data["hand_right_keypoints_2d"] = adjust_precision(person_data["hand_right_keypoints_2d"])

        json_data["people"].append(person_data)
    return json_data


def inference_and_save(image_path, body_estimation, hand_estimation, output_json_path):
    oriImg = cv2.imread(image_path)  # B,G,R order
    if oriImg is None:
        print(f"Error: Could not read image from path {image_path}")
        return

    # Body estimation
    candidate, subset = body_estimation(oriImg)

    # Hand estimation
    all_hand_peaks = []
    if hand_estimation is not None:
        hands_list = util.handDetect(candidate, subset, oriImg)
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)

    # JSON 데이터 생성
    json_data = map_openpose_to_hrviton(candidate, subset, all_hand_peaks)

    # JSON 저장 경로와 이름 설정
    json_path = os.path.join(output_json_path, "00001_00_keypoints.json")

    # JSON 저장
    os.makedirs(output_json_path, exist_ok=True)
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON saved at {json_path}")


if __name__ == "__main__":
    # Load models
    body_estimation, hand_estimation = load_model(use_hand=True)

    # 입력 및 출력 경로 설정
    input_path = './input/model.jpg'  # 단일 이미지 파일 경로
    output_json_path = './HR-VITON/test/test/openpose_json'  # JSON 저장 경로

    # Ensure output directory exists
    os.makedirs(output_json_path, exist_ok=True)

    # 단일 이미지 처리
    if not input_path.endswith(('.jpg', '.png')):
        raise ValueError(f"Unsupported file format: {input_path}")

    print(f'Processing: {input_path}')
    inference_and_save(input_path, body_estimation, hand_estimation, output_json_path)