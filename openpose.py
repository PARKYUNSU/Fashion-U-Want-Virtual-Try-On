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

def inference_and_save(image_path, body_estimation, hand_estimation, output_json_path):
    oriImg = cv2.imread(image_path)  # B,G,R order
    if oriImg is None:
        print(f"Error: Could not read image from path {image_path}")
        return

    # Body estimation
    candidate, subset = body_estimation(oriImg)

    # 검정 배경 생성 (원본 이미지 크기와 동일)
    canvas = np.zeros_like(oriImg)

    # Body pose 그리기
    canvas = util.draw_bodypose(canvas, candidate, subset, 'body25')

    # Hand estimation
    all_hand_peaks = []
    if hand_estimation is not None:
        hands_list = util.handDetect(candidate, subset, oriImg)
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)

    # JSON 저장 경로와 이름 설정
    json_path = os.path.join(output_json_path, "00001_00_keypoints.json")

    # Prepare JSON output
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

        # Add body keypoints
        for idx in range(len(candidate)):
            if idx in person[:len(candidate)]:
                keypoint = candidate[int(idx)]
                person_data["pose_keypoints_2d"].extend([float(keypoint[0]), float(keypoint[1]), float(keypoint[2])])
            else:
                person_data["pose_keypoints_2d"].extend([0.0, 0.0, 0.0])  # Default for missing points

        # Add hand keypoints
        if len(all_hand_peaks) > 0:
            if len(all_hand_peaks) > 0:  # Left hand
                for peak in all_hand_peaks[0]:
                    person_data["hand_left_keypoints_2d"].extend(
                        [float(peak[0]), float(peak[1]), 1.0 if peak[0] > 0 else 0.0]
                    )
            else:
                person_data["hand_left_keypoints_2d"].extend([0.0] * 63)
            if len(all_hand_peaks) > 1:  # Right hand
                for peak in all_hand_peaks[1]:
                    person_data["hand_right_keypoints_2d"].extend(
                        [float(peak[0]), float(peak[1]), 1.0 if peak[0] > 0 else 0.0]
                    )
            else:
                person_data["hand_right_keypoints_2d"].extend([0.0] * 63)

        json_data["people"].append(person_data)

    os.makedirs(output_json_path, exist_ok=True)
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON saved at {json_path}")


if __name__ == "__main__":
    # Load models
    body_estimation, hand_estimation = load_model(use_hand=True)

    # 입력 및 출력 경로 설정
    input_path = './input/image'
    output_json_path = './HR-VITON-main/test/test/openpose_json'  # JSON 저장 경로

    # Ensure output directory exists
    os.makedirs(output_json_path, exist_ok=True)

    # 입력 디렉토리 내 모든 이미지 처리
    for image_name in os.listdir(input_path):
        if not image_name.endswith(('.jpg', '.png')):  # 이미지 파일만 처리
            continue

        image_path = os.path.join(input_path, image_name)
        print(f'Processing: {image_path}')
        inference_and_save(image_path, body_estimation, hand_estimation, output_json_path)
