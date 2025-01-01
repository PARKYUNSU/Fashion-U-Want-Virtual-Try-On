import cv2
import os
import json
from openpose import pyopenpose as op

def load_openpose():
    """
    OpenPose 초기화 및 Wrapper 생성
    """
    params = {
        "model_folder": "./models/",  # OpenPose 모델 경로
        "face": True,                # 얼굴 추정 활성화
        "hand": True,                # 손 추정 활성화
    }

    # Wrapper 초기화
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()
    return op_wrapper


def inference_and_save_with_openpose(image_path, op_wrapper, output_json_path):
    """
    OpenPose로 이미지를 추론하고, JSON으로 저장 및 시각화
    """
    # 이미지 읽기
    ori_img = cv2.imread(image_path)
    if ori_img is None:
        print(f"Error: Could not read image from path {image_path}")
        return

    # Datum 생성
    datum = op.Datum()
    datum.cvInputData = ori_img
    op_wrapper.emplaceAndPop([datum])

    # JSON 데이터 생성
    json_data = {"version": 1.3, "people": []}
    person_data = {
        "person_id": [-1],
        "pose_keypoints_2d": datum.poseKeypoints.tolist() if datum.poseKeypoints is not None else [],
        "face_keypoints_2d": datum.faceKeypoints.tolist() if datum.faceKeypoints is not None else [],
        "hand_left_keypoints_2d": datum.handKeypoints[0].tolist() if datum.handKeypoints is not None else [],
        "hand_right_keypoints_2d": datum.handKeypoints[1].tolist() if datum.handKeypoints is not None else [],
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": []
    }
    json_data["people"].append(person_data)

    # JSON 저장
    os.makedirs(output_json_path, exist_ok=True)
    json_path = os.path.join(output_json_path, "00001_00_keypoints.json")
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON saved at {json_path}")

    # 결과 시각화
    cv2.imshow("OpenPose Result", datum.cvOutputData)
    cv2.waitKey(0)


if __name__ == "__main__":
    # OpenPose 초기화
    op_wrapper = load_openpose()

    # 입력 및 출력 경로 설정
    input_path = './input/model.jpg'  # 입력 이미지 경로
    output_json_path = './HR-VITON/test/test/openpose_json'  # JSON 저장 경로