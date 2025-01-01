import cv2
import json
import os

def visualize_keypoints(image_path, json_path, output_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from path {image_path}")
        return

    # JSON 로드
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    
    # Keypoints 시각화
    for person in data["people"]:
        # Pose Keypoints
        pose_keypoints = person["pose_keypoints_2d"]
        for i in range(0, len(pose_keypoints), 3):
            x, y, confidence = pose_keypoints[i], pose_keypoints[i+1], pose_keypoints[i+2]
            if confidence > 0.5:  # Confidence threshold
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circle
        
        # Left Hand Keypoints
        left_hand_keypoints = person["hand_left_keypoints_2d"]
        for i in range(0, len(left_hand_keypoints), 3):
            x, y, confidence = left_hand_keypoints[i], left_hand_keypoints[i+1], left_hand_keypoints[i+2]
            if confidence > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)  # Blue circle
        
        # Right Hand Keypoints
        right_hand_keypoints = person["hand_right_keypoints_2d"]
        for i in range(0, len(right_hand_keypoints), 3):
            x, y, confidence = right_hand_keypoints[i], right_hand_keypoints[i+1], right_hand_keypoints[i+2]
            if confidence > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circle

    # 출력 경로 생성
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "visualized_image2.jpg")

    # 결과 저장
    cv2.imwrite(output_file, image)
    print(f"Visualized image saved at {output_file}")


if __name__ == "__main__":
    # 입력 경로
    image_path = "00001_00.jpg"  # 원본 이미지 경로
    json_path = "original_00001_00_keypoints.json"  # 생성된 JSON 경로
    output_path = "."  # 결과 이미지 저장 경로

    # 시각화 실행
    visualize_keypoints(image_path, json_path, output_path)