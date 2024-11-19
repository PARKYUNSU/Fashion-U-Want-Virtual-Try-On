import json
import cv2
import matplotlib.pyplot as plt


def plot_keypoints(image_path, json_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract keypoints
    for person in data["people"]:
        # Plot pose keypoints
        pose_keypoints = person["pose_keypoints_2d"]
        for i in range(0, len(pose_keypoints), 3):
            x, y, confidence = pose_keypoints[i:i + 3]
            if confidence > 0.5:  # Only plot confident keypoints
                cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

        # Plot left hand keypoints
        left_hand_keypoints = person["hand_left_keypoints_2d"]
        for i in range(0, len(left_hand_keypoints), 3):
            x, y, confidence = left_hand_keypoints[i:i + 3]
            if confidence > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Plot right hand keypoints
        right_hand_keypoints = person["hand_right_keypoints_2d"]
        for i in range(0, len(right_hand_keypoints), 3):
            x, y, confidence = right_hand_keypoints[i:i + 3]
            if confidence > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Show image with keypoints
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Example usage
    image_path = "/Users/parkyunsu/Downloads/00002_00.jpg"  # 이미지 파일 경로
    json_path = "/Users/parkyunsu/project/Fashion_Segmentation/1.Openpose/test_results/result_keypoints.json"  # JSON 파일 경로
    plot_keypoints(image_path, json_path)
