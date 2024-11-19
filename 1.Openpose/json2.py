import cv2
import os
import json
import numpy as np
from src import util
from src.body import Body
from src.hand import Hand


def load_model(model_type='coco', use_hand=False):
    if model_type == 'body25':
        model_path = 'model/pose_iter_584000.caffemodel.pt'
    else:
        model_path = './model/body_pose_model.pth'
    body_estimation = Body(model_path, model_type)
    hand_estimation = Hand('model/hand_pose_model.pth') if use_hand else None
    return body_estimation, hand_estimation


def draw_body_and_hands(canvas, candidate, subset, all_hand_peaks):
    """
    Draw BodyPose and HandPose separately with connecting lines.
    """
    # Draw BodyPose
    canvas = util.draw_bodypose(canvas, candidate, subset, model_type='body25')

    # Draw HandPose
    for peaks in all_hand_peaks:
        for edge in [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], 
                     [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
                     [0, 17], [17, 18], [18, 19], [19, 20]]:
            start, end = peaks[edge[0]], peaks[edge[1]]
            if np.any(start) and np.any(end):  # Both points exist
                cv2.line(canvas, tuple(map(int, start[:2])), tuple(map(int, end[:2])), (0, 255, 0), 2)
    return canvas


def inference_and_save_json(image_path, model_type, body_estimation, hand_estimation, output_json_path):
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

    # Draw on canvas
    canvas = draw_body_and_hands(oriImg, candidate, subset, all_hand_peaks)

    # Save the resulting image
    output_image_path = output_json_path.replace(".json", ".jpg")
    cv2.imwrite(output_image_path, canvas)

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

    # Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON saved at {output_json_path}")


if __name__ == "__main__":
    # Load models
    model_type = 'body25'  # Options: 'body25' or 'coco'
    body_estimation, hand_estimation = load_model(model_type=model_type, use_hand=True)

    # Input image path
    test_image_path = '/Users/parkyunsu/Downloads/00022_00.jpg'
    output_json_path = 'test_results/result_keypoints.json'

    # Run inference and save JSON
    inference_and_save_json(test_image_path, model_type, body_estimation, hand_estimation, output_json_path)
