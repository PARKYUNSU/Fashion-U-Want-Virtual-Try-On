import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic

if __name__ =="__main__":
    data_path = './HR-VITON/test/test'
    output_path = './HR-VITON/test/test/agnostic-v3.2'
    
    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        # load pose image
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        pose_json_path = osp.join(data_path, 'openpose_json', pose_name)

        if not osp.exists(pose_json_path):
            print(f"Pose JSON file not found: {pose_json_path}")
            continue

        try:
            with open(pose_json_path, 'r') as f:
                pose_label = json.load(f)
                if not pose_label['people']:
                    print(f"No people found in pose JSON file: {pose_name}")
                    continue
                pose_data = pose_label['people'][0].get('pose_keypoints_2d', [])
                if not pose_data:
                    print(f"pose_keypoints_2d is empty in file: {pose_name}")
                    continue
                pose_data = np.array(pose_data).reshape((-1, 3))[:, :2]
        except Exception as e:
            print(f"Error processing pose file {pose_name}: {e}")
            continue

        # load parsing image
        image_path = osp.join(data_path, 'image', im_name)
        label_name = im_name.replace('.jpg', '.png')
        label_path = osp.join(data_path, 'image-parse-v3', label_name)

        if not osp.exists(image_path) or not osp.exists(label_path):
            print(f"Image or label file missing for {im_name}")
            continue

        im = Image.open(image_path)
        im_label = Image.open(label_path)

        try:
            agnostic = get_img_agnostic(im, im_label, pose_data)
            agnostic.save(osp.join(output_path, im_name))
        except Exception as e:
            print(f"Error generating agnostic image for {im_name}: {e}")
            continue