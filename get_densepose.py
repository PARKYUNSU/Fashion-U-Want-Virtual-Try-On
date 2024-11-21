import os
import cv2
from PIL import Image
import json
import numpy as np

# Colormap 설정
colormap = {
    2 : [20, 80, 194],
    3 : [4, 98, 224],
    4 : [8, 110, 221],
    9 : [6, 166, 198],
    10 : [22, 173, 184],
    15 : [145, 191, 116],
    16 : [170, 190, 105],
    17 : [191, 188, 97],
    18 : [216, 187, 87],
    19 : [228, 191, 74],
    20 : [240, 198, 60],
    21 : [252, 205, 47],
    22 : [250, 220, 36],
    23 : [251, 235, 25],
    24 : [248, 251, 14],
}

# 경로 설정
input_path = './input/image'
output_path = './output/image-densepose'

# 출력 디렉토리가 존재하지 않으면 생성
os.makedirs(output_path, exist_ok=True)

# input_path 내의 모든 이미지 파일 처리
for file_name in os.listdir(input_path):
    if file_name.endswith('.jpg'):  # .jpg 파일만 처리
        img_file = os.path.join(input_path, file_name)
        json_file = os.path.join(input_path, file_name.replace('.jpg', '.json'))

        # JSON 파일이 존재하지 않으면 건너뜀
        if not os.path.exists(json_file):
            print(f"Warning: JSON file for {file_name} not found. Skipping.")
            continue

        # 이미지 및 JSON 데이터 로드
        img = Image.open(img_file)
        img_w, img_h = img.size

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        i = np.array(json_data[0])

        # 세그멘테이션 이미지 생성
        seg_img = np.zeros((i.shape[0], i.shape[1], 3))

        for y_idx in range(i.shape[0]):
            for x_idx in range(i.shape[1]):
                if i[y_idx][x_idx] in colormap:
                    seg_img[y_idx][x_idx] = colormap[i[y_idx][x_idx]]
                else:
                    seg_img[y_idx][x_idx] = [0, 0, 0]

        # 박스 정보 처리
        box = json_data[2]
        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        x, y, w, h = [int(v) for v in box]

        # 배경 이미지 생성
        bg = np.zeros((img_h, img_w, 3))
        bg[y:y+h, x:x+w, :] = seg_img

        # 결과 저장
        output_file = os.path.join(output_path, file_name.replace('.jpg', '_densepose.jpg'))
        bg_img = Image.fromarray(np.uint8(bg), "RGB")
        bg_img.save(output_file)

        print(f"Processed {file_name}, saved to {output_file}")