import os
import cv2
from PIL import Image
import numpy as np

# 입력 및 출력 경로 설정
input_dir = './output/human_parse'
output_dir = './output/image-parse-v3'

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 입력 디렉토리의 모든 이미지 처리
for image_name in os.listdir(input_dir):
    if not image_name.endswith(('.jpg', '.png')):  # 이미지 파일만 처리
        continue

    # 입력 및 출력 경로 설정
    input_path = os.path.join(input_dir, image_name)
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.png")

    # 이미지 로드 및 처리
    img = Image.open(input_path)
    img_w, img_h = img.size
    img = np.array(img)
    gray_img = np.zeros((img_h, img_w))

    for y_idx in range(img.shape[0]):
        for x_idx in range(img.shape[1]):    
            tmp = img[y_idx][x_idx]
            if np.array_equal(tmp, [0, 0, 0]):
                gray_img[y_idx][x_idx] = 0
            elif np.array_equal(tmp, [255, 0, 0]):
                gray_img[y_idx][x_idx] = 2  # 머리카락
            elif np.array_equal(tmp, [0, 0, 255]):
                gray_img[y_idx][x_idx] = 13  # 머리
            elif np.array_equal(tmp, [85, 51, 0]):
                gray_img[y_idx][x_idx] = 10  # 목
            elif np.array_equal(tmp, [255, 85, 0]):
                gray_img[y_idx][x_idx] = 5  # 몸통
            elif np.array_equal(tmp, [0, 255, 255]):
                gray_img[y_idx][x_idx] = 15  # 왼팔
            elif np.array_equal(tmp, [51, 170, 221]):
                gray_img[y_idx][x_idx] = 14  # 오른팔
            elif np.array_equal(tmp, [0, 85, 85]):
                gray_img[y_idx][x_idx] = 9  # 바지
            elif np.array_equal(tmp, [0, 0, 85]):
                gray_img[y_idx][x_idx] = 6  # 원피스
            elif np.array_equal(tmp, [0, 128, 0]):
                gray_img[y_idx][x_idx] = 12  # 치마
            elif np.array_equal(tmp, [177, 255, 85]):
                gray_img[y_idx][x_idx] = 17  # 왼다리
            elif np.array_equal(tmp, [85, 255, 170]):
                gray_img[y_idx][x_idx] = 16  # 오른다리
            elif np.array_equal(tmp, [0, 119, 221]):
                gray_img[y_idx][x_idx] = 5  # 외투
            else:
                gray_img[y_idx][x_idx] = 0

    # 크기 조정 및 저장
    img = cv2.resize(gray_img, (768, 1024), interpolation=cv2.INTER_NEAREST)
    bg_img = Image.fromarray(np.uint8(img), "L")
    bg_img.save(output_path)

print("Processing complete. Output saved in:", output_dir)