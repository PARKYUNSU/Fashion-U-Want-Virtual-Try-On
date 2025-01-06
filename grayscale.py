import cv2
from PIL import Image
import pickle
import json
import numpy as np
import os


img = Image.open('./resized_segmentation_img.png')
img_w ,img_h = img.size

img = np.array(img)
gray_img=np.zeros((img_h,img_w))

for y_idx in range(img.shape[0]):
    for x_idx in range(img.shape[1]):    
        tmp = img[y_idx][x_idx]
        if np.array_equal(tmp, [0,0,0]):
            gray_img[y_idx][x_idx] = 0
        if np.array_equal(tmp, [255,0,0]):
            gray_img[y_idx][x_idx] = 2 # hair
        elif np.array_equal(tmp, [0,0,255]):
            gray_img[y_idx][x_idx] = 13 # head
        elif np.array_equal(tmp, [85, 51, 0]):
            gray_img[y_idx][x_idx] = 10 # neck
        elif np.array_equal(tmp, [255, 85, 0]):
            gray_img[y_idx][x_idx] = 5 # body
        elif np.array_equal(tmp, [0, 255, 255]):
            gray_img[y_idx][x_idx] = 15 # left arm
        elif np.array_equal(tmp, [51, 170, 221]):
            gray_img[y_idx][x_idx] = 14 # right arm
        elif np.array_equal(tmp, [0, 85, 85]):
            gray_img[y_idx][x_idx] = 9 # pants
        elif np.array_equal(tmp, [0, 0, 85]):
            gray_img[y_idx][x_idx] = 6 # dresser
        elif np.array_equal(tmp, [0, 128, 0]):
            gray_img[y_idx][x_idx] = 12 # skirt
        elif np.array_equal(tmp, [177, 255, 85]):
            gray_img[y_idx][x_idx] = 17 # left leg
        elif np.array_equal(tmp, [85, 255, 170]):
            gray_img[y_idx][x_idx] = 16 # right leg
        elif np.array_equal(tmp, [0, 119, 221]):
            gray_img[y_idx][x_idx] = 5 # outer
        else:
            gray_img[y_idx][x_idx] = 0

# Resize the grayscale image
img = cv2.resize(gray_img, (768, 1024), interpolation=cv2.INTER_NEAREST)

# Convert to PIL Image
bg_img = Image.fromarray(np.uint8(img), "L")

# Ensure the output directory exists
output_dir = "./HR-VITON/test/test/image-parse-v3/"
os.makedirs(output_dir, exist_ok=True)

# Save the image to the specified path
output_file = os.path.join(output_dir, "00001_00.png")
bg_img.save(output_file)
print(f"Image saved to {output_file}")