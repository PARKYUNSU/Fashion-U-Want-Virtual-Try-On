import os
import cv2
from PIL import Image
import json
import numpy as np


# Define the colormap
colormap = {
    2: [20, 80, 194],
    3: [4, 98, 224],
    4: [8, 110, 221],
    9: [6, 166, 198],
    10: [22, 173, 184],
    15: [145, 191, 116],
    16: [170, 190, 105],
    17: [191, 188, 97],
    18: [216, 187, 87],
    19: [228, 191, 74],
    20: [240, 198, 60],
    21: [252, 205, 47],
    22: [250, 220, 36],
    23: [251, 235, 25],
    24: [248, 251, 14],
}

# Load the input image and its dimensions
img = Image.open('./model.jpg')
img_w, img_h = img.size

# Load the JSON data
with open('./data.json', 'r') as f:
    json_data = json.load(f)

# Convert JSON segmentation data to a NumPy array
segmentation_data = np.array(json_data[0])

# Create an empty image for the segmentation
seg_img = np.zeros((segmentation_data.shape[0], segmentation_data.shape[1], 3), dtype=np.uint8)

# Assign colors to the segmentation image based on the colormap
for key, color in colormap.items():
    seg_img[segmentation_data == key] = color

# Extract bounding box data
box = json_data[2]
box[2] = box[2] - box[0]  # Convert to width
box[3] = box[3] - box[1]  # Convert to height
x, y, w, h = [int(v) for v in box]

# Create the background image
bg = np.zeros((img_h, img_w, 3), dtype=np.uint8)
bg[y:y + h, x:x + w, :] = seg_img

# Convert the background image to a PIL image
bg_img = Image.fromarray(bg, "RGB")

# Ensure the output directory exists
output_dir = "./HR-VITON/test/test/image-densepose/"
os.makedirs(output_dir, exist_ok=True)

# Save the output image
output_file = os.path.join(output_dir, "00001_00.jpg")
bg_img.save(output_file)

print(f"Image saved at {output_file}")