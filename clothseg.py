import os
import numpy as np
from PIL import Image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
import cv2


def process_images(input_path="./input/cloth.jpg", 
                   cloth_output_path="./HR-VITON/test/test/cloth",
                   mask_output_path="./HR-VITON/test/test/cloth-mask",
                   segmentation_network="u2net", 
                   preprocessing_method="none", 
                   postprocessing_method="none", 
                   seg_mask_size=320, 
                   trimap_dilation=50, 
                   trimap_erosion=10, 
                   device='cuda'):
    
    # Initialize the configuration
    config = MLConfig(segmentation_network=segmentation_network,
                      preprocessing_method=preprocessing_method,
                      postprocessing_method=postprocessing_method,
                      seg_mask_size=seg_mask_size,
                      trimap_dilation=trimap_dilation,
                      trimap_erosion=trimap_erosion,
                      device=device)
    interface = init_interface(config)

    # Ensure output directories exist
    os.makedirs(cloth_output_path, exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)

    # Read the single input image
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input path is not a valid file: {input_path}")

    print(f"Processing file: {input_path}")
    original_image = cv2.imread(input_path)
    if original_image is None:
        raise ValueError(f"Failed to read the image file: {input_path}")

    # Process the image
    processed_image = interface([input_path])[0]

    # Preprocessing: Ensure image size compatibility
    if original_image.shape[1] <= 600 and original_image.shape[0] <= 500:
        original_image = cv2.resize(original_image, 
                                    (int(original_image.shape[1] * 1.2), int(original_image.shape[0] * 1.2)))
    
    # Segmentation mask
    img = np.array(processed_image)
    img = img[..., :3]  # Remove alpha channel if exists
    idx = ((img[..., 0] == 0) & (img[..., 1] == 0) & (img[..., 2] == 0)) | \
          ((img[..., 0] == 130) & (img[..., 1] == 130) & (img[..., 2] == 130))
    mask = np.ones(idx.shape) * 255
    mask[idx] = 0  # Convert background to black and object to white

    # Resize the mask if needed
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create output canvas with fixed size
    canvas_shape = (1024, 768, 3)  # Fixed output canvas size (Height x Width x Channels)
    output_canvas = np.full(canvas_shape, 255, dtype=np.uint8)  # Fill with white background
    mask_canvas = np.zeros((1024, 768), dtype=np.uint8)  # Mask canvas initialized to black

    # Center the original image and mask on the canvas
    h, w, _ = original_image.shape
    y_offset = (canvas_shape[0] - h) // 2  # Calculate vertical offset
    x_offset = (canvas_shape[1] - w) // 2  # Calculate horizontal offset

    # Check if the image fits into the canvas
    if h > canvas_shape[0] or w > canvas_shape[1]:
        print(f"Resizing original image to fit into canvas of size {canvas_shape[:2]}")
        scaling_factor = min(canvas_shape[0] / h, canvas_shape[1] / w)
        new_h = int(h * scaling_factor)
        new_w = int(w * scaling_factor)
        original_image = cv2.resize(original_image, (new_w, new_h))
        mask_resized = cv2.resize(mask_resized, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        h, w = original_image.shape[:2]  # Update dimensions after resizing
        y_offset = (canvas_shape[0] - h) // 2
        x_offset = (canvas_shape[1] - w) // 2

    # Place the resized image and mask onto the canvas
    output_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = original_image
    mask_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = mask_resized

    # Save results
    output_name = "00001_00"  # Fixed output name
    cv2.imwrite(os.path.join(cloth_output_path, f"{output_name}.jpg"), output_canvas)
    cv2.imwrite(os.path.join(mask_output_path, f"{output_name}.jpg"), mask_canvas)


if __name__ == "__main__":
    input_path = "./input/cloth.jpg"  # Path to input image
    cloth_output_path = "./HR-VITON/test/test/cloth"  # Path to save processed images
    mask_output_path = "./HR-VITON/test/test/cloth-mask"  # Path to save masks
    process_images(input_path, cloth_output_path, mask_output_path)