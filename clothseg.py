import os
import numpy as np
from PIL import Image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
import cv2


def process_images(input_path="input/cloth", 
                   cloth_output_path="./HR-VITON-main/test/test/cloth",
                   mask_output_path="./HR-VITON-main/test/test/cloth-mask",
                   segmentation_network="u2net", 
                   preprocessing_method="none", 
                   postprocessing_method="none", 
                   seg_mask_size=320, 
                   trimap_dilation=30, 
                   trimap_erosion=5, 
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

    # Collect all images from the input path
    imgs = [os.path.join(input_path, name) for name in os.listdir(input_path) if name.endswith(('jpg', 'jpeg', 'png'))]

    # Process images
    processed_images = interface(imgs)

    # Save the output masks and original images
    for i, im in enumerate(processed_images):
        # Original image
        original_image = cv2.imread(imgs[i])
        
        # Preprocessing: Ensure image size compatibility
        if original_image.shape[1] <= 600 and original_image.shape[0] <= 500:
            original_image = cv2.resize(original_image, 
                                        (int(original_image.shape[1] * 1.2), int(original_image.shape[0] * 1.2)))
        
        # Segmentation mask
        img = np.array(im)
        img = img[..., :3]  # Remove alpha channel if exists
        idx = ((img[..., 0] == 0) & (img[..., 1] == 0) & (img[..., 2] == 0)) | \
              ((img[..., 0] == 130) & (img[..., 1] == 130) & (img[..., 2] == 130))
        mask = np.ones(idx.shape) * 255
        mask[idx] = 0  # Convert background to black and object to white

        # Resize the mask if needed
        mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create output canvas
        canvas_shape = (1024, 768, 3)
        output_canvas = np.full(canvas_shape, 255, dtype=np.uint8)
        mask_canvas = np.zeros((1024, 768), dtype=np.uint8)

        # Center the image and mask
        h, w, _ = original_image.shape
        y_offset = (1024 - h) // 2
        x_offset = (768 - w) // 2
        output_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = original_image
        mask_canvas[y_offset:y_offset + h, x_offset:x_offset + w] = mask_resized

        # Save results
        output_name = "00001_00"  # 고정된 파일 이름
        cv2.imwrite(os.path.join(cloth_output_path, f"{output_name}.jpg"), output_canvas)
        cv2.imwrite(os.path.join(mask_output_path, f"{output_name}.jpg"), mask_canvas)


if __name__ == "__main__":
    input_path = "input/cloth"  # Path to input images
    cloth_output_path = "./HR-VITON-main/test/test/cloth"  # Path to save original images
    mask_output_path = "./HR-VITON-main/test/test/cloth-mask"  # Path to save masks
    process_images(input_path, cloth_output_path, mask_output_path)
