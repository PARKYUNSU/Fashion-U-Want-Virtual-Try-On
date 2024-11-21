import os
import numpy as np
from PIL import Image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

def process_images(output_path, 
                   input_path="input/cloth",  # 기본 입력 경로 설정
                   segmentation_network="u2net", 
                   preprocessing_method="none", 
                   postprocessing_method="none", 
                   seg_mask_size=320, 
                   trimap_dilation=30, 
                   trimap_erosion=5, 
                   device='cpu'):
    
    # Initialize the configuration
    config = MLConfig(segmentation_network=segmentation_network,
                      preprocessing_method=preprocessing_method,
                      postprocessing_method=postprocessing_method,
                      seg_mask_size=seg_mask_size,
                      trimap_dilation=trimap_dilation,
                      trimap_erosion=trimap_erosion,
                      device=device)
    interface = init_interface(config)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Collect all images from the input path
    imgs = [os.path.join(input_path, name) for name in os.listdir(input_path) if name.endswith(('jpg', 'jpeg', 'png'))]

    # Process images
    processed_images = interface(imgs)

    # Save the output masks
    for i, im in enumerate(processed_images):
        img = np.array(im)
        img = img[..., :3]  # Remove alpha channel if exists
        idx = ((img[..., 0] == 0) & (img[..., 1] == 0) & (img[..., 2] == 0)) | \
            ((img[..., 0] == 130) & (img[..., 1] == 130) & (img[..., 2] == 130))
        mask = np.ones(idx.shape) * 255
        mask[idx] = 0  # Convert background to white and object to black
        output_image = Image.fromarray(np.uint8(mask), 'L')
        output_name = os.path.basename(imgs[i]).split(".")[0] + ".jpg"
        output_image.save(os.path.join(output_path, output_name))

if __name__ == "__main__":
    input_path = "input/cloth"  # Path to input images
    output_path = "output/cloth_segment"  # Path to save processed masks
    process_images(input_path, output_path)
