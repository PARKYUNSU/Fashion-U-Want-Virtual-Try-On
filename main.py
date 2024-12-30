import os
import subprocess
import warnings
import numpy as np
import cv2
import glob
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()

    img=cv2.imread("./input/model.jpg")
    # (768, 1024) 크기로 맞추기
    if img.shape[:2] != (1024, 768):  # 높이와 너비 비교
        model_img = cv2.resize(img, (768, 1024))  # OpenCV는 (width, height) 순서로 받음
        print("Resized to (768, 1024)")
    else:
        model_img = img
        print("Already (768, 1024)")

    # 저장
    cv2.imwrite("./model.jpg", model_img)

    img=cv2.imread("model.jpg")
    img=cv2.resize(img,(384,512))
    cv2.imwrite('resized_img.jpg',img)
    
    # Get mask of cloth
    print("Get mask of cloth\n")
    terminnal_command = "python clothseg.py" 
    os.system(terminnal_command)

    # Get openpose coordinate using posenet
    print("Get openpose coordinate using posenet\n")
    terminnal_command = "python openpose2.py" 
    os.system(terminnal_command)

    # Generate semantic segmentation using Graphonomy-Master library
    print("Generate semantic segmentation using Graphonomy-Master library\n")
    os.chdir("./Graphonomy-master")
    terminnal_command ="python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img"
    os.system(terminnal_command)
    os.chdir("../")

    output_dir = "./HR-VITON/test/test/image"
    os.makedirs(output_dir, exist_ok=True)
    # Remove background image using semantic segmentation mask
    mask_img=cv2.imread('./resized_segmentation_img.png',cv2.IMREAD_GRAYSCALE)
    mask_img=cv2.resize(mask_img,(768,1024))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_img = cv2.erode(mask_img, k)
    img_seg=cv2.bitwise_and(model_img,model_img,mask=mask_img)
    back_ground=model_img-img_seg
    img_seg=np.where(img_seg==0,215,img_seg)

    cv2.imwrite("./seg_img.png",img_seg)
    img=cv2.resize(img_seg,(768,1024))
    cv2.imwrite('./HR-VITON/test/test/image/00001_00.jpg',img)

    terminnal_command ="python grayscale.py"
    os.system(terminnal_command)

    # Generate Densepose image using detectron2 library
    print("\nGenerate Densepose image using detectron2 library\n")
    terminnal_command ="python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    model.jpg --output output.pkl -v"
    os.system(terminnal_command)
    terminnal_command ="python get_densepose.py"
    os.system(terminnal_command)

    # Run HR-VITON to generate final image
    print("\nRun HR-VITON to generate final image\n")
    os.chdir("./HR-VITON")
    terminnal_command = "python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test" 
    os.system(terminnal_command)

    # Add Background or Not
    l=glob.glob("./Output/*.png")

    # Add Background
    if opt.background:
        for i in l:
            img=cv2.imread(i)
            img=cv2.bitwise_and(img,img,mask=mask_img)
            img=img+back_ground
            cv2.imwrite(i,img)

    # Remove Background
    else:
        for i in l:
            img=cv2.imread(i)
            cv2.imwrite(i,img)

    print("All processing is complete.")
    os.chdir("../")
    cv2.imwrite("./input/finalimg.png", img)