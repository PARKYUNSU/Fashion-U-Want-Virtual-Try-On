import os
import shutil
import warnings
import numpy as np
import cv2
import glob
import argparse
import draw_agnostic

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()

    img=cv2.imread("./input/model.jpg")

    # (768, 1024) resize
    if img.shape[:2] != (1024, 768):
        model_img = cv2.resize(img, (768, 1024))
        print("Resized to (768, 1024)")
    else:
        model_img = img
        print("Already (768, 1024)")

    # save
    cv2.imwrite("./model.jpg", model_img)

    img=cv2.imread("model.jpg")
    img=cv2.resize(img,(384,512))
    cv2.imwrite('resized_img.jpg',img)
    
    # Get mask of cloth
    print("Get mask of cloth\n")
    os.system("python clothseg.py")

    # OpenPose Keypoints JSON 생성
    input_dir = "/content/Fashion-U-Want-Virtual-Try-On/input"
    input_image = "model.jpg"
    output_json_path = "/content/Fashion-U-Want-Virtual-Try-On/HR-VITON/test/test/openpose_json"
    json_filename = "00001_00_keypoints.json"
    os.makedirs(output_json_path, exist_ok=True)

    print("Get OpenPose coordinates\n")
    os.system(f"cd /content/Fashion-U-Want-Virtual-Try-On/openpose && ./build/examples/openpose/openpose.bin "
            f"--image_dir {input_dir} "
            f"--write_json {output_json_path} "
            f"--model_folder ./models/ "
            f"--render_pose 0 "
            f"--display 0")

    generated_json = os.path.join(output_json_path, f"{os.path.splitext(input_image)[0]}_keypoints.json")
    target_json = os.path.join(output_json_path, json_filename)

    if os.path.exists(generated_json):
        shutil.move(generated_json, target_json)
        print(f"JSON file renamed to {target_json}")
    else:
        print(f"Error: Expected JSON file {generated_json} not found.")
    os.chdir("../")

    # Graphonomy 마스크 생성
    print("Generate semantic segmentation using Graphonomy-Master\n")
    os.chdir("/content/Fashion-U-Want-Virtual-Try-On/Graphonomy-master")
    os.system("python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img")
    os.chdir("../")

    output_dir = "./HR-VITON/test/test/image"
    os.makedirs(output_dir, exist_ok=True)
    
    # 사람이 직접 agnostic 마스크 그리기
    print("\n[사용자 입력 필요] Agnostic 영역을 직접 마우스로 지정하세요.\n")
    draw_agnostic.draw_agnostic_mask("model.jpg", "HR-VITON/test/test/agnostic-v3.2/custom_agnostic_mask.png")

    # HR-VITON 실행
    print("\nRun HR-VITON to generate final image\n")
    os.chdir("./HR-VITON")
    os.system("python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test") 

    # 배경 추가 여부
    l=glob.glob("./Output/*.png")

    mask_img=cv2.imread("HR-VITON/test/test/agnostic-v3.2/custom_agnostic_mask.png", cv2.IMREAD_GRAYSCALE)
    back_ground = cv2.imread("./model.jpg")  # 원본 이미지 사용

    if opt.background:
        for i in l:
            img=cv2.imread(i)
            img=cv2.bitwise_and(img, img, mask=mask_img)
            img=img+back_ground
            cv2.imwrite(i, img)
    else:
        for i in l:
            img=cv2.imread(i)
            cv2.imwrite(i, img)

    print("All processing is complete.")
    os.chdir("../")
    cv2.imwrite("./input/finalimg.png", img)