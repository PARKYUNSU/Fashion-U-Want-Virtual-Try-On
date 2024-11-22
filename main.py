import os
import warnings
import cv2
import glob
import argparse
from openpose import load_model, inference_and_save
from clothseg import process_images

warnings.filterwarnings("ignore", category=FutureWarning)

def process_densepose(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 결과 디렉토리 생성

    for image_name in os.listdir(input_dir):
        if not image_name.endswith(('.jpg', '.png')):  # 이미지 파일만 처리
            continue

        input_image_path = os.path.join(input_dir, image_name)
        output_pkl_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_output.pkl")
        output_image_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_densepose.png")

        # DensePose 추론
        print(f"\nProcessing DensePose for {image_name}\n")
        detectron_command = (
            f"python detectron2/projects/DensePose/apply_net.py dump "
            f"detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml "
            f"https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl "
            f"{input_image_path} --output {output_pkl_path} -v"
        )
        os.system(detectron_command)

        # 후처리 스크립트 호출
        print(f"\nGenerating DensePose image for {image_name}\n")
        densepose_command = f"python get_densepose.py --input {output_pkl_path} --output {output_image_path}"
        os.system(densepose_command)

        print(f"Saved DensePose result to {output_image_path}")


def process_graphonomy(input_path, output_path):
    """
    Graphonomy-Master를 사용한 세분화 결과 생성
    """
    print("Generate semantic segmentation using Graphonomy-Master library\n")

    # 절대 경로로 img_path 설정
    absolute_input_path = os.path.abspath(input_path)
    absolute_output_path = os.path.abspath(output_path)

    # Graphonomy-Master 실행
    terminal_command = (
        f"python ./Graphonomy-master/exp/inference/inference.py "
        f"--loadmodel ./Graphonomy-master/inference.pth "
        f"--img_path {absolute_input_path} --output_path {absolute_output_path} "
        f"--output_name resized_segmentation_img"
    )
    print(f"Executing: {terminal_command}")
    os.system(terminal_command)

def run_hr_viton(background_flag):
    """
    HR-VITON 실행 및 배경 추가/제거 처리
    """
    print("\nRun HR-VITON to generate final image\n")
    os.chdir("./HR-VITON-main")
    terminal_command = (
        "python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth "
        "--gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test"
    )
    os.system(terminal_command)
    os.chdir("../")

    print("\nProcessing HR-VITON output images")
    output_images = glob.glob("./HR-VITON-main/Output/*.png")
    mask_img = cv2.imread('./output/image-densepose/mask.png', cv2.IMREAD_GRAYSCALE)
    background = cv2.imread('./input/background.jpg')

    for image_path in output_images:
        img = cv2.imread(image_path)
        if background_flag and mask_img is not None:
            img = cv2.bitwise_and(img, img, mask=mask_img)
            img = cv2.add(img, background)
        cv2.imwrite(image_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', action='store_true', help='Add background to the final image')
    opt = parser.parse_args()

    # OpenPose 모델 로드
    body_estimation, hand_estimation = load_model(use_hand=True)

    # 경로 설정
    input_path = './input/image'
    output_openpose_image_path = './output/openpose_img'
    output_openpose_json_path = './output/openpose_json'
    output_densepose_path = './output/image-densepose'
    output_clothseg_path = './output/cloth_segment'
    output_parse_path = './output/human_parse'

    # 출력 디렉토리 생성
    os.makedirs(output_openpose_image_path, exist_ok=True)
    os.makedirs(output_openpose_json_path, exist_ok=True)
    os.makedirs(output_densepose_path, exist_ok=True)
    os.makedirs(output_clothseg_path, exist_ok=True)
    os.makedirs(output_parse_path, exist_ok=True)

    # 입력 디렉토리의 모든 이미지 처리
    for image_name in os.listdir(input_path):
        if not image_name.endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(input_path, image_name)

        # OpenPose 처리
        print(f'\nProcessing OpenPose for: {image_path}')
        inference_and_save(image_path, body_estimation, hand_estimation, output_openpose_image_path, output_openpose_json_path)

    # DensePose 처리
    print(f'\nProcessing DensePose for images in: {input_path}')
    process_densepose(input_path, output_densepose_path)

    # Cloth Segmentation 처리
    print(f'\nProcessing Cloth Segmentation for images in: input/cloth')
    process_images(output_path=output_clothseg_path)

    # Graphonomy-Master 처리
    process_graphonomy(input_path, output_parse_path)

    # HR-VITON 실행
    run_hr_viton(opt.background)

    print("All processing is complete.")


if __name__ == "__main__":
    main()
