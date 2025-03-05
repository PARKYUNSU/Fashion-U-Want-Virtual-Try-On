import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

def draw_agnostic_mask_colab(img_path, output_path):
    """
    Colab 환경에서 마우스로 Agnostic Mask를 그리는 함수.
    """
    # 원본 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: '{img_path}' Cannot find the File")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 이미지를 RGB로 변환
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Colab에서는 cv2.imshow() 대신 matplotlib 사용
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title("Agnostic Mask를 그릴 영역을 클릭하세요. 완료 후 ENTER 키를 누르세요.")

    # 사용자 입력 받기 (마우스로 클릭한 좌표를 반환)
    points = plt.ginput(n=-1, timeout=0)  # 무제한 클릭 가능
    plt.close()

    # 클릭한 점을 흰색 마스크로 변환
    for x, y in points:
        cv2.circle(mask, (int(x), int(y)), 10, (255, 255, 255), -1)

    # 결과 마스크 출력 (Colab에서 보기 위해 `cv2_imshow()` 사용)
    print("생성된 Agnostic Mask:")
    cv2_imshow(mask)

    # 마스크 저장
    cv2.imwrite(output_path, mask)
    print(f"Agnostic Saved: {output_path}")

# 직접 실행할 때만 실행하도록 설정
if __name__ == "__main__":
    img_path = "model.jpg"
    output_path = "HR-VITON/test/test/agnostic-v3.2/custom_agnostic_mask.png"
    draw_agnostic_mask_colab(img_path, output_path)