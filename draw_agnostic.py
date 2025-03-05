import cv2
import numpy as np
import os

def draw_agnostic_mask(img_path, output_path):
    drawing = False  # 마우스 드로잉 활성화 여부
    ix, iy = -1, -1  # 마우스 클릭 위치

    def draw_mask(event, x, y, flags, param):
        nonlocal drawing, ix, iy, mask
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)  # 흰색 마스크 영역

        elif event == cv2.EVENT_LBUTTONUP:  # 마우스 클릭 해제 시
            drawing = False
            cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)

    # 원본 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: '{img_path}' Cannot find the File")
        return

    h, w, _ = img.shape

    # 빈 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)

    # 창 만들기 및 마우스 콜백 설정
    cv2.namedWindow('Draw Agnostic Mask')
    cv2.setMouseCallback('Draw Agnostic Mask', draw_mask)

    while True:
        display_img = img.copy()
        display_img[mask == 255] = (0, 0, 255)  # 마스크 영역을 빨간색으로 표시
        cv2.imshow('Draw Agnostic Mask', display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)

    cv2.destroyAllWindows()
    print(f"Agnostic Saved: {output_path}")

if __name__ == "__main__":
    img_path = "model.jpg"
    output_path = "HR-VITON/test/test/agnostic-v3.2/custom_agnostic_mask.png"
    draw_agnostic_mask(img_path, output_path)