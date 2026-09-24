#!/usr/bin/env python3

import cv2


# 0, 1, 2... を変更してiPhoneのカメラを探す
camera_index = 1

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("カメラを開けませんでした")
    raise SystemExit(1)

# 実際に取得されている解像度を確認
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera index: {camera_index}")
print(f"Resolution: {width:.0f} x {height:.0f}")
print(f"FPS: {fps}")

while True:
    ret, frame = cap.read()

    if not ret:
        print("フレーム取得失敗")
        break

    cv2.imshow("Simple OpenCV Camera Test", frame)

    # q または ESC で終了
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
