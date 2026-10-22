import cv2
import sys
import cv2
import numpy as np

# 0, 1, 2... を変更してiPhoneのカメラを探す
camera_index = 1

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

print("Python executable:", sys.executable)

print("Python version:", sys.version)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print()
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

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
    print(ret, frame.shape if ret else None, frame.mean() if ret else None)

    # q または ESC で終了
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()