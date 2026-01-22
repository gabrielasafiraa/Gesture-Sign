import os
import cv2

# ================== KONFIGURASI ==================
DATA_DIR = './data'
NUMBER_OF_CLASSES = 9
DATASET_SIZE = 100
CAMERA_INDEX = 0

# =================================================

os.makedirs(DATA_DIR, exist_ok=True)

# Paksa DirectShow (WAJIB di Windows)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Kamera tidak bisa dibuka")
    exit()

for class_id in range(NUMBER_OF_CLASSES):
    class_path = os.path.join(DATA_DIR, str(class_id))
    os.makedirs(class_path, exist_ok=True)

    print(f"\n📸 Collecting data for class {class_id}")
    print("👉 Tekan tombol S untuk mulai, Q untuk keluar")

    # --------- WAIT UNTIL 'S' ---------
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(
            frame,
            f'Class {class_id} - Press S to start',
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow('Collect Images', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # --------- COLLECT IMAGES ---------
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('Collect Images', frame)

        img_path = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)

        counter += 1
        cv2.waitKey(30)

    print(f"✅ Class {class_id} selesai ({DATASET_SIZE} images)")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Semua data berhasil dikumpulkan")
