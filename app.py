import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time



# ---------------------------
# Inisialisasi session state
# ---------------------------
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "gesture_history" not in st.session_state:
    st.session_state.gesture_history = []

# ---------------------------
# Load model CNN
# ---------------------------
MODEL_PATH = r"C:\Users\marcell asmoro\Documents\gaby minjem\CODE + DATASET + ENV\cnn_sign_language_all_classes.keras"  # ganti sesuai modelmu
DATASET_PATH = r"C:\Users\marcell asmoro\Documents\gaby minjem\CODE + DATASET + ENV\data"  

if "model_loaded" not in st.session_state:
    try:
        st.session_state.model = load_model(MODEL_PATH)
        st.session_state.model_loaded = True
        st.session_state.labels = sorted(os.listdir(DATASET_PATH))
    except:
        st.warning("Model CNN tidak ditemukan!")
        st.session_state.model_loaded = False

# ---------------------------
# Tombol Start / Stop
# ---------------------------
def start_webcam():
    st.session_state.run_webcam = True

def stop_webcam():
    st.session_state.run_webcam = False

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Kontrol App")
st.sidebar.button("Start Webcam", on_click=start_webcam)
st.sidebar.button("Stop Webcam", on_click=stop_webcam)
fps = st.sidebar.slider("FPS", 5, 30, 15)

# ---------------------------
# Layout utama
# ---------------------------
st.markdown("<h1 style='text-align:center;color:green;'>CNN Gesture Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Realtime gesture recognition dengan webcam</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Webcam Feed")
    FRAME_WINDOW = st.image([])  # placeholder frame
with col2:
    st.subheader("Gesture History")
    if st.session_state.gesture_history:
        st.table(st.session_state.gesture_history[-10:])  # 10 terakhir
    else:
        st.write("Belum ada gesture terdeteksi")

# ---------------------------
# Webcam loop
# ---------------------------
if st.session_state.run_webcam:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Tidak bisa membuka webcam")
    else:
        run = True
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membaca frame")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------------------
            # Preprocessing untuk CNN
            # ---------------------------
            if st.session_state.model_loaded:
                img_input = cv2.resize(frame_rgb, (64,64))  # ukuran sesuai training
                img_input = img_input / 255.0
                img_input = np.expand_dims(img_input, axis=0)  # batch dim

                # Prediksi
                pred = st.session_state.model.predict(img_input, verbose=0)
                class_index = int(np.argmax(pred))
                confidence = float(np.max(pred))
                class_label = st.session_state.labels[class_index]

                # Overlay hasil
                cv2.putText(frame_rgb, f"{class_label} ({confidence*100:.1f}%)",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # Update history
                st.session_state.gesture_history.append(f"{class_label} ({confidence*100:.1f}%)")

            # Update frame Streamlit
            FRAME_WINDOW.image(frame_rgb)

            # Batasi FPS
            key = cv2.waitKey(int(1000/fps))
            if not st.session_state.run_webcam:
                run = False

        cap.release()
else:
    st.info("Tekan tombol Start Webcam di sidebar untuk memulai")
