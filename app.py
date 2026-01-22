import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "cnn_sign_language_all_classes.keras"

try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except:
    st.error(f"Model {MODEL_PATH} tidak ditemukan!")
    model_loaded = False

# ---------------------------
# App title
# ---------------------------
st.title("Gesture Detection (Upload Gambar)")
st.write("Upload gambar tangan, dan model CNN akan memprediksi gesture.")

# ---------------------------
# Upload gambar
# ---------------------------
uploaded_file = st.file_uploader("Pilih gambar (.jpg / .png)", type=["jpg", "png"])

if uploaded_file is not None and model_loaded:
    # Convert ke array OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Tampilkan gambar
    st.image(img_rgb, caption="Gambar yang diupload", width=500)

    # Preprocess untuk CNN
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # Prediksi gesture
    pred = model.predict(img_input)
    gesture_class = int(np.argmax(pred))

    st.success(f"Prediksi gesture: **{gesture_class}**")

