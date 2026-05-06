
import streamlit as st
import cv2
import joblib
import numpy as np

# -----------------------------
# Settings
# -----------------------------
MODEL_NAME = "gear_model.pkl"
IMG_SIZE = 100
CATEGORIES = ["Bad Part", "Good Part"]

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load(MODEL_NAME)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("AI Part Inspection")
st.write("Upload an image to predict if the part is GOOD or BAD")

uploaded_file = st.file_uploader(
    "Upload Part Image",
    type=["jpg", "png", "jpeg"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    flattened = resized.flatten().reshape(1, -1)

    prediction = model.predict(flattened)
    confidence = model.predict_proba(flattened)

    result = CATEGORIES[prediction[0]]
    confidence_score = max(confidence[0]) * 100

    st.image(uploaded_file, caption="Uploaded Image")

    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence_score:.2f}%")
