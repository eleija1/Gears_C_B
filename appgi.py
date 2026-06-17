import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

MODEL_PATH = "gear_model.pkl"
IMG_SIZE = 100
CATEGORIES = ["Bad Gear", "Good Gear"]

st.set_page_config(page_title="Gear Inspector", layout="centered")

st.title("Gear Quality Inspector")
st.write("Select multiple gear images to analyze a batch at once.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Could not load gear_model.pkl. Make sure the model file is in GitHub.")
    st.write(e)
    st.stop()

uploaded_files = st.file_uploader(
    "Choose gear images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"### Analyzing {len(uploaded_files)} images")
    st.divider()

    for file in uploaded_files:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(file)
            st.image(image, caption=file.name, width=150)

        with col2:
            file_bytes = np.asarray(bytearray(file.getvalue()), dtype=np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if img_array is None:
                st.error("Error processing this image.")
                continue

            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            flattened_array = resized_array.flatten().reshape(1, -1)

            prediction = model.predict(flattened_array)[0]
            result_class = CATEGORIES[prediction]

            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(flattened_array)
                confidence_score = max(confidence[0]) * 100
            else:
                confidence_score = None

            if result_class == "Good Gear":
                st.success(f"Result: {result_class} - PASS")
            else:
                st.error(f"Result: {result_class} - FAIL")

            if confidence_score is not None:
                st.info(f"AI Confidence: {confidence_score:.2f}%")

        st.divider()
else:
    st.info("Upload one or more gear images to begin.")
