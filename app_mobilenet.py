import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import os

# Load model and class mapping
MODEL_PATH = "models/best_model_mobilenet.h5"
CLASS_INDEX_PATH = "models/class_indices_mobilenet.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)

# Invert mapping and rename for user-friendly display
class_map = {v: k for k, v in class_indices.items()}
friendly_labels = {'NDVI': 'Healthy', 'GNDVI': 'Deficient'}

# Streamlit App Title
st.set_page_config(page_title="Nutrient Deficiency Detection", layout="centered")
st.title("🌿 Nutrient Deficiency Detection (MobileNetV2)")

# File uploader
uploaded_file = st.file_uploader("Upload a NDVI or GNDVI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display original image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Image preprocessing
    img_resized = cv2.resize(image, (224, 224))
    img_array = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_batch)
    predicted_class = int(np.round(prediction[0][0]))
    class_name = class_map[predicted_class]
    user_friendly = friendly_labels[class_name]

    # Display prediction in bold, larger font
    st.markdown(
        f"<h2 style='text-align: center; color: green;'>🌾 Prediction: {user_friendly}</h2>",
        unsafe_allow_html=True
    )