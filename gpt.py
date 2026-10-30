
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐶",
    layout="centered"
)

# Title
st.title("🐶 Dog vs Cat Image Classifier")
st.write("Upload an image to identify whether it is a Dog or Cat.")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("CAT.keras")

model = load_model()

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Image upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Resize image
    img = img.resize((64, 64))

    # Convert image to array
    x = np.array(img, dtype=np.float32)

    # Expand dimensions
    x = np.expand_dims(x, axis=0)

    # Prediction
    if st.button("Predict"):

        prediction = model.predict(x, verbose=0)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.subheader("Prediction Result")

        st.success(f"Prediction: {predicted_class.upper()}")

        st.write(f"Confidence: {confidence:.2f}%")

        st.progress(float(confidence / 100))
