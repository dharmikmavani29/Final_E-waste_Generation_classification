import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit Page Setup
st.set_page_config(page_title="E-Waste Image Classifier", layout="centered")

st.title("♻️ E-Waste Image Classification App")
st.write("Upload single or multiple images to classify them using a trained EfficientNet model.")

# Define your class labels
class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
               'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

# Load your trained Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.keras')

model = load_model()

# File uploader supporting multiple images
uploaded_files = st.file_uploader("Upload E-Waste Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Correct Preprocessing (matching training)
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img).astype(np.float32)  
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"🏷️ Predicted Class: **{predicted_class}**")
        st.info(f"🔥 Confidence: {confidence:.2f}%")

        # Show Raw Probabilities
        st.write("### 🔍 Prediction Probabilities for all classes:")
        for cls, prob in zip(class_names, prediction[0]):
            st.write(f"- {cls}: `{prob * 100:.2f}%`")

        # Bar Chart for Confidence Distribution
        fig, ax = plt.subplots()
        ax.barh(class_names, prediction[0] * 100, color='cyan')
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Class-wise Confidence Distribution")
        st.pyplot(fig)
