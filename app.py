import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "model/1"

model = tf.saved_model.load(MODEL_PATH)

serving_fn = model.signatures["serving_default"]

# Input name is detected automatically
input_name = list(
    serving_fn.structured_input_signature[1].keys()
)[0]


# -----------------------------
# Class names
# -----------------------------
class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="COVID X-Ray Classifier",
    page_icon="🩻"
)

st.title("🩻 COVID X-Ray Classifier")

st.write(
    "Upload a chest X-ray image and the trained ResNet50 "
    "model will classify it."
)

st.warning(
    "This is an AI model demonstration and is not a medical diagnosis."
)


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-Ray",
        width="stretch"
    )

    if st.button("🔍 Predict"):

        # Resize image
        image_resized = image.resize((224, 224))

        # Convert to NumPy
        image_array = np.array(
            image_resized,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Model prediction
        result = serving_fn(
            **{
                input_name:
                tf.convert_to_tensor(image_array)
            }
        )

        # Get probabilities
        probabilities = result["output_0"].numpy()[0]

        # Get predicted class
        predicted_index = np.argmax(probabilities)

        predicted_class = class_names[predicted_index]

        confidence = (
            probabilities[predicted_index] * 100
        )

        # Display result
        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # Show all probabilities
        st.subheader("Prediction Probabilities")

        for name, probability in zip(
            class_names,
            probabilities
        ):
            st.write(
                f"**{name}:** "
                f"{probability * 100:.2f}%"
            )