import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Image Recognition",
    page_icon="🖼️",
    layout="wide"
)


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

@st.cache_resource
def load_model():

    model = MobileNetV2(
        weights="imagenet"
    )

    return model

image_model = load_model()

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🖼️ AI Image Recognition System")

st.write(
    """
Upload an image and the AI model will identify
the object using the pre-trained MobileNetV2 model.
"""
)

st.divider()

# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    original_width = image.width
    original_height = image.height
    image_format = uploaded_file.type
    image_mode = image.mode

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            caption="Original Image",
            width="stretch"
        )

    with col2:

        st.subheader("📋 Image Information")

        st.metric(
            "Width",
            f"{original_width} px"
        )

        st.metric(
            "Height",
            f"{original_height} px"
        )

        st.metric(
            "Format",
            image_format
        )

        st.metric(
            "Mode",
            image_mode
        )

    st.divider()

    # -----------------------------
    # Pre-processing
    # -----------------------------

    resized_image = image.resize((224,224))

    image_array = img_to_array(resized_image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    processed_image = preprocess_input(
        image_array
    )

    if st.button("🔍 Predict Object"):

            with st.spinner("🤖 AI is analysing the image..."):

             predictions = image_model.predict(
                processed_image,
                verbose=0
            )

            decoded_predictions = decode_predictions(
                predictions,
                top=5
            )[0]

            st.success("✅ Prediction Completed Successfully!")

            st.divider()

        # ---------------------------------------------------
        # Best Prediction
        # ---------------------------------------------------

            best_prediction = decoded_predictions[0]

            best_label = best_prediction[1].replace("_", " ").title()
            best_confidence = best_prediction[2] * 100

            st.subheader("🏆 Best Prediction")

            col1, col2 = st.columns(2)

            with col1:

             st.metric(
                "Predicted Object",
                best_label
            )

            with col2:
 
             st.metric(
                "Confidence",
                f"{best_confidence:.2f}%"
            )

            st.divider()

        # ---------------------------------------------------
        # Top 5 Predictions
        # ---------------------------------------------------

            st.subheader("📊 Top 5 Predictions")

            result_data = []

            for rank, (_, label, probability) in enumerate(decoded_predictions, start=1):

             confidence = probability * 100

            label = label.replace("_", " ").title()

            st.write(f"### {rank}. {label}")

            st.progress(float(probability))

            st.write(f"Confidence : **{confidence:.2f}%**")

            result_data.append(
                {
                    "Rank": rank,
                    "Object": label,
                    "Confidence (%)": round(confidence, 2)
                }
            )

            st.write("---")

        # ---------------------------------------------------
        # Result Table
        # ---------------------------------------------------

            st.subheader("📋 Prediction Summary")

            df = pd.DataFrame(result_data)

            st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )

            st.divider()



# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.markdown(
    """
<div style="text-align:center; color:gray;">
<b>AI Image Recognition System</b><br>
Built with ❤️ using Python, Streamlit, TensorFlow and MobileNetV2
</div>
""",
    unsafe_allow_html=True
)