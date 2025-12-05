# app.py
import streamlit as st
from PIL import Image
from inference import preprocess_image, super_resolve

st.title("Fluorescence Microscopy Super-Resolution (Demo)")

uploaded_file = st.file_uploader("Upload a microscopy image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Load image
    temp_path = "temp_upload.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    orig_img = preprocess_image(temp_path)
    st.subheader("Original Image")
    st.image(orig_img, use_column_width=True)

    # Super-resolution demo
    scale_factor = st.slider("Upscale Factor", 1.5, 4.0, 2.0, 0.1)
    enhanced_img = super_resolve(orig_img, scale_factor=scale_factor)

    st.subheader("Enhanced Image (Demo)")
    st.image(enhanced_img, use_column_width=True)

    # Optional save
    save_path = f"enhanced_{uploaded_file.name}"
    enhanced_img.save(save_path)
    st.success(f"Enhanced image saved as: {save_path}")
