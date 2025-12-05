# app.py
import streamlit as st
from PIL import Image

st.title("Fluorescence Microscopy Super-Resolution (Demo)")

uploaded_file = st.file_uploader("Upload a microscopy image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Load uploaded image
    img = Image.open(uploaded_file).convert("L")  # grayscale
    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    # Simulate super-resolution by upscaling
    scale_factor = st.slider("Upscale Factor", 1.5, 4.0, 2.0, 0.1)
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    enhanced_img = img.resize(new_size, resample=Image.BICUBIC)

    st.subheader("Enhanced Image (Demo)")
    st.image(enhanced_img, use_column_width=True)

    # Optional: Save enhanced image
    save_path = f"enhanced_{uploaded_file.name}"
    enhanced_img.save(save_path)
    st.success(f"Enhanced image saved as: {save_path}")
