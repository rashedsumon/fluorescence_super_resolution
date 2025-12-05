# app.py
import streamlit as st
from PIL import Image
from inference import preprocess_image, super_resolve, device
from pathlib import Path

st.title("Fluorescence Microscopy Super-Resolution")

# Upload image
uploaded_file = st.file_uploader("Upload a microscopy image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Save uploaded file temporarily
    temp_path = Path("data/raw/temp_upload.png")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preprocess
    orig_img, input_tensor = preprocess_image(temp_path)
    st.subheader("Original Image")
    st.image(orig_img, use_column_width=True)
    
    # Super-resolution
    results_dir = Path("results/comparisons")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"enhanced_{uploaded_file.name}"
    enhanced_img = super_resolve(input_tensor, save_path=save_path)
    
    st.subheader("Enhanced Image")
    st.image(enhanced_img, use_column_width=True)
    
    st.success(f"Enhanced image saved at: {save_path}")
