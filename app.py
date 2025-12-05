import streamlit as st
from PIL import Image
from inference import SRCNN, device, transforms
import torch

st.title("Fluorescence Microscopy Super-Resolution")

uploaded_file = st.file_uploader("Upload a microscopy image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Original Image", use_column_width=True)

    # Inference
    model = SRCNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)

    enhanced_img = transforms.ToPILImage()(output.squeeze().cpu())
    st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)
