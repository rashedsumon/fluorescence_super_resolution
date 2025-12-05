import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SRCNN
from pathlib import Path
import matplotlib.pyplot as plt
import kagglehub
import os

# ----------------------------
# Setup device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# Ensure model file exists
# ----------------------------
model_path = Path("model.pth")
if not model_path.exists():
    print("Model not found. Downloading from KaggleHub...")
    # Replace with your actual KaggleHub model dataset path
    kagglehub.dataset_download("shiveshcgatech/fluorescence-super-resolution-model", target_dir=".")
    print("Model downloaded.")

# ----------------------------
# Load SRCNN model
# ----------------------------
model = SRCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# Ensure image file exists
# ----------------------------
img_path = Path("data/raw/images/sample.png")
if not img_path.exists():
    raise FileNotFoundError(f"Image file not found at {img_path}. Please check your path.")

# ----------------------------
# Load and preprocess image
# ----------------------------
img = Image.open(img_path).convert("L")  # grayscale
transform = transforms.ToTensor()
input_tensor = transform(img).unsqueeze(0).to(device)

# ----------------------------
# Super-resolution
# ----------------------------
with torch.no_grad():
    output = model(input_tensor)

# Convert output tensor to image
output_img = transforms.ToPILImage()(output.squeeze().cpu())

# ----------------------------
# Display images
# ----------------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Enhanced")
plt.imshow(output_img, cmap="gray")
plt.axis("off")
plt.show()
