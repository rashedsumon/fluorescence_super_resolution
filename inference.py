# inference.py
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SRCNN
from pathlib import Path
import kagglehub
import shutil

# ----------------------------
# Device setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Ensure model file exists
# ----------------------------
model_path = Path("model.pth")
if not model_path.exists():
    print("Model not found. Downloading from KaggleHub...")
    dataset_path = kagglehub.dataset_download("shiveshcgatech/fluorescence-super-resolution-model")
    downloaded_model = Path(dataset_path) / "model.pth"
    
    if downloaded_model.exists():
        shutil.copy(downloaded_model, model_path)
        print(f"Model copied to {model_path}")
    else:
        raise FileNotFoundError(f"model.pth not found in downloaded dataset at {dataset_path}")

# ----------------------------
# Load SRCNN model
# ----------------------------
model = SRCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ----------------------------
# Image preprocessing function
# ----------------------------
def preprocess_image(img_path: str):
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")
    img = Image.open(img_path).convert("L")  # grayscale
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)
    return img, input_tensor

# ----------------------------
# Super-resolution function
# ----------------------------
def super_resolve(input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    return transforms.ToPILImage()(output.squeeze().cpu())
