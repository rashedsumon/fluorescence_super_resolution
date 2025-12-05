# inference.py
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SRCNN
from pathlib import Path

# ----------------------------
# Device setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Define paths
# ----------------------------
weights_dir = Path("data/weights")
weights_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing
model_path = weights_dir / "model.pth"

# ----------------------------
# Load SRCNN model
# ----------------------------
if not model_path.exists():
    raise FileNotFoundError(
        f"{model_path} not found. Please download model.pth from Kaggle and place it in {weights_dir}"
    )

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
    img = Image.open(img_path).convert("L")  # convert to grayscale
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)
    return img, input_tensor

# ----------------------------
# Super-resolution function
# ----------------------------
def super_resolve(input_tensor, save_path: str = None):
    """
    input_tensor: torch.Tensor of shape [1, 1, H, W]
    save_path: optional path to save enhanced image
    """
    with torch.no_grad():
        output = model(input_tensor)
    
    output_img = transforms.ToPILImage()(output.squeeze().cpu())
    
    # Save enhanced image if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(save_path)
    
    return output_img
