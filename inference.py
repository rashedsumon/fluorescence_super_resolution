import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SRCNN
from pathlib import Path
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SRCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Load image
img_path = Path("data/raw/images/sample.png")
img = Image.open(img_path).convert("L")
transform = transforms.ToTensor()
input_tensor = transform(img).unsqueeze(0).to(device)

# Super-resolution
with torch.no_grad():
    output = model(input_tensor)

# Convert tensor to image
output_img = transforms.ToPILImage()(output.squeeze().cpu())
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.subplot(1,2,2)
plt.title("Enhanced")
plt.imshow(output_img, cmap="gray")
plt.show()
