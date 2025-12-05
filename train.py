import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from model import SRCNN
from data_loader import download_dataset
from pathlib import Path

# Download dataset
download_dataset()

# Paths
DATA_DIR = Path("data/raw/images")  # adjust if folder differs
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset class
class MicroscopyDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = list(Path(folder).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, img  # input and target are same for simplicity

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = MicroscopyDataset(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model
model = SRCNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):  # small epoch for demo
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "model.pth")

# Convert to Core ML
import coremltools as ct
model.eval()
example_input = torch.rand(1,1,256,256)  # example input
traced_model = torch.jit.trace(model, example_input)
mlmodel = ct.convert(traced_model, inputs=[ct.ImageType(name="input_image", shape=example_input.shape, scale=1/255.0)])
mlmodel.save("coreml_model/SRCNN.mlmodel")
print("Core ML model saved to coreml_model/SRCNN.mlmodel")
