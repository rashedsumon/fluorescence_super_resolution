import os
import kagglehub
from pathlib import Path
from zipfile import ZipFile

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    print("Downloading dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download("shiveshcgatech/fluorescence-microscopy-image-denoising-dataset")
    
    # Extract if zip
    if dataset_path.endswith(".zip"):
        with ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
    print(f"Dataset is ready at {DATA_DIR}")

if __name__ == "__main__":
    download_dataset()
