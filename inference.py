# inference.py
from PIL import Image

def preprocess_image(img_path: str):
    """Load an image from path and convert to grayscale."""
    img = Image.open(img_path).convert("L")
    return img

def super_resolve(img, scale_factor: float = 2.0):
    """
    Simulate super-resolution by upscaling the image.
    img: PIL.Image
    scale_factor: how much to upscale the image
    """
    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
    enhanced_img = img.resize(new_size, resample=Image.BICUBIC)
    return enhanced_img
