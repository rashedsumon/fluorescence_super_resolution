# Fluorescence Microscopy Super-Resolution

## Overview
Enhance the resolution and quality of fluorescence microscopy images using a deep learning super-resolution model. The project enables more precise visualization and analysis of biological samples and deploys the model on Apple devices via Core ML.

## Features
- Super-resolution enhancement of microscopy images
- Side-by-side comparisons for before/after visualization
- Export model to Core ML for macOS and iPad deployment
- Streamlit app for interactive visualization

## Folder Structure
- `app.py`: Main Streamlit app
- `data_loader.py`: Dataset download & preprocessing
- `model.py`: Model definition
- `train.py`: Training script
- `inference.py`: Image enhancement
- `utils.py`: Utilities (metrics, visualization)
- `data/`: Raw dataset
- `results/`: Enhanced images & comparisons
- `coreml_model/`: Exported Core ML model

## Dataset
Fluorescence microscopy images downloaded from Kaggle using KaggleHub:
