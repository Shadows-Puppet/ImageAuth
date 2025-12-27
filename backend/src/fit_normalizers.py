# fit_normalizers.py

from features import FeatureExtractor
from data import HuggingFaceDataset
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

print("="*60)
print("FITTING NORMALIZERS")
print("="*60)

# Load feature extractor
extractor = FeatureExtractor(device='cuda')

# Load training data
print("\nLoading training dataset...")
dataset_path = "../data/new" 
train_dataset = HuggingFaceDataset(dataset_path, split='train')

# Sample a subset for fitting (default: 1000 images or entire dataset if smaller)
max_samples = min(1000, len(train_dataset))
print(f"\nSampling {max_samples} images from training set...")

# Random sample
indices = np.random.choice(len(train_dataset), max_samples, replace=False)

# Collect sample images
print("Collecting sample images...")
sample_images = []
for idx in tqdm(indices, desc="Loading samples"):
    img, _ = train_dataset[idx]
    # Resize to standard size
    img = img.resize((256, 256), Image.BILINEAR)
    sample_images.append(img)

# Compute normalization statistics
print(f"\nComputing normalization statistics from {len(sample_images)} images...")
freq_features_list = []
comp_features_list = []

for img in tqdm(sample_images, desc="Computing statistics"):
    freq_feat = extractor.extract_frequency_features(img)
    comp_feat = extractor.extract_compression_features(img)
    freq_features_list.append(freq_feat)
    comp_features_list.append(comp_feat)

# Convert to arrays
freq_features = np.array(freq_features_list)
comp_features = np.array(comp_features_list)

# Compute mean and std
extractor.freq_mean = freq_features.mean(axis=0)
extractor.freq_std = freq_features.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero

extractor.comp_mean = comp_features.mean(axis=0)
extractor.comp_std = comp_features.std(axis=0) + 1e-8

extractor.is_fitted = True

print("\n✓ Normalization statistics computed:")
print(f"  Frequency - Mean: {extractor.freq_mean}, Std: {extractor.freq_std}")
print(f"  Compression - Mean: {extractor.comp_mean}, Std: {extractor.comp_std}")

# Save normalizers
os.makedirs('checkpoints', exist_ok=True)
extractor.save_normalizers('checkpoints/normalizers.npz')

print("\n✓✓✓ Done! Normalizers saved to checkpoints/normalizers.npz")