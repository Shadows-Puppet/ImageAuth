from features import FeatureExtractor
from data import UniversalFakeDetectDataset
import os

print("="*60)
print("FITTING NORMALIZERS")
print("="*60)

# Load feature extractor
extractor = FeatureExtractor(device='cuda')

# Load a sample of training data
print("\nLoading sample of training data...")
train_dataset = UniversalFakeDetectDataset(
    root_dir='../data/training',
    max_samples_per_class=50  # 50 images per class
)

# Get image paths (mix of real and fake)
image_paths = [path for path, label in train_dataset]

print(f"Computing normalization statistics from {len(image_paths)} images...")

# Fit normalizers
extractor.fit_normalizers(image_paths, max_samples=1000)

# Save
os.makedirs('checkpoints', exist_ok=True)
extractor.save_normalizers('checkpoints/normalizers.npz')