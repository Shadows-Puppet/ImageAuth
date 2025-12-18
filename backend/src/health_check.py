import torch
import open_clip
from PIL import Image
import os

print("="*50)
print("TESTING YOUR SETUP")
print("="*50)

# 1. Check CUDA
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   WARNING: CUDA not available, will use CPU")

# 2. Test CLIP loading
print("\n2. Loading CLIP model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='openai'
)
model = model.to(device)
print(f"   ✓ CLIP loaded successfully on {device}")

# 3. Test on one image
print("\n3. Testing feature extraction...")

# Your data structure: data/training/airplane/0_real/
test_image_path = None
data_root = '../data/training'

if os.path.exists(data_root):
    # Find first class folder (airplane, etc.)
    classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    if classes:
        first_class = classes[0]
        real_dir = os.path.join(data_root, first_class, '0_real')
        
        if os.path.exists(real_dir):
            files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                test_image_path = os.path.join(real_dir, files[0])
                print(f"   Testing with class: {first_class}")

if test_image_path:
    img = Image.open(test_image_path)
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.encode_image(img_tensor)
    
    print(f"   ✓ Image loaded: {test_image_path}")
    print(f"   ✓ Features extracted: {features.shape}")
    print(f"   ✓ Feature norm: {features.norm():.2f}")
else:
    print("   ✗ Could not find test image. Check your data paths!")

print("\nCleaning up GPU memory...")
del model  # Delete model
torch.cuda.empty_cache()  # Clear GPU cache
print("Done!")