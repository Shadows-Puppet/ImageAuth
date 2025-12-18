import torch
import torch.nn as nn
import open_clip
from PIL import Image
import numpy as np
from scipy import fftpack
from io import BytesIO
import os

class FeatureExtractor:
    """
    Extract three types of features for AI image detection:
    1. CLIP features (768-dim) - semantic/visual features
    2. Frequency features (4-dim) - spectral domain analysis
    3. Compression features (3-dim) - statistical properties
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Initializing FeatureExtractor on {device}...")
        
        # Load CLIP model (frozen)
        print("Loading CLIP model...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='openai'
        )
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        
        # Freeze CLIP parameters (we never train it)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print(f"✓ CLIP model loaded on {device}")
        
        # Normalization statistics (will be computed from training data)
        self.freq_mean = None
        self.freq_std = None
        self.comp_mean = None
        self.comp_std = None
        self.is_fitted = False
    
    @torch.no_grad()
    def extract_clip_features(self, image):
        """
        Extract CLIP image features (768-dim)
        
        Args:
            image: PIL Image or torch.Tensor
            
        Returns:
            torch.Tensor: (768,) normalized CLIP features
        """
        # Preprocess image
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        
        # Extract features
        features = self.clip_model.encode_image(image)
        
        # Normalize (CLIP was trained with normalized features)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.squeeze().cpu()
    
    def extract_frequency_features(self, image):
        """
        Extract frequency domain features (4-dim)
        Analyzes the FFT spectrum to detect GAN/diffusion artifacts
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            np.ndarray: (4,) frequency features
                [low_freq_energy, mid_freq_energy, high_freq_energy, phase_coherence]
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to grayscale for frequency analysis
        if len(img_array.shape) == 3:
            # RGB to grayscale: standard formula
            gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        else:
            gray = img_array
        
        # Compute 2D FFT
        fft = fftpack.fft2(gray)
        fft_shift = fftpack.fftshift(fft)  # Shift zero frequency to center
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)
        
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Create distance map from center (DC component)
        y, x = np.ogrid[:h, :w]
        distance_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_distance = np.sqrt(center_h**2 + center_w**2)
        
        # Normalize distances to 0-1
        normalized_distance = distance_from_center / max_distance
        
        # Define frequency bands
        # Low: 0-15% of max distance (DC and very low frequencies)
        # Mid: 15-50% (where GAN artifacts often appear)
        # High: 50-100% (fine details)
        low_mask = normalized_distance < 0.15
        mid_mask = (normalized_distance >= 0.15) & (normalized_distance < 0.5)
        high_mask = normalized_distance >= 0.5
        
        # Compute energy in each band
        low_freq_energy = np.sum(magnitude * low_mask) / np.sum(low_mask)
        mid_freq_energy = np.sum(magnitude * mid_mask) / np.sum(mid_mask)
        high_freq_energy = np.sum(magnitude * high_mask) / np.sum(high_mask)
        
        # Phase coherence: measure of phase uniformity
        # Higher values = less coherent = more natural
        # AI images often have too-coherent phase
        phase_grad_y = np.gradient(phase, axis=0)
        phase_grad_x = np.gradient(phase, axis=1)
        phase_coherence = np.sqrt(phase_grad_y**2 + phase_grad_x**2).mean()
        
        # Combine features
        features = np.array([
            low_freq_energy,
            mid_freq_energy,
            high_freq_energy,
            phase_coherence
        ], dtype=np.float32)
        
        # Apply log transform to reduce scale
        # This brings values from ~10000 range down to ~10 range
        features = np.log1p(features)  # log(1 + x) to handle zeros
        
        return features
    
    def extract_compression_features(self, image):
        """
        Extract compression-based features (3-dim)
        AI images often compress differently than natural images
        
        Args:
            image: PIL Image
            
        Returns:
            np.ndarray: (3,) compression features
                [png_ratio, entropy, jpeg_ratio]
        """
        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image as numpy array for calculations
        img_array = np.array(image)
        original_size = img_array.nbytes
        
        # Feature 1: PNG compression ratio (lossless)
        png_buffer = BytesIO()
        image.save(png_buffer, format='PNG', optimize=True)
        png_size = png_buffer.tell()
        png_ratio = png_size / original_size
        png_buffer.close()
        
        # Feature 2: Shannon entropy (information content)
        # Measures randomness/complexity of the image
        def compute_entropy(img_channel):
            """Compute Shannon entropy of a single channel"""
            # Compute histogram
            hist, _ = np.histogram(img_channel.flatten(), bins=256, range=(0, 256))
            
            # Normalize to get probabilities
            hist = hist.astype(np.float32) / hist.sum()
            
            # Remove zeros to avoid log(0)
            hist = hist[hist > 0]
            
            # Entropy = -sum(p * log2(p))
            entropy = -np.sum(hist * np.log2(hist))
            
            return entropy
        
        # Compute entropy for each channel and average
        if len(img_array.shape) == 3:
            entropies = [compute_entropy(img_array[:,:,i]) for i in range(3)]
            entropy = np.mean(entropies)
        else:
            entropy = compute_entropy(img_array)
        
        # Normalize entropy (max is 8 bits for 256 levels)
        entropy = entropy / 8.0
        
        # Feature 3: JPEG compression ratio
        # Different JPEG artifacts between real and AI images
        jpeg_buffer = BytesIO()
        image.save(jpeg_buffer, format='JPEG', quality=95)
        jpeg_size = jpeg_buffer.tell()
        jpeg_ratio = jpeg_size / original_size
        jpeg_buffer.close()
        
        features = np.array([
            png_ratio,
            entropy,
            jpeg_ratio
        ], dtype=np.float32)
        
        return features
    
    def fit_normalizers(self, image_paths, max_samples=1000):
        """
        Compute normalization statistics from training data
        Call this once before training on a sample of your training images
        
        Args:
            image_paths: List of image file paths
            max_samples: Maximum number of samples to use for statistics
        """
        print(f"\nFitting normalizers on {min(len(image_paths), max_samples)} images...")
        
        freq_features_list = []
        comp_features_list = []
        
        # Sample random images if we have too many
        if len(image_paths) > max_samples:
            import random
            image_paths = random.sample(image_paths, max_samples)
        
        from tqdm import tqdm
        for img_path in tqdm(image_paths, desc="Computing statistics"):
            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize((256, 256), Image.BILINEAR)
                
                freq_feat = self.extract_frequency_features(image)
                comp_feat = self.extract_compression_features(image)
                
                freq_features_list.append(freq_feat)
                comp_features_list.append(comp_feat)
                
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
                continue
        
        # Convert to arrays
        freq_features = np.array(freq_features_list)
        comp_features = np.array(comp_features_list)
        
        # Compute mean and std
        self.freq_mean = freq_features.mean(axis=0)
        self.freq_std = freq_features.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
        
        self.comp_mean = comp_features.mean(axis=0)
        self.comp_std = comp_features.std(axis=0) + 1e-8
        
        self.is_fitted = True
        
        print("✓ Normalization statistics computed:")
        print(f"  Frequency - Mean: {self.freq_mean}, Std: {self.freq_std}")
        print(f"  Compression - Mean: {self.comp_mean}, Std: {self.comp_std}")
    
    def extract_all_features(self, image_path):
        """
        Extract all features (CLIP + Frequency + Compression) for an image
        
        Args:
            image_path: Path to image file OR PIL Image
            
        Returns:
            torch.Tensor: (775,) combined feature vector
        """
        # Load image if path provided
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise ValueError("Input must be image path or PIL Image")
        
        # Resize to standard size (256x256)
        image = image.resize((256, 256), Image.BILINEAR)
        
        # Extract each feature type
        clip_feat = self.extract_clip_features(image)  # (768,)
        freq_feat = torch.from_numpy(
            self.extract_frequency_features(image)
        ).float()  # (4,)
        comp_feat = torch.from_numpy(
            self.extract_compression_features(image)
        ).float()  # (3,)
        
        # Normalize frequency and compression features if fitted
        if self.is_fitted:
            freq_feat = (freq_feat - torch.from_numpy(self.freq_mean).float()) / torch.from_numpy(self.freq_std).float()
            comp_feat = (comp_feat - torch.from_numpy(self.comp_mean).float()) / torch.from_numpy(self.comp_std).float()
        
        # Concatenate all features
        all_features = torch.cat([
            clip_feat,      # 768-dim
            freq_feat,      # 4-dim
            comp_feat       # 3-dim
        ])  # Total: 775-dim
        
        return all_features
    
    def save_normalizers(self, path):
        """Save normalization statistics to file"""
        if not self.is_fitted:
            raise ValueError("Normalizers not fitted yet. Call fit_normalizers() first.")
        
        np.savez(path,
                 freq_mean=self.freq_mean,
                 freq_std=self.freq_std,
                 comp_mean=self.comp_mean,
                 comp_std=self.comp_std)
        print(f"✓ Saved normalizers to {path}")
    
    def load_normalizers(self, path):
        """Load normalization statistics from file"""
        data = np.load(path)
        self.freq_mean = data['freq_mean']
        self.freq_std = data['freq_std']
        self.comp_mean = data['comp_mean']
        self.comp_std = data['comp_std']
        self.is_fitted = True
        print(f"✓ Loaded normalizers from {path}")


# Convenience function for quick testing
def test_feature_extractor():
    """Test the feature extractor on a sample image"""
    print("="*60)
    print("TESTING FEATURE EXTRACTOR")
    print("="*60)
    
    extractor = FeatureExtractor()
    
    # Find a test image
    test_dirs = [
        '../data/training',
        '../data/testing'
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            # Find first class/model
            subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            if subdirs:
                first_subdir = subdirs[0]
                real_dir = os.path.join(test_dir, first_subdir, '0_real')
                if os.path.exists(real_dir):
                    images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if images:
                        test_image = os.path.join(real_dir, images[0])
                        break
    
    if not test_image:
        print("✗ Could not find test image")
        return
    
    print(f"\nTesting on: {test_image}")
    
    # Extract features
    print("\n1. Extracting CLIP features...")
    img = Image.open(test_image).convert('RGB')
    clip_feat = extractor.extract_clip_features(img)
    print(f"   Shape: {clip_feat.shape}")
    print(f"   Range: [{clip_feat.min():.3f}, {clip_feat.max():.3f}]")
    print(f"   Norm: {clip_feat.norm():.3f}")
    
    print("\n2. Extracting frequency features...")
    freq_feat = extractor.extract_frequency_features(img)
    print(f"   Shape: {freq_feat.shape}")
    print(f"   Values: {freq_feat}")
    
    print("\n3. Extracting compression features...")
    comp_feat = extractor.extract_compression_features(img)
    print(f"   Shape: {comp_feat.shape}")
    print(f"   Values: {comp_feat}")
    
    print("\n4. Extracting all features...")
    all_feat = extractor.extract_all_features(test_image)
    print(f"   Combined shape: {all_feat.shape}")
    print(f"   Expected: torch.Size([775])")
    
    if all_feat.shape[0] == 775:
        print("\n✓✓✓ Feature extraction working perfectly! ✓✓✓")
    else:
        print(f"\n✗ Wrong shape! Expected 775, got {all_feat.shape[0]}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    test_feature_extractor()