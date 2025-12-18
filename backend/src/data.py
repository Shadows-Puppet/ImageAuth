import os
from torch.utils.data import Dataset
from PIL import Image
import random

class UniversalFakeDetectDataset(Dataset):
    """
    Dataset for UniversalFakeDetect structure:
    data/training/[class]/0_real/*.png
    data/training/[class]/1_fake/*.png
    """
    
    def __init__(self, root_dir, max_samples_per_class=None, seed=42):
        """
        Args:
            root_dir: Path to data/training or data/testing
            max_samples_per_class: Max images to load per class (None = all)
            seed: Random seed for reproducibility
        """
        self.samples = []
        self.labels = []
        
        random.seed(seed)
        
        # Get all class directories
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
        
        classes = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d))]
        
        print(f"Found {len(classes)} classes: {classes[:5]}...")
        
        total_real = 0
        total_fake = 0
        
        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            
            # Real images (0_real)
            real_dir = os.path.join(class_dir, '0_real')
            if os.path.exists(real_dir):
                real_files = [f for f in os.listdir(real_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Limit samples if requested
                if max_samples_per_class:
                    real_files = random.sample(real_files, 
                                              min(len(real_files), max_samples_per_class))
                
                for fname in real_files:
                    self.samples.append(os.path.join(real_dir, fname))
                    self.labels.append(0)  # 0 = real
                
                total_real += len(real_files)
            
            # Fake images (1_fake)
            fake_dir = os.path.join(class_dir, '1_fake')
            if os.path.exists(fake_dir):
                fake_files = [f for f in os.listdir(fake_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Limit samples if requested
                if max_samples_per_class:
                    fake_files = random.sample(fake_files,
                                              min(len(fake_files), max_samples_per_class))
                
                for fname in fake_files:
                    self.samples.append(os.path.join(fake_dir, fname))
                    self.labels.append(1)  # 1 = fake
                
                total_fake += len(fake_files)
        
        print(f"Loaded {total_real} real and {total_fake} fake images")
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        return img_path, label


class TestingDataset(Dataset):
    """
    Dataset for testing structure:
    data/testing/[model]/0_real/*.png
    data/testing/[model]/1_fake/*.png
    """
    
    def __init__(self, root_dir, model_name, max_samples=None):
        """
        Args:
            root_dir: Path to data/testing
            model_name: Specific model to test (e.g., 'biggan', 'stylegan')
            max_samples: Max images to load (None = all)
        """
        self.samples = []
        self.labels = []
        
        model_dir = os.path.join(root_dir, model_name)
        
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")
        
        print(f"Loading test data for model: {model_name}")
        
        # Real images
        real_dir = os.path.join(model_dir, '0_real')
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if max_samples:
                real_files = real_files[:max_samples]
            
            for fname in real_files:
                self.samples.append(os.path.join(real_dir, fname))
                self.labels.append(0)
        
        # Fake images
        fake_dir = os.path.join(model_dir, '1_fake')
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if max_samples:
                fake_files = fake_files[:max_samples]
            
            for fname in fake_files:
                self.samples.append(os.path.join(fake_dir, fname))
                self.labels.append(1)
        
        print(f"Loaded {len([l for l in self.labels if l == 0])} real "
              f"and {len([l for l in self.labels if l == 1])} fake images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        return img_path, label


def get_test_models(test_dir='../data/testing'):
    """Get list of available test models"""
    if not os.path.exists(test_dir):
        return []
    
    models = [d for d in os.listdir(test_dir)
             if os.path.isdir(os.path.join(test_dir, d))]
    return models