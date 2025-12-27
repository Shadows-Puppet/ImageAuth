# data.py

from datasets import load_from_disk, Dataset as HFDataset
from torch.utils.data import Dataset
from PIL import Image
import os

class HuggingFaceDataset(Dataset):
    """
    Wrapper for HuggingFace datasets to work with PyTorch DataLoader
    """
    def __init__(self, dataset_path, split='train'):
        """
        Args:
            dataset_path: Path to the saved HuggingFace dataset directory (parent folder)
            split: Which split to use ('train', 'eval', or 'test')
        """
        print(f"Loading {split} split from {dataset_path}...")
        
        # Check if this is a split subfolder or parent folder
        split_path = os.path.join(dataset_path, split)
        
        if os.path.exists(split_path):
            # Load from the specific split subfolder
            print(f"  Loading from subfolder: {split_path}")
            self.dataset = load_from_disk(split_path)
        else:
            # Try loading as DatasetDict and selecting split
            full_dataset = load_from_disk(dataset_path)
            
            # Check if it's a DatasetDict or a single Dataset
            if hasattr(full_dataset, 'keys'):
                # It's a DatasetDict
                if split not in full_dataset:
                    raise ValueError(f"Split '{split}' not found. Available splits: {list(full_dataset.keys())}")
                self.dataset = full_dataset[split]
            else:
                # It's a single Dataset - assume it's the requested split
                print(f"  Loaded as single dataset (no split structure)")
                self.dataset = full_dataset
        
        print(f"✓ Loaded {len(self.dataset)} samples from {split} split")
        
        # Determine the column names (adjust these based on your dataset)
        self.image_column = self._find_image_column()
        self.label_column = self._find_label_column()
        
        print(f"  Using image column: '{self.image_column}'")
        print(f"  Using label column: '{self.label_column}'")
    
    def _find_image_column(self):
        """Auto-detect the image column name"""
        possible_names = ['image', 'img', 'images', 'picture', 'photo']
        for name in possible_names:
            if name in self.dataset.column_names:
                return name
        # If none found, use first column
        return self.dataset.column_names[0]
    
    def _find_label_column(self):
        """Auto-detect the label column name"""
        possible_names = ['binary_label', 'label', 'class', 'target', 'y']
        for name in possible_names:
            if name in self.dataset.column_names:
                return name
        # If none found, raise error
        raise ValueError(f"Could not find label column. Available columns: {self.dataset.column_names}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, label) where image is a PIL Image and label is an integer
        """
        sample = self.dataset[idx]
        
        # Get the image (can be PIL Image, dict, or array)
        image = sample[self.image_column]
        
        # Handle different image formats
        if isinstance(image, Image.Image):
            # Already a PIL Image
            pass
        elif isinstance(image, dict):
            # Image stored as dict (common in HuggingFace datasets)
            # Usually has 'bytes' or 'path' key
            if 'bytes' in image:
                from io import BytesIO
                image = Image.open(BytesIO(image['bytes']))
            elif 'path' in image:
                image = Image.open(image['path'])
            elif 'array' in image:
                import numpy as np
                image = Image.fromarray(np.array(image['array']))
            else:
                raise ValueError(f"Unknown image dict format: {image.keys()}")
        elif hasattr(image, '__array_interface__') or hasattr(image, '__array__'):
            # NumPy array or array-like
            import numpy as np
            image = Image.fromarray(np.array(image))
        else:
            raise ValueError(f"Unknown image format: {type(image)}")
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get the label
        label = sample[self.label_column]
        
        return image, label


# For backwards compatibility, keep the old class name
UniversalFakeDetectDataset = HuggingFaceDataset