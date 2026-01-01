import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
import numpy as np

class MentalHealthDrawingDataset(Dataset):
    """Dataset for mental health drawing analysis."""
    
    def __init__(self, data_dir, config, transform=None, mode='train'):
        self.data_dir = data_dir
        self.config = config
        self.mode = mode
        self.transform = transform or self.get_default_transform()
        
        # Load data annotations (assuming CSV with image paths and labels)
        self.annotations = self.load_annotations()
        
        # Get class names from config
        self.class_names = list(config['classes'].values())
        
    def load_annotations(self):
        """Load image annotations - modify based on your data structure."""
        annotations = []
        
        # Example structure: data_dir/class_name/*.png
        for class_id, class_name in self.config['classes'].items():
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        annotations.append({
                            'image_path': os.path.join(class_dir, img_file),
                            'label': int(class_id),
                            'class_name': class_name
                        })
        
        # If no structured data, create dummy data for testing
        if len(annotations) == 0:
            print("⚠️  No data found, creating sample data for testing...")
            annotations = self.create_sample_data()
        
        return annotations
    
    def create_sample_data(self):
        """Create sample data for testing (replace with real data)."""
        from .helpers import create_sample_drawing
        
        sample_data = []
        classes = self.config['classes']
        
        # Create sample drawings for each class
        for class_id, class_name in classes.items():
            # Create multiple variations
            for i in range(20):  # 20 samples per class
                # Different drawing types for different mental states
                drawing_types = ["house", "tree", "person"]
                drawing_type = drawing_types[i % len(drawing_types)]
                
                # Create drawing with slight variations
                img = create_sample_drawing(drawing_type)
                
                # Save temporary image
                temp_dir = "temp_samples"
                os.makedirs(temp_dir, exist_ok=True)
                img_path = os.path.join(temp_dir, f"{class_name}_{i}.png")
                img.save(img_path)
                
                sample_data.append({
                    'image_path': img_path,
                    'label': int(class_id),
                    'class_name': class_name
                })
        
        return sample_data
    
    def get_default_transform(self):
        """Get default transforms based on mode."""
        if self.mode == 'train' and self.config['dataset']['augmentation']:
            return transforms.Compose([
                transforms.Resize(self.config['dataset']['image_size']),
                transforms.RandomRotation(self.config['augmentation']['rotation_range']),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(
                        self.config['augmentation']['width_shift_range'],
                        self.config['augmentation']['height_shift_range']
                    )
                ),
                transforms.ColorJitter(
                    brightness=self.config['augmentation']['brightness_range']
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.config['dataset']['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        image = Image.open(annotation['image_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': annotation['label'],
            'class_name': annotation['class_name'],
            'image_path': annotation['image_path']
        }

def create_data_loaders(config):
    """Create data loaders for training, validation, and testing."""
    from torch.utils.data import random_split
    
    # Create full dataset
    full_dataset = MentalHealthDrawingDataset(
        config['paths']['raw_data_dir'], 
        config, 
        mode='train'
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config['dataset']['train_ratio'] * total_size)
    val_size = int(config['dataset']['val_ratio'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Set modes for validation and test
    val_dataset.dataset.mode = 'val'
    test_dataset.dataset.mode = 'test'
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader, full_dataset.class_names