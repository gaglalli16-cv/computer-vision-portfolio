"""
PyTorch Dataset, Augmentation & DataLoader Setup
=================================================

This script provides:
1. Custom PyTorch Dataset class
2. Comprehensive data augmentation pipeline
3. DataLoader configuration
4. Utilities for training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Image settings
IMG_SIZE = 192  # Standard for most pre-trained models (can use 384, 512 for better accuracy)
BATCH_SIZE = 32
NUM_WORKERS = 0  # Adjust based on your CPU

# Class names
CLASS_COLUMNS = [
    'antelope_duiker', 'bird', 'blank', 'civet_genet',
    'hog', 'leopard', 'monkey_prosimian', 'rodent'
]

print("="*80)
print("PYTORCH DATASET & AUGMENTATION SETUP")
print("="*80)

# ============================================================================
# 1. DATA AUGMENTATION PIPELINE
# ============================================================================

print("\n[1/4] Setting up augmentation pipelines...")

def get_train_transforms(img_size=192):
    """
    Aggressive augmentation for training
    Uses Albumentations for more options and better performance
    """
    return A.Compose([
        # Resize
        A.Resize(img_size, img_size),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),  # Less common but can help
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,
            p=0.7
        ),
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),
        
        # Color/lighting transforms (critical for different sites/conditions)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.7),
        
        # Weather/blur effects
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
        
        # Advanced augmentations
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size//8,
            max_width=img_size//8,
            fill_value=0,
            p=0.3
        ),
        
        # Normalize (ImageNet stats)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        
        # Convert to tensor
        ToTensorV2(),
    ])

def get_val_transforms(img_size=192):
    """
    Minimal transforms for validation (no augmentation)
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_tta_transforms(img_size=192):
    """
    Test-Time Augmentation (TTA) transforms
    Returns multiple versions of the same image
    """
    transforms_list = [
        # Original
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Brightness adjusted
        A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    return transforms_list

print(" Augmentation pipelines created")
print(f"   • Training: Aggressive augmentation")
print(f"   • Validation: Resize + Normalize only")
print(f"   • TTA: 3 variations per image")

# ============================================================================
# 2. CUSTOM PYTORCH DATASET
# ============================================================================

print("\n[2/4] Creating PyTorch Dataset class...")

class WildlifeDataset(Dataset):
    """
    Custom Dataset for Wildlife Classification
    """
    
    def __init__(self, df, data_dir, transform=None, is_test=False):
        """
        Args:
            df: DataFrame with image metadata and labels
            data_dir: Root directory containing images
            transform: Albumentations transform pipeline
            is_test: Whether this is test data (no labels)
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_test = is_test
        self.class_columns = CLASS_COLUMNS
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path
        row = self.df.iloc[idx]
        img_path = self.data_dir / row['filepath']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = np.zeros((192, 192, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Get labels
        if not self.is_test:
            labels = torch.tensor(
                row[self.class_columns].values.astype(np.float32)
            )
            return {
                'image': image,
                'labels': labels,
                'image_id': row['id'],
                'site': row['site']
            }
        else:
            return {
                'image': image,
                'image_id': row['id'],
                'site': row['site']
            }

print(" WildlifeDataset class created")

# ============================================================================
# 3. CREATE DATALOADERS
# ============================================================================

print("\n[3/4] Creating DataLoaders...")

def create_dataloaders(
    train_csv_path,
    val_csv_path,
    data_dir,
    batch_size=32,
    num_workers=0,
    img_size=192
):
    """
    Create train and validation dataloaders
    """
    
    # Load splits
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = WildlifeDataset(
        df=train_df,
        data_dir=data_dir,
        transform=get_train_transforms(img_size),
        is_test=False
    )
    
    val_dataset = WildlifeDataset(
        df=val_df,
        data_dir=data_dir,
        transform=get_val_transforms(img_size),
        is_test=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Create the dataloaders
train_loader, val_loader = create_dataloaders(
    train_csv_path=OUTPUT_DIR / 'train_split.csv',
    val_csv_path=OUTPUT_DIR / 'val_split.csv',
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    img_size=IMG_SIZE
)

print(f" DataLoaders created")
print(f"   • Train batches: {len(train_loader)}")
print(f"   • Val batches: {len(val_loader)}")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Image size: {IMG_SIZE}x{IMG_SIZE}")

# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================

print("\n[4/4] Creating utility functions...")

def get_class_weights(split_info_path):
    """Load class weights from split info"""
    with open(split_info_path, 'r') as f:
        split_info = json.load(f)
    
    class_weights = split_info['class_weights']
    # Convert to tensor in correct order
    weights = torch.tensor([class_weights[cls] for cls in CLASS_COLUMNS])
    return weights

def visualize_batch(loader, num_images=8):
    """Visualize a batch of augmented images"""
    import matplotlib.pyplot as plt
    
    batch = next(iter(loader))
    images = batch['image'][:num_images]
    labels = batch['labels'][:num_images]
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        img_np = img.permute(1, 2, 0).numpy()
        axes[idx].imshow(img_np)
        
        # Get class name
        class_idx = torch.argmax(label).item()
        class_name = CLASS_COLUMNS[class_idx]
        axes[idx].set_title(f"{class_name}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'visualizations' / 'augmented_batch.png', dpi=200)
    plt.show()
    print(" Batch visualization saved")

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load class weights
class_weights = get_class_weights(OUTPUT_DIR / 'split_info.json')
print(f" Class weights loaded: {class_weights.shape}")

# Visualize a training batch
print("\n Visualizing augmented training batch...")
visualize_batch(train_loader, num_images=8)

# ============================================================================
# 5. TEST DATALOADER
# ============================================================================

print("\n" + "="*80)
print("TESTING DATALOADERS")
print("="*80)

# Test training loader
print("\n Testing training DataLoader...")
train_batch = next(iter(train_loader))
print(f" Batch loaded successfully")
print(f"   • Image shape: {train_batch['image'].shape}")
print(f"   • Labels shape: {train_batch['labels'].shape}")
print(f"   • Image IDs: {len(train_batch['image_id'])}")

# Test validation loader
print("\n Testing validation DataLoader...")
val_batch = next(iter(val_loader))
print(f" Batch loaded successfully")
print(f"   • Image shape: {val_batch['image'].shape}")
print(f"   • Labels shape: {val_batch['labels'].shape}")

# Verify label sums
print("\n Verifying label integrity...")
label_sums = train_batch['labels'].sum(dim=1)
all_ones = torch.allclose(label_sums, torch.ones_like(label_sums))
print(f" All labels sum to 1.0: {all_ones}")
if not all_ones:
    print(f"  WARNING: Some labels don't sum to 1.0: {label_sums}")

# ============================================================================
# 6. SAVE CONFIGURATION
# ============================================================================

config = {
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'num_workers': NUM_WORKERS,
    'num_classes': len(CLASS_COLUMNS),
    'class_names': CLASS_COLUMNS,
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'augmentation': {
        'train': 'aggressive',
        'val': 'minimal',
        'tta': 'enabled'
    }
}

with open(OUTPUT_DIR / 'dataloader_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*80)
print("DATALOADER SETUP COMPLETE!")
print("="*80)

print(f"\n Configuration saved to: {OUTPUT_DIR / 'dataloader_config.json'}")

print("\n Summary:")
print(f"   • Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Training batches per epoch: {len(train_loader)}")
print(f"   • Validation batches per epoch: {len(val_loader)}")
print(f"   • Classes: {len(CLASS_COLUMNS)}")