"""
Simplified Optimized Training Script for DARU-Net
Focus on getting the best test accuracy with robust data handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np
import cv2
import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from darunet import DARU_Net
from dataset import DualSentinelDataset

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :]  # Get positive class
        targets = targets.float()
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combined loss function"""
    def __init__(self, focal_weight=0.5, dice_weight=0.3, ce_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets.long())
        
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.ce_weight * ce)
        
        return total_loss

def create_simple_datasets(s1_dir, s2_dir, mask_dir, train_split=0.8, val_split=0.1, input_size=(256, 256)):
    """Create datasets with simple robust pairing"""
    print(f"📊 Loading data from directories...")
    
    # Get all files
    s1_files = sorted(list(Path(s1_dir).glob('*.png')))
    s2_files = sorted(list(Path(s2_dir).glob('*.png')))
    mask_files = sorted(list(Path(mask_dir).glob('*.png')))
    
    print(f"Found {len(s1_files)} S1 files, {len(s2_files)} S2 files, {len(mask_files)} mask files")
    
    # Create mapping for S2 files
    s2_mapping = {}
    for s2_file in s2_files:
        s1_name = s2_file.name.replace('_s2_', '_s1_')
        s2_mapping[s1_name] = s2_file
    
    # Create mapping for mask files
    mask_mapping = {}
    for mask_file in mask_files:
        mask_mapping[mask_file.name] = mask_file
    
    # Find valid triplets
    valid_triplets = []
    for s1_file in s1_files:
        s1_name = s1_file.name
        
        # Find corresponding S2 file
        if s1_name in s2_mapping:
            s2_file = s2_mapping[s1_name]
            
            # Find corresponding mask file
            if s1_name in mask_mapping:
                mask_file = mask_mapping[s1_name]
                
                # Verify all files exist
                if s1_file.exists() and s2_file.exists() and mask_file.exists():
                    valid_triplets.append((str(s1_file), str(s2_file), str(mask_file)))
    
    print(f"Found {len(valid_triplets)} valid triplets")
    
    if len(valid_triplets) == 0:
        raise ValueError("No valid triplets found! Check your data directory structure.")
    
    # Split data
    total_size = len(valid_triplets)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_triplets = [valid_triplets[i] for i in train_indices]
    val_triplets = [valid_triplets[i] for i in val_indices]
    test_triplets = [valid_triplets[i] for i in test_indices]
    
    train_dataset = DualSentinelDataset(
        [t[0] for t in train_triplets],
        [t[1] for t in train_triplets], 
        [t[2] for t in train_triplets],
        transform=True,
        input_size=input_size
    )
    
    val_dataset = DualSentinelDataset(
        [t[0] for t in val_triplets],
        [t[1] for t in val_triplets],
        [t[2] for t in val_triplets],
        transform=False,
        input_size=input_size
    )
    
    test_dataset = DualSentinelDataset(
        [t[0] for t in test_triplets],
        [t[1] for t in test_triplets],
        [t[2] for t in test_triplets],
        transform=False,
        input_size=input_size
    )
    
    print(f"📊 Data Split (80% train, 10% val, 10% test):")
    print(f"   Train: {len(train_dataset)} samples ({len(train_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Val: {len(val_dataset)} samples ({len(val_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Test: {len(test_dataset)} samples ({len(test_dataset)/len(valid_triplets)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

def calculate_metrics(outputs, targets):
    """Calculate accuracy and F1 score"""
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=1)
        targets = targets.long()
        
        # Accuracy
        correct = (predictions == targets).float().sum()
        total = torch.numel(targets)
        accuracy = (correct / total) * 100
        
        # F1 Score
        tp = ((predictions == 1) & (targets == 1)).float().sum()
        fp = ((predictions == 1) & (targets == 0)).float().sum()
        fn = ((predictions == 0) & (targets == 1)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return accuracy.item(), f1.item()

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for s1_imgs, s2_imgs, masks in progress_bar:
        s1_imgs = s1_imgs.to(device)
        s2_imgs = s2_imgs.to(device)
        masks = masks.to(device).squeeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(s1_imgs, s2_imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Calculate metrics
        accuracy, f1 = calculate_metrics(outputs, masks)
        
        total_loss += loss.item()
        total_accuracy += accuracy
        total_f1 += f1
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%',
            'F1': f'{f1:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_f1 = total_f1 / num_batches
    
    return avg_loss, avg_accuracy, avg_f1

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    num_batches = 0
    
    progress_bar = tqdm(val_loader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for s1_imgs, s2_imgs, masks in progress_bar:
            s1_imgs = s1_imgs.to(device)
            s2_imgs = s2_imgs.to(device)
            masks = masks.to(device).squeeze(1)
            
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            accuracy, f1 = calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            total_f1 += f1
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%',
                'F1': f'{f1:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_f1 = total_f1 / num_batches
    
    return avg_loss, avg_accuracy, avg_f1

def train_optimized_model():
    """Main training function"""
    print("🚀 Starting Optimized DARU-Net Training")
    print("=" * 50)
    
    # CPU-optimized configuration
    config = {
        'batch_size': 4,  # Smaller batch for CPU
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 50,  # Fewer epochs for faster training
        'patience': 10,
        'input_size': (256, 256)
    }
    
    # Force CPU usage as requested
    device = torch.device('cpu')
    print(f"Using device: {device} (forced CPU)")
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_simple_datasets(
        s1_dir='data/sentinel1',
        s2_dir='data/sentinel2', 
        mask_dir='data/masks',
        input_size=config['input_size']
    )
    
    # Create data loaders optimized for CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Use 0 for CPU to avoid multiprocessing issues
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model with proper 12-channel support
    print("\n🧠 Initializing model with 12-channel Sentinel-2 support...")
    model = DARU_Net(use_paper_config=False, enhanced_complexity=True)  # Full model for 12 channels
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = CombinedLoss(focal_weight=0.4, dice_weight=0.4, ce_weight=0.2)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'] * 10,
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    # Training loop
    print(f"\n🏋️ Starting training for {config['num_epochs']} epochs...")
    start_time = time.time()
    best_f1 = 0.0
    patience_counter = 0
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Training phase
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        
        # Validation phase
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} ({epoch_time:.2f}s)")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'config': config,
                'history': history
            }, 'results/best_model.pth')
            
            print(f"💾 Saved new best model! F1: {val_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print("\n🧪 Final evaluation on test set...")
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📈 Best Validation F1: {best_f1:.4f}")
    print(f"🎯 Test Results:")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   F1-Score: {test_f1:.4f}")
    
    # Save final results
    results = {
        'best_val_f1': best_f1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'training_time': total_time,
        'total_epochs': epoch + 1,
        'config': config,
        'history': history
    }
    
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_optimized_model()
    print("\n✅ Training completed successfully!")
