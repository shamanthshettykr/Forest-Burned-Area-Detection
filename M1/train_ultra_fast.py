"""
Ultra-Fast CPU Training Script
Designed to complete in ~30 minutes instead of 46 hours
Uses lightweight model with 128x128 input and 90% training data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataset import DualSentinelDataset

class UltraFastBurnNet(nn.Module):
    """Ultra-lightweight CNN for burned area detection - optimized for CPU speed"""
    
    def __init__(self, s1_channels=1, s2_channels=12, num_classes=2):
        super(UltraFastBurnNet, self).__init__()
        
        # Very lightweight Sentinel-1 branch
        self.s1_conv1 = nn.Conv2d(s1_channels, 8, 5, stride=2, padding=2)  # Reduce spatial size quickly
        self.s1_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        
        # Very lightweight Sentinel-2 branch
        self.s2_conv1 = nn.Conv2d(s2_channels, 8, 5, stride=2, padding=2)  # Reduce spatial size quickly
        self.s2_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        
        # Simple fusion
        self.fusion = nn.Conv2d(32, 16, 3, padding=1)
        self.classifier = nn.Conv2d(16, num_classes, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, s1, s2):
        # Process Sentinel-1
        s1 = self.relu(self.s1_conv1(s1))  # 128->64
        s1 = self.pool(s1)  # 64->32
        s1 = self.relu(self.s1_conv2(s1))
        
        # Process Sentinel-2
        s2 = self.relu(self.s2_conv1(s2))  # 128->64
        s2 = self.pool(s2)  # 64->32
        s2 = self.relu(self.s2_conv2(s2))
        
        # Fuse and classify
        fused = torch.cat([s1, s2], dim=1)
        fused = self.relu(self.fusion(fused))
        output = self.classifier(fused)
        
        # Upsample to original size
        output = F.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)
        
        return output

def create_fast_datasets(s1_dir, s2_dir, mask_dir):
    """Create datasets using ALL 4000 images for maximum test accuracy"""
    print(f"📊 Loading ALL data for maximum test accuracy...")

    # Get ALL files - no limit for best accuracy
    s1_files = sorted(list(Path(s1_dir).glob('*.png')))
    s2_files = sorted(list(Path(s2_dir).glob('*.png')))
    mask_files = sorted(list(Path(mask_dir).glob('*.png')))
    
    print(f"Found {len(s1_files)} S1 files (limited), {len(s2_files)} S2 files, {len(mask_files)} mask files")
    
    # Create mappings
    s2_mapping = {s2.name.replace('_s2_', '_s1_'): s2 for s2 in s2_files}
    mask_mapping = {mask.name: mask for mask in mask_files}
    
    # Find valid triplets
    valid_triplets = []
    for s1_file in s1_files:
        s1_name = s1_file.name
        if s1_name in s2_mapping and s1_name in mask_mapping:
            s2_file = s2_mapping[s1_name]
            mask_file = mask_mapping[s1_name]
            if s1_file.exists() and s2_file.exists() and mask_file.exists():
                valid_triplets.append((str(s1_file), str(s2_file), str(mask_file)))
    
    print(f"Found {len(valid_triplets)} valid triplets")
    
    # 90% train, 5% val, 5% test split
    total_size = len(valid_triplets)
    train_size = int(total_size * 0.9)
    val_size = int(total_size * 0.05)
    
    np.random.seed(42)
    indices = np.random.permutation(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_triplets = [valid_triplets[i] for i in train_indices]
    val_triplets = [valid_triplets[i] for i in val_indices]
    test_triplets = [valid_triplets[i] for i in test_indices]
    
    # Create datasets with small input size
    input_size = (128, 128)
    
    train_dataset = DualSentinelDataset(
        [t[0] for t in train_triplets], [t[1] for t in train_triplets], [t[2] for t in train_triplets],
        transform=True, input_size=input_size
    )
    
    val_dataset = DualSentinelDataset(
        [t[0] for t in val_triplets], [t[1] for t in val_triplets], [t[2] for t in val_triplets],
        transform=False, input_size=input_size
    )
    
    test_dataset = DualSentinelDataset(
        [t[0] for t in test_triplets], [t[1] for t in test_triplets], [t[2] for t in test_triplets],
        transform=False, input_size=input_size
    )
    
    print(f"📊 Fast Data Split:")
    print(f"   Train: {len(train_dataset)} samples ({len(train_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Val: {len(val_dataset)} samples ({len(val_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Test: {len(test_dataset)} samples ({len(test_dataset)/len(valid_triplets)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

def calculate_metrics(outputs, targets):
    """Calculate accuracy and F1 score"""
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=1)
        targets = targets.long()
        
        correct = (predictions == targets).float().sum()
        total = torch.numel(targets)
        accuracy = (correct / total) * 100
        
        tp = ((predictions == 1) & (targets == 1)).float().sum()
        fp = ((predictions == 1) & (targets == 0)).float().sum()
        fn = ((predictions == 0) & (targets == 1)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return accuracy.item(), f1.item()

def train_ultra_fast():
    """Ultra-fast training function"""
    print("⚡ Starting CPU Training with ALL 4000 Images")
    print("=" * 50)
    print("🎯 Target: Maximum test accuracy using full dataset")
    
    # Configuration for maximum test accuracy with ALL 4000 images
    config = {
        'batch_size': 16,  # Smaller batch to handle full dataset
        'learning_rate': 2e-3,  # Moderate LR for stability with full data
        'num_epochs': 30,  # More epochs for full dataset
        'patience': 8,
        'use_all_data': True  # Use ALL 4000 images
    }
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create datasets with full data
    print("\n📊 Creating datasets with ALL 4000 images for maximum accuracy...")
    train_dataset, val_dataset, test_dataset = create_fast_datasets(
        s1_dir='data/sentinel1',
        s2_dir='data/sentinel2',
        mask_dir='data/masks'
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Initialize ultra-lightweight model
    print("\n🧠 Initializing ultra-lightweight model...")
    model = UltraFastBurnNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (ultra-lightweight!)")
    
    # Simple loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    # Training loop
    print(f"\n🏋️ Starting ultra-fast training for {config['num_epochs']} epochs...")
    start_time = time.time()
    best_f1 = 0.0
    patience_counter = 0
    
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_f1 = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for s1_imgs, s2_imgs, masks in progress_bar:
            masks = masks.squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()
            
            accuracy, f1 = calculate_metrics(outputs, masks)
            
            train_loss += loss.item()
            train_acc += accuracy
            train_f1 += f1
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{accuracy:.1f}%',
                'F1': f'{f1:.3f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in val_loader:
                masks = masks.squeeze(1)
                outputs = model(s1_imgs, s2_imgs)
                loss = criterion(outputs, masks.long())
                
                accuracy, f1 = calculate_metrics(outputs, masks)
                
                val_loss += loss.item()
                val_acc += accuracy
                val_f1 += f1
                val_batches += 1
        
        scheduler.step()
        
        # Calculate averages
        train_loss /= num_batches
        train_acc /= num_batches
        train_f1 /= num_batches
        val_loss /= val_batches
        val_acc /= val_batches
        val_f1 /= val_batches
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"Train - Loss: {train_loss:.3f}, Acc: {train_acc:.1f}%, F1: {train_f1:.3f}")
        print(f"Val   - Loss: {val_loss:.3f}, Acc: {val_acc:.1f}%, F1: {val_f1:.3f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'config': config
            }, 'results/ultra_fast_best_model.pth')
            
            print(f"💾 New best! F1: {val_f1:.3f}")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"⏹️ Early stopping after {epoch+1} epochs")
            break
    
    # Test evaluation
    print("\n🧪 Final test evaluation...")
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_f1 = 0.0
    test_batches = 0
    
    with torch.no_grad():
        for s1_imgs, s2_imgs, masks in test_loader:
            masks = masks.squeeze(1)
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks.long())
            
            accuracy, f1 = calculate_metrics(outputs, masks)
            
            test_loss += loss.item()
            test_acc += accuracy
            test_f1 += f1
            test_batches += 1
    
    test_loss /= test_batches
    test_acc /= test_batches
    test_f1 /= test_batches
    
    total_time = time.time() - start_time
    
    print(f"\n⚡ Ultra-fast training completed in {total_time/60:.1f} minutes!")
    print(f"📈 Best Validation F1: {best_f1:.3f}")
    print(f"🎯 Test Results:")
    print(f"   Accuracy: {test_acc:.1f}%")
    print(f"   F1-Score: {test_f1:.3f}")
    
    # Save results
    results = {
        'best_val_f1': best_f1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'training_time_minutes': total_time/60,
        'total_epochs': epoch + 1,
        'model_parameters': total_params,
        'config': config
    }
    
    with open('results/ultra_fast_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_ultra_fast()
    print("\n✅ Ultra-fast training completed!")
