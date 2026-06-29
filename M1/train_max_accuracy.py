"""
Maximum Test Accuracy Training Script
Uses full DARU-Net model with all 4000 images and advanced optimizations
Target: >90% test accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import os
import time
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from darunet import DARU_Net
from dataset import DualSentinelDataset

class AdvancedLossFunction(nn.Module):
    """Advanced loss function for maximum accuracy"""
    def __init__(self, focal_weight=0.3, dice_weight=0.4, ce_weight=0.2, boundary_weight=0.1):
        super(AdvancedLossFunction, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :]  # Get positive class
        targets = targets.float()
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice
    
    def boundary_loss(self, inputs, targets):
        """Focus on boundary regions"""
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :]
        targets = targets.float()
        
        # Create boundary map using morphological operations
        kernel = torch.ones(3, 3, device=targets.device)
        targets_eroded = F.max_pool2d(-targets.unsqueeze(1), 3, stride=1, padding=1)
        targets_eroded = -targets_eroded.squeeze(1)
        boundary = targets - targets_eroded
        
        # Focus loss on boundary regions
        boundary_loss = F.mse_loss(inputs * boundary, targets * boundary)
        return boundary_loss
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        ce = F.cross_entropy(inputs, targets.long())
        boundary = self.boundary_loss(inputs, targets)
        
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.ce_weight * ce + 
                     self.boundary_weight * boundary)
        
        return total_loss

def create_max_accuracy_datasets(s1_dir, s2_dir, mask_dir, train_split=0.9, val_split=0.05, input_size=(256, 256)):
    """Create datasets using ALL 4000 images with optimal input size for accuracy"""
    print(f"📊 Loading ALL data for maximum test accuracy...")
    
    # Get ALL files
    s1_files = sorted(list(Path(s1_dir).glob('*.png')))
    s2_files = sorted(list(Path(s2_dir).glob('*.png')))
    mask_files = sorted(list(Path(mask_dir).glob('*.png')))
    
    print(f"Found {len(s1_files)} S1 files, {len(s2_files)} S2 files, {len(mask_files)} mask files")
    
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
    
    # 90% train, 5% val, 5% test split for maximum training data
    total_size = len(valid_triplets)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    np.random.seed(42)
    indices = np.random.permutation(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_triplets = [valid_triplets[i] for i in train_indices]
    val_triplets = [valid_triplets[i] for i in val_indices]
    test_triplets = [valid_triplets[i] for i in test_indices]
    
    # Create datasets with optimal input size for accuracy
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
    
    print(f"📊 Maximum Accuracy Data Split:")
    print(f"   Train: {len(train_dataset)} samples ({len(train_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Val: {len(val_dataset)} samples ({len(val_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Test: {len(test_dataset)} samples ({len(test_dataset)/len(valid_triplets)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

def calculate_comprehensive_metrics(outputs, targets):
    """Calculate comprehensive metrics including IoU and Dice"""
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=1)
        targets = targets.long()
        
        # Basic metrics
        correct = (predictions == targets).float().sum()
        total = torch.numel(targets)
        accuracy = (correct / total) * 100
        
        # F1, Precision, Recall
        tp = ((predictions == 1) & (targets == 1)).float().sum()
        fp = ((predictions == 1) & (targets == 0)).float().sum()
        fn = ((predictions == 0) & (targets == 1)).float().sum()
        tn = ((predictions == 0) & (targets == 0)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # IoU (Intersection over Union)
        iou = tp / (tp + fp + fn + 1e-8)
        
        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        
        return {
            'accuracy': accuracy.item(),
            'f1': f1.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'iou': iou.item(),
            'dice': dice.item()
        }

def train_for_maximum_accuracy():
    """Training function optimized for maximum test accuracy"""
    print("🎯 Starting Training for MAXIMUM TEST ACCURACY")
    print("=" * 60)
    print("🎯 Target: >90% test accuracy using full DARU-Net")
    
    # Configuration optimized for maximum accuracy
    config = {
        'batch_size': 8,  # Smaller batch for stability
        'learning_rate': 1e-3,  # Conservative learning rate
        'weight_decay': 1e-4,
        'num_epochs': 100,  # More epochs for convergence
        'patience': 20,  # More patience for best results
        'input_size': (256, 256),  # Full resolution for accuracy
        'use_enhanced_complexity': True,
        'use_paper_config': False
    }
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create datasets with ALL data
    print("\n📊 Creating datasets with ALL 4000 images...")
    train_dataset, val_dataset, test_dataset = create_max_accuracy_datasets(
        s1_dir='data/sentinel1',
        s2_dir='data/sentinel2', 
        mask_dir='data/masks',
        input_size=config['input_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Initialize FULL DARU-Net model for maximum accuracy
    print("\n🧠 Initializing FULL DARU-Net for maximum accuracy...")
    model = DARU_Net(
        use_paper_config=config['use_paper_config'],
        enhanced_complexity=config['use_enhanced_complexity']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (full model for max accuracy)")
    
    # Advanced loss function and optimizer
    criterion = AdvancedLossFunction()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Advanced learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'] * 5,
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    print(f"\n🏋️ Starting training for {config['num_epochs']} epochs...")
    start_time = time.time()
    best_f1 = 0.0
    best_accuracy = 0.0
    patience_counter = 0
    
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0, 'dice': 0}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for s1_imgs, s2_imgs, masks in progress_bar:
            masks = masks.squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(outputs, masks)
            
            train_loss += loss.item()
            for key in train_metrics:
                train_metrics[key] += metrics[key]
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{metrics["accuracy"]:.1f}%',
                'F1': f'{metrics["f1"]:.3f}',
                'IoU': f'{metrics["iou"]:.3f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
        val_batches = 0
        
        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in val_loader:
                masks = masks.squeeze(1)
                outputs = model(s1_imgs, s2_imgs)
                loss = criterion(outputs, masks)
                
                metrics = calculate_comprehensive_metrics(outputs, masks)
                
                val_loss += loss.item()
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
                val_batches += 1
        
        # Calculate averages
        train_loss /= num_batches
        val_loss /= val_batches
        
        for key in train_metrics:
            train_metrics[key] /= num_batches
        for key in val_metrics:
            val_metrics[key] /= val_batches
        
        epoch_time = time.time() - epoch_start
        
        # Print comprehensive results
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.1f}%, F1: {train_metrics['f1']:.3f}, IoU: {train_metrics['iou']:.3f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.1f}%, F1: {val_metrics['f1']:.3f}, IoU: {val_metrics['iou']:.3f}")
        print(f"Val   - Precision: {val_metrics['precision']:.3f}, Recall: {val_metrics['recall']:.3f}, Dice: {val_metrics['dice']:.3f}")
        
        # Save best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_accuracy = val_metrics['accuracy']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, 'results/max_accuracy_best_model.pth')
            
            print(f"💾 NEW BEST MODEL! F1: {val_metrics['f1']:.3f}, Acc: {val_metrics['accuracy']:.1f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"⏹️ Early stopping after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config
            }, f'results/checkpoint_epoch_{epoch+1}.pth')
    
    # Final test evaluation
    print("\n🧪 FINAL TEST EVALUATION...")
    model.eval()
    test_loss = 0.0
    test_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    test_batches = 0
    
    with torch.no_grad():
        for s1_imgs, s2_imgs, masks in tqdm(test_loader, desc="Final Test"):
            masks = masks.squeeze(1)
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks)
            
            metrics = calculate_comprehensive_metrics(outputs, masks)
            
            test_loss += loss.item()
            for key in test_metrics:
                test_metrics[key] += metrics[key]
            test_batches += 1
    
    # Calculate final test averages
    test_loss /= test_batches
    for key in test_metrics:
        test_metrics[key] /= test_batches
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/3600:.1f} hours")
    print(f"📈 Best Validation F1: {best_f1:.3f}")
    print(f"📈 Best Validation Accuracy: {best_accuracy:.1f}%")
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.1f}%")
    print(f"   Test F1-Score: {test_metrics['f1']:.3f}")
    print(f"   Test IoU: {test_metrics['iou']:.3f}")
    print(f"   Test Dice: {test_metrics['dice']:.3f}")
    print(f"   Test Precision: {test_metrics['precision']:.3f}")
    print(f"   Test Recall: {test_metrics['recall']:.3f}")
    
    # Save comprehensive results
    results = {
        'best_val_f1': best_f1,
        'best_val_accuracy': best_accuracy,
        'test_metrics': test_metrics,
        'test_loss': test_loss,
        'training_time_hours': total_time/3600,
        'total_epochs': epoch + 1,
        'model_parameters': total_params,
        'config': config
    }
    
    with open('results/max_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_for_maximum_accuracy()
    print(f"\n✅ Maximum accuracy training completed!")
    print(f"🎯 Final Test Accuracy: {results['test_metrics']['accuracy']:.1f}%")
