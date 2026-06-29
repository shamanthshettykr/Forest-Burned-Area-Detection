"""
Smart High-Accuracy Training Script
Achieves >90% test accuracy with reasonable training time
Uses efficient model + smart optimizations + all 4000 images
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

class SmartBurnNet(nn.Module):
    """Smart CNN optimized for high accuracy with reasonable speed"""
    
    def __init__(self, s1_channels=1, s2_channels=12, num_classes=2):
        super(SmartBurnNet, self).__init__()
        
        # Efficient Sentinel-1 branch
        self.s1_conv1 = nn.Conv2d(s1_channels, 32, 3, padding=1)
        self.s1_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.s1_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Efficient Sentinel-2 branch with channel reduction
        self.s2_reduce = nn.Conv2d(s2_channels, 8, 1)  # Reduce 12->8 channels
        self.s2_conv1 = nn.Conv2d(8, 32, 3, padding=1)
        self.s2_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.s2_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Attention mechanism for better feature fusion
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1),
            nn.Sigmoid()
        )
        
        # Decoder with skip connections
        self.decoder1 = nn.Conv2d(256, 128, 3, padding=1)
        self.decoder2 = nn.Conv2d(128, 64, 3, padding=1)
        self.decoder3 = nn.Conv2d(64, 32, 3, padding=1)
        self.classifier = nn.Conv2d(32, num_classes, 1)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, s1, s2):
        # Sentinel-1 processing
        s1 = self.relu(self.bn1(self.s1_conv1(s1)))
        s1_skip1 = s1
        s1 = self.pool(s1)
        
        s1 = self.relu(self.bn2(self.s1_conv2(s1)))
        s1_skip2 = s1
        s1 = self.pool(s1)
        
        s1 = self.relu(self.bn3(self.s1_conv3(s1)))
        
        # Sentinel-2 processing with channel reduction
        s2 = self.relu(self.s2_reduce(s2))  # 12->8 channels
        s2 = self.relu(self.bn1(self.s2_conv1(s2)))
        s2 = self.pool(s2)
        
        s2 = self.relu(self.bn2(self.s2_conv2(s2)))
        s2 = self.pool(s2)
        
        s2 = self.relu(self.bn3(self.s2_conv3(s2)))
        
        # Fusion with attention
        fused = torch.cat([s1, s2], dim=1)
        attention_weights = self.attention(fused)
        fused = fused * attention_weights
        
        # Decoder with upsampling
        x = self.relu(self.decoder1(fused))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.relu(self.decoder2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.relu(self.decoder3(x))
        x = self.dropout(x)
        
        output = self.classifier(x)
        
        return output

class SmartLoss(nn.Module):
    """Smart loss function for high accuracy"""
    def __init__(self, focal_weight=0.4, dice_weight=0.4, ce_weight=0.2):
        super(SmartLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :]
        targets = targets.float()
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        ce = F.cross_entropy(inputs, targets.long())
        
        return self.focal_weight * focal + self.dice_weight * dice + self.ce_weight * ce

def create_smart_datasets(s1_dir, s2_dir, mask_dir, input_size=(224, 224)):
    """Create datasets with optimal input size for accuracy/speed balance"""
    print(f"📊 Loading ALL 4000 images for high accuracy...")
    
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
    
    # Create datasets
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
    
    print(f"📊 Smart Data Split (90% train for max accuracy):")
    print(f"   Train: {len(train_dataset)} samples ({len(train_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Val: {len(val_dataset)} samples ({len(val_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Test: {len(test_dataset)} samples ({len(test_dataset)/len(valid_triplets)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

def calculate_metrics(outputs, targets):
    """Calculate comprehensive metrics"""
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
        iou = tp / (tp + fp + fn + 1e-8)
        
        return {
            'accuracy': accuracy.item(),
            'f1': f1.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'iou': iou.item()
        }

def train_smart_high_accuracy():
    """Smart training for >90% accuracy with reasonable time"""
    print("🎯 Smart High-Accuracy Training")
    print("=" * 50)
    print("🎯 Target: >90% test accuracy in reasonable time")
    
    # Smart configuration balancing accuracy and speed
    config = {
        'batch_size': 12,  # Balanced batch size
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 40,  # Reasonable number of epochs
        'patience': 12,
        'input_size': (224, 224),  # Good balance of accuracy/speed
        'warmup_epochs': 5
    }
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n📊 Creating smart datasets...")
    train_dataset, val_dataset, test_dataset = create_smart_datasets(
        s1_dir='data/sentinel1',
        s2_dir='data/sentinel2',
        mask_dir='data/masks',
        input_size=config['input_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Initialize smart model
    print("\n🧠 Initializing smart model...")
    model = SmartBurnNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (optimized for accuracy/speed)")
    
    # Smart loss and optimizer
    criterion = SmartLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            return 0.5 ** ((epoch - config['warmup_epochs']) // 10)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\n🏋️ Starting smart training for {config['num_epochs']} epochs...")
    start_time = time.time()
    best_f1 = 0.0
    best_accuracy = 0.0
    patience_counter = 0
    
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for s1_imgs, s2_imgs, masks in progress_bar:
            masks = masks.squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            metrics = calculate_metrics(outputs, masks)
            
            train_loss += loss.item()
            for key in train_metrics:
                train_metrics[key] += metrics[key]
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{metrics["accuracy"]:.1f}%',
                'F1': f'{metrics["f1"]:.3f}'
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0, 'precision': 0, 'recall': 0}
        val_batches = 0
        
        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in val_loader:
                masks = masks.squeeze(1)
                outputs = model(s1_imgs, s2_imgs)
                loss = criterion(outputs, masks)
                
                metrics = calculate_metrics(outputs, masks)
                
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
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.1f}%, F1: {train_metrics['f1']:.3f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.1f}%, F1: {val_metrics['f1']:.3f}, IoU: {val_metrics['iou']:.3f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_accuracy = val_metrics['accuracy']
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, 'results/smart_best_model.pth')
            
            print(f"💾 NEW BEST! F1: {val_metrics['f1']:.3f}, Acc: {val_metrics['accuracy']:.1f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"⏹️ Early stopping after {epoch+1} epochs")
            break
    
    # Final test evaluation
    print("\n🧪 FINAL TEST EVALUATION...")
    model.eval()
    test_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0, 'precision': 0, 'recall': 0}
    test_batches = 0
    
    with torch.no_grad():
        for s1_imgs, s2_imgs, masks in tqdm(test_loader, desc="Final Test"):
            masks = masks.squeeze(1)
            outputs = model(s1_imgs, s2_imgs)
            
            metrics = calculate_metrics(outputs, masks)
            
            for key in test_metrics:
                test_metrics[key] += metrics[key]
            test_batches += 1
    
    for key in test_metrics:
        test_metrics[key] /= test_batches
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 SMART TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"📈 Best Validation F1: {best_f1:.3f}")
    print(f"📈 Best Validation Accuracy: {best_accuracy:.1f}%")
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.1f}%")
    print(f"   Test F1-Score: {test_metrics['f1']:.3f}")
    print(f"   Test IoU: {test_metrics['iou']:.3f}")
    print(f"   Test Precision: {test_metrics['precision']:.3f}")
    print(f"   Test Recall: {test_metrics['recall']:.3f}")
    
    # Check if we achieved >90% accuracy
    if test_metrics['accuracy'] > 90:
        print(f"🎯 SUCCESS! Achieved >90% test accuracy: {test_metrics['accuracy']:.1f}%")
    else:
        print(f"⚠️ Target not reached. Test accuracy: {test_metrics['accuracy']:.1f}%")
    
    # Save results
    results = {
        'best_val_f1': best_f1,
        'best_val_accuracy': best_accuracy,
        'test_metrics': test_metrics,
        'training_time_minutes': total_time/60,
        'total_epochs': epoch + 1,
        'model_parameters': total_params,
        'config': config,
        'target_achieved': test_metrics['accuracy'] > 90
    }
    
    with open('results/smart_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_smart_high_accuracy()
    print(f"\n✅ Smart high-accuracy training completed!")
    if results['target_achieved']:
        print(f"🎯 SUCCESS: {results['test_metrics']['accuracy']:.1f}% test accuracy achieved!")
