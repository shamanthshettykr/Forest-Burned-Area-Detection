"""
Paper-Compliant DARU-Net Training
Follows the exact methodology from the DARU-Net paper for optimal results
Uses paper's hyperparameters, loss functions, and training strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
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

class PaperCompliantLoss(nn.Module):
    """Loss function exactly as described in DARU-Net paper"""
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super(PaperCompliantLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def focal_loss(self, inputs, targets):
        """Focal loss as per paper"""
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets):
        """Dice loss as per paper"""
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :]  # Get burn class
        targets = targets.float()
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        return 1 - dice
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        # Combined loss as per paper
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        return total_loss

def create_paper_datasets(s1_dir, s2_dir, mask_dir, train_split=0.9, val_split=0.05):
    """Create datasets following paper's data split strategy"""
    print(f"📊 Loading data following paper methodology...")
    
    # Get all files
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
    
    # Paper's data split: 90% train, 5% val, 5% test
    total_size = len(valid_triplets)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    # Deterministic split as per paper
    np.random.seed(42)
    indices = np.random.permutation(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_triplets = [valid_triplets[i] for i in train_indices]
    val_triplets = [valid_triplets[i] for i in val_indices]
    test_triplets = [valid_triplets[i] for i in test_indices]
    
    # Paper uses 256x256 input size
    input_size = (256, 256)
    
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
    
    print(f"📊 Paper-Compliant Data Split:")
    print(f"   Train: {len(train_dataset)} samples ({len(train_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Val: {len(val_dataset)} samples ({len(val_dataset)/len(valid_triplets)*100:.1f}%)")
    print(f"   Test: {len(test_dataset)} samples ({len(test_dataset)/len(valid_triplets)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

def calculate_paper_metrics(outputs, targets):
    """Calculate metrics as reported in the paper"""
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=1)
        targets = targets.long()
        
        # Overall accuracy
        correct = (predictions == targets).float().sum()
        total = torch.numel(targets)
        accuracy = (correct / total) * 100
        
        # Class-specific metrics for burn detection
        tp = ((predictions == 1) & (targets == 1)).float().sum()
        fp = ((predictions == 1) & (targets == 0)).float().sum()
        fn = ((predictions == 0) & (targets == 1)).float().sum()
        tn = ((predictions == 0) & (targets == 0)).float().sum()
        
        # Precision, Recall, F1 for burn class
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # IoU for burn class (as reported in paper)
        iou = tp / (tp + fp + fn + 1e-8)
        
        # Kappa coefficient (often used in remote sensing papers)
        po = (tp + tn) / (tp + tn + fp + fn)  # Observed accuracy
        pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / ((tp + tn + fp + fn) ** 2)  # Expected accuracy
        kappa = (po - pe) / (1 - pe + 1e-8)
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item(),
            'kappa': kappa.item()
        }

def train_paper_compliant():
    """Training following exact paper methodology"""
    print("📄 DARU-Net Paper-Compliant Training")
    print("=" * 50)
    print("🎯 Following paper's exact methodology for optimal results")
    
    # Paper's hyperparameters (reduced batch size for CPU efficiency)
    config = {
        'batch_size': 4,  # Reduced for better CPU performance
        'learning_rate': 1e-3,  # Paper's initial LR
        'weight_decay': 1e-4,  # Paper's weight decay
        'num_epochs': 50,  # Paper's training epochs
        'patience': 15,  # Early stopping patience
        'lr_step_size': 20,  # LR decay step
        'lr_gamma': 0.1,  # LR decay factor
        'input_size': (256, 256)  # Paper's input size
    }
    
    # Force CPU usage as requested - no GPU
    device = torch.device('cpu')
    torch.cuda.is_available = lambda: False  # Force disable CUDA
    print(f"Using device: {device} (CPU ONLY - no GPU)")
    
    # Create datasets following paper
    print("\n📊 Creating paper-compliant datasets...")
    train_dataset, val_dataset, test_dataset = create_paper_datasets(
        s1_dir='data/sentinel1',
        s2_dir='data/sentinel2', 
        mask_dir='data/masks'
    )
    
    # Data loaders as per paper
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,
        drop_last=True  # Paper uses drop_last
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
    
    # Initialize DARU-Net exactly as in paper
    print("\n🧠 Initializing DARU-Net (paper configuration)...")
    model = DARU_Net(
        use_paper_config=True,  # Use exact paper configuration
        enhanced_complexity=False  # Paper's original complexity
    )
    model = model.to(device)  # Explicitly move to CPU
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (paper configuration)")
    
    # Paper's loss function and optimizer
    criterion = PaperCompliantLoss(
        alpha=0.25,  # Paper's focal loss alpha
        gamma=2.0,   # Paper's focal loss gamma
        dice_weight=0.5,  # Paper's loss weights
        focal_weight=0.5
    )
    
    # Paper uses Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)  # Paper's beta values
    )
    
    # Paper's learning rate scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )
    
    # Training loop following paper
    print(f"\n🏋️ Starting paper-compliant training for {config['num_epochs']} epochs...")
    start_time = time.time()
    best_f1 = 0.0
    best_accuracy = 0.0
    patience_counter = 0
    
    os.makedirs('results', exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_iou': [], 'val_iou': []
    }
    
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'kappa': 0}
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        for s1_imgs, s2_imgs, masks in progress_bar:
            # Ensure all tensors are on CPU
            s1_imgs = s1_imgs.to(device)
            s2_imgs = s2_imgs.to(device)
            masks = masks.to(device).squeeze(1)

            optimizer.zero_grad()
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping as often used in papers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Calculate metrics
            metrics = calculate_paper_metrics(outputs, masks)
            
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
        val_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'kappa': 0}
        val_batches = 0
        
        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in val_loader:
                # Ensure all tensors are on CPU
                s1_imgs = s1_imgs.to(device)
                s2_imgs = s2_imgs.to(device)
                masks = masks.to(device).squeeze(1)
                outputs = model(s1_imgs, s2_imgs)
                loss = criterion(outputs, masks)
                
                metrics = calculate_paper_metrics(outputs, masks)
                
                val_loss += loss.item()
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
                val_batches += 1
        
        # Update scheduler
        scheduler.step()
        
        # Calculate averages
        train_loss /= num_batches
        val_loss /= val_batches
        
        for key in train_metrics:
            train_metrics[key] /= num_batches
        for key in val_metrics:
            val_metrics[key] /= val_batches
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        
        epoch_time = time.time() - epoch_start
        
        # Print comprehensive results as in paper
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} ({epoch_time:.1f}s)")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.1f}%, F1: {train_metrics['f1']:.3f}, IoU: {train_metrics['iou']:.3f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.1f}%, F1: {val_metrics['f1']:.3f}, IoU: {val_metrics['iou']:.3f}")
        print(f"Val   - Precision: {val_metrics['precision']:.3f}, Recall: {val_metrics['recall']:.3f}, Kappa: {val_metrics['kappa']:.3f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model based on F1 (common in segmentation papers)
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
                'config': config,
                'history': history
            }, 'results/paper_compliant_best_model.pth')
            
            print(f"💾 NEW BEST MODEL! F1: {val_metrics['f1']:.3f}, Acc: {val_metrics['accuracy']:.1f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"⏹️ Early stopping after {epoch+1} epochs")
            break
    
    # Final test evaluation
    print("\n🧪 FINAL TEST EVALUATION (Paper Metrics)...")
    model.eval()
    test_loss = 0.0
    test_metrics = {'accuracy': 0, 'f1': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'kappa': 0}
    test_batches = 0
    
    with torch.no_grad():
        for s1_imgs, s2_imgs, masks in tqdm(test_loader, desc="Final Test"):
            # Ensure all tensors are on CPU
            s1_imgs = s1_imgs.to(device)
            s2_imgs = s2_imgs.to(device)
            masks = masks.to(device).squeeze(1)
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks)
            
            metrics = calculate_paper_metrics(outputs, masks)
            
            test_loss += loss.item()
            for key in test_metrics:
                test_metrics[key] += metrics[key]
            test_batches += 1
    
    # Calculate final test averages
    test_loss /= test_batches
    for key in test_metrics:
        test_metrics[key] /= test_batches
    
    total_time = time.time() - start_time
    
    print(f"\n📄 PAPER-COMPLIANT TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"📈 Best Validation F1: {best_f1:.3f}")
    print(f"📈 Best Validation Accuracy: {best_accuracy:.1f}%")
    print(f"\n🎯 FINAL TEST RESULTS (Paper Metrics):")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.1f}%")
    print(f"   Test F1-Score: {test_metrics['f1']:.3f}")
    print(f"   Test IoU: {test_metrics['iou']:.3f}")
    print(f"   Test Precision: {test_metrics['precision']:.3f}")
    print(f"   Test Recall: {test_metrics['recall']:.3f}")
    print(f"   Test Kappa: {test_metrics['kappa']:.3f}")
    
    # Check if we achieved paper-level performance
    if test_metrics['accuracy'] > 90:
        print(f"🎯 EXCELLENT! Achieved >90% test accuracy: {test_metrics['accuracy']:.1f}%")
    elif test_metrics['accuracy'] > 85:
        print(f"✅ GOOD! Achieved >85% test accuracy: {test_metrics['accuracy']:.1f}%")
    else:
        print(f"📊 Test accuracy: {test_metrics['accuracy']:.1f}%")
    
    # Save comprehensive results
    results = {
        'best_val_f1': best_f1,
        'best_val_accuracy': best_accuracy,
        'test_metrics': test_metrics,
        'test_loss': test_loss,
        'training_time_minutes': total_time/60,
        'total_epochs': epoch + 1,
        'model_parameters': total_params,
        'config': config,
        'history': history,
        'paper_compliant': True
    }
    
    with open('results/paper_compliant_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results

if __name__ == "__main__":
    model, results = train_paper_compliant()
    print(f"\n✅ Paper-compliant training completed!")
    print(f"📄 Results saved following paper's evaluation methodology")
    print(f"🎯 Final Test Accuracy: {results['test_metrics']['accuracy']:.1f}%")
