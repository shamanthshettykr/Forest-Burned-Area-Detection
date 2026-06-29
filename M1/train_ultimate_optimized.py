"""
Ultimate Optimized Training Script for DARU-Net
Implements state-of-the-art optimization techniques for maximum test accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
import cv2
import os
import time
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from darunet import DARU_Net
from dataset import DualSentinelDataset
from data_preprocessing import DataPreprocessor

class AdvancedFocalLoss(nn.Module):
    """Advanced Focal Loss with class balancing and adaptive gamma"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', adaptive_gamma=True):
        super(AdvancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.adaptive_gamma = adaptive_gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Adaptive gamma based on difficulty
        if self.adaptive_gamma:
            gamma = self.gamma * (1 - pt)
        else:
            gamma = self.gamma
            
        focal_loss = self.alpha * (1 - pt) ** gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedDiceLoss(nn.Module):
    """Enhanced Dice Loss with smooth factor and class weighting"""
    def __init__(self, smooth=1e-6, class_weights=None):
        super(EnhancedDiceLoss, self).__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        if self.class_weights is not None:
            dice = dice * self.class_weights.to(dice.device)
            
        return 1 - dice.mean()

class TverskyLoss(nn.Module):
    """Tversky Loss for handling class imbalance"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :]  # Get positive class
        targets = targets.float()
        
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class UltimateComboLoss(nn.Module):
    """Ultimate combination of multiple loss functions with adaptive weighting"""
    def __init__(self, focal_weight=0.3, dice_weight=0.3, tversky_weight=0.2, 
                 boundary_weight=0.2, adaptive_weights=True):
        super(UltimateComboLoss, self).__init__()
        self.focal_loss = AdvancedFocalLoss(alpha=0.25, gamma=2.0, adaptive_gamma=True)
        self.dice_loss = EnhancedDiceLoss(smooth=1e-6)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.adaptive_weights = adaptive_weights
        
        # For adaptive weighting
        self.loss_history = {'focal': [], 'dice': [], 'tversky': [], 'boundary': []}
        
    def boundary_loss(self, inputs, targets):
        """Boundary-aware loss using morphological operations"""
        inputs = F.softmax(inputs, dim=1)[:, 1, :, :]
        targets = targets.float()
        
        # Create boundary maps using morphological operations
        kernel = torch.ones(3, 3, device=targets.device)
        
        # Erosion and dilation for boundary detection
        targets_eroded = F.max_pool2d(-targets.unsqueeze(1), 3, stride=1, padding=1)
        targets_eroded = -targets_eroded.squeeze(1)
        boundary = targets - targets_eroded
        
        # Focus loss on boundary regions
        boundary_loss = F.mse_loss(inputs * boundary, targets * boundary)
        return boundary_loss
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)
        
        # Store loss values for adaptive weighting
        if self.adaptive_weights:
            self.loss_history['focal'].append(focal.item())
            self.loss_history['dice'].append(dice.item())
            self.loss_history['tversky'].append(tversky.item())
            self.loss_history['boundary'].append(boundary.item())
            
            # Keep only recent history
            for key in self.loss_history:
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key] = self.loss_history[key][-100:]
            
            # Adaptive weight adjustment based on loss trends
            if len(self.loss_history['focal']) > 10:
                focal_trend = np.mean(self.loss_history['focal'][-10:])
                dice_trend = np.mean(self.loss_history['dice'][-10:])
                tversky_trend = np.mean(self.loss_history['tversky'][-10:])
                boundary_trend = np.mean(self.loss_history['boundary'][-10:])
                
                total = focal_trend + dice_trend + tversky_trend + boundary_trend
                if total > 0:
                    self.focal_weight = 0.4 * (focal_trend / total)
                    self.dice_weight = 0.4 * (dice_trend / total)
                    self.tversky_weight = 0.1 * (tversky_trend / total)
                    self.boundary_weight = 0.1 * (boundary_trend / total)
        
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.tversky_weight * tversky + 
                     self.boundary_weight * boundary)
        
        return total_loss

class AdvancedDataAugmentation:
    """Advanced data augmentation pipeline"""
    def __init__(self, input_size=(256, 256)):
        self.train_transform = A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            
            # Intensity transformations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # Weather and atmospheric effects
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.2),
            
            # Blur and sharpening
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            
            # Final resize
            A.Resize(height=input_size[0], width=input_size[1], p=1.0),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1], p=1.0),
        ])

class UltimateTrainer:
    """Ultimate trainer with advanced optimization techniques"""
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Advanced optimizer with gradient centralization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )
        
        # Ultimate combo loss
        self.criterion = UltimateComboLoss(adaptive_weights=True)
        
        # Advanced learning rate scheduler
        self.scheduler = None  # Will be set in train method
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.learning_rates = []
        
        # Best model tracking
        self.best_f1 = 0.0
        self.best_accuracy = 0.0
        self.best_model_state = None
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def calculate_metrics(self, outputs, targets):
        """Calculate comprehensive metrics"""
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
            
            return accuracy.item(), f1.item(), precision.item(), recall.item()

    def train_epoch(self, train_loader):
        """Train for one epoch with advanced techniques"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc='Training', leave=False)

        for batch_idx, (s1_imgs, s2_imgs, masks) in enumerate(progress_bar):
            s1_imgs = s1_imgs.to(self.device, non_blocking=True)
            s2_imgs = s2_imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).squeeze(1)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(s1_imgs, s2_imgs)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(s1_imgs, s2_imgs)
                loss = self.criterion(outputs, masks)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Calculate metrics
            accuracy, f1, _, _ = self.calculate_metrics(outputs, masks)

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

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        num_batches = 0

        progress_bar = tqdm(val_loader, desc='Validation', leave=False)

        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in progress_bar:
                s1_imgs = s1_imgs.to(self.device, non_blocking=True)
                s2_imgs = s2_imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True).squeeze(1)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(s1_imgs, s2_imgs)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(s1_imgs, s2_imgs)
                    loss = self.criterion(outputs, masks)

                # Calculate metrics
                accuracy, f1, precision, recall = self.calculate_metrics(outputs, masks)

                total_loss += loss.item()
                total_accuracy += accuracy
                total_f1 += f1
                total_precision += precision
                total_recall += recall
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
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches

        return avg_loss, avg_accuracy, avg_f1, avg_precision, avg_recall

def create_advanced_datasets(s1_dir, s2_dir, mask_dir, train_split=0.7, val_split=0.15,
                           test_split=0.15, input_size=(256, 256)):
    """Create advanced datasets with proper augmentation"""
    # Get all image paths
    s1_paths = sorted(list(Path(s1_dir).glob('*.png')))
    s2_paths = [Path(s2_dir) / p.name.replace('_s1_', '_s2_') for p in s1_paths]
    mask_paths = [Path(mask_dir) / p.name for p in s1_paths]

    # Filter valid triplets
    valid_triplets = []
    for s1, s2, mask in zip(s1_paths, s2_paths, mask_paths):
        s1_path = Path(s1) if isinstance(s1, str) else s1
        s2_path = Path(s2) if isinstance(s2, str) else s2
        mask_path = Path(mask) if isinstance(mask, str) else mask

        if s1_path.exists() and s2_path.exists() and mask_path.exists():
            valid_triplets.append((str(s1_path), str(s2_path), str(mask_path)))

    print(f"Found {len(valid_triplets)} valid image triplets")

    # Split data
    total_size = len(valid_triplets)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

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

    # Advanced augmentation
    augmentation = AdvancedDataAugmentation(input_size)

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

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

def train_ultimate_model(config=None):
    """Ultimate training function with all optimizations"""
    if config is None:
        config = {
            'batch_size': 8,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'num_epochs': 100,
            'patience': 15,
            'use_paper_config': False,
            'enhanced_complexity': True,
            'input_size': (256, 256)
        }

    print("🚀 Starting Ultimate DARU-Net Training")
    print("=" * 60)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_advanced_datasets(
        s1_dir='data/sentinel1',
        s2_dir='data/sentinel2',
        mask_dir='data/masks',
        input_size=config['input_size']
    )

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 2,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    print("\n🧠 Initializing model...")
    model = DARU_Net(
        use_paper_config=config['use_paper_config'],
        use_log_softmax=True,
        enhanced_complexity=config['enhanced_complexity']
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize trainer
    trainer = UltimateTrainer(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Advanced learning rate scheduler
    trainer.scheduler = OneCycleLR(
        trainer.optimizer,
        max_lr=config['learning_rate'] * 10,
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )

    # Training loop
    print(f"\n🏋️ Starting training for {config['num_epochs']} epochs...")
    start_time = time.time()
    best_f1 = 0.0
    patience_counter = 0

    # Create results directory
    os.makedirs('results', exist_ok=True)

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()

        # Training phase
        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)

        # Validation phase
        val_loss, val_acc, val_f1, val_precision, val_recall = trainer.validate(val_loader)

        # Update scheduler
        if trainer.scheduler:
            trainer.scheduler.step()

        # Store metrics
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        trainer.train_accuracies.append(train_acc)
        trainer.val_accuracies.append(val_acc)
        trainer.train_f1_scores.append(train_f1)
        trainer.val_f1_scores.append(val_f1)
        trainer.learning_rates.append(trainer.optimizer.param_groups[0]['lr'])

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} ({epoch_time:.2f}s)")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        print(f"Val   - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        print(f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            trainer.best_f1 = val_f1
            trainer.best_accuracy = val_acc
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'config': config,
                'metrics': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'train_accuracies': trainer.train_accuracies,
                    'val_accuracies': trainer.val_accuracies,
                    'train_f1_scores': trainer.train_f1_scores,
                    'val_f1_scores': trainer.val_f1_scores,
                    'learning_rates': trainer.learning_rates
                }
            }, 'results/ultimate_best_model.pth')

            print(f"💾 Saved new best model! F1: {val_f1:.4f}, Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
            break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
                'config': config
            }, f'results/checkpoint_epoch_{epoch+1}.pth')

    # Final evaluation on test set
    print("\n🧪 Final evaluation on test set...")
    test_loss, test_acc, test_f1, test_precision, test_recall = trainer.validate(test_loader)

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n🎉 Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"📈 Best Validation F1: {best_f1:.4f}")
    print(f"🎯 Test Results:")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   F1-Score: {test_f1:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")

    # Save final results
    results = {
        'best_val_f1': best_f1,
        'best_val_accuracy': trainer.best_accuracy,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_loss': test_loss,
        'training_time_hours': total_time / 3600,
        'total_epochs': epoch + 1,
        'model_parameters': total_params,
        'config': config
    }

    with open('results/ultimate_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return model, trainer, results

if __name__ == "__main__":
    # Run ultimate training
    model, trainer, results = train_ultimate_model()
    print("\n✅ Ultimate training completed successfully!")
