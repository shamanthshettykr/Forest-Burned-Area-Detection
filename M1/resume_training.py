"""
Resume training script for DARU-Net from the last checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os
import time
from sklearn.model_selection import train_test_split

# Import our modules
from darunet import DARU_Net, PaperL2Loss
from data_preprocessing import DataPreprocessor
from train_enhanced import EnhancedForestBurnedAreaDataset, EnhancedTrainer

def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    results_dir = Path('results')
    if not results_dir.exists():
        return None
    
    checkpoint_files = list(results_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoint_files[-1]

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint and return epoch and metrics"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {epoch + 1}")
    return epoch + 1, metrics

def main():
    """Main function to resume training"""
    # Configuration (same as original training)
    use_paper_config = True
    enhanced_complexity = True
    use_paper_loss = True
    num_epochs = 50
    batch_size = 2
    learning_rate = 5e-5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Data paths
    s1_dir = Path('data/sentinel1')
    s2_dir = Path('data/sentinel2')
    mask_dir = Path('data/masks')
    
    # Get file paths (same logic as original)
    s1_paths = sorted(list(s1_dir.glob('*.png')))
    valid_triplets = []
    for s1_path in s1_paths:
        s2_path = s2_dir / s1_path.name.replace('_s1_', '_s2_')
        mask_path = mask_dir / s1_path.name
        
        if s2_path.exists() and mask_path.exists():
            valid_triplets.append((s1_path, s2_path, mask_path))
    
    s1_paths, s2_paths, mask_paths = zip(*valid_triplets) if valid_triplets else ([], [], [])
    
    print(f"Found {len(s1_paths)} valid image triplets")
    
    # Split data (same as original)
    s1_train, s1_temp, s2_train, s2_temp, mask_train, mask_temp = train_test_split(
        s1_paths, s2_paths, mask_paths, test_size=0.1, random_state=42
    )
    
    s1_val, _, s2_val, _, mask_val, _ = train_test_split(
        s1_temp, s2_temp, mask_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training samples: {len(s1_train)}")
    print(f"Validation samples: {len(s1_val)}")
    
    # Create datasets
    train_dataset = EnhancedForestBurnedAreaDataset(
        s1_train, s2_train, mask_train, use_paper_config=use_paper_config
    )
    val_dataset = EnhancedForestBurnedAreaDataset(
        s1_val, s2_val, mask_val, use_paper_config=use_paper_config
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    # Initialize model
    model = DARU_Net(
        use_paper_config=use_paper_config,
        use_log_softmax=not use_paper_loss,
        enhanced_complexity=enhanced_complexity
    )
    
    # Initialize trainer
    trainer = EnhancedTrainer(model, device, use_paper_loss=use_paper_loss, learning_rate=learning_rate)
    
    # Find and load checkpoint
    checkpoint_path = find_latest_checkpoint()
    start_epoch = 0
    
    if checkpoint_path:
        start_epoch, metrics = load_checkpoint(
            model, trainer.optimizer, trainer.scheduler, checkpoint_path
        )
        # Restore metrics if available
        if metrics:
            trainer.train_losses = metrics.get('train_losses', [])
            trainer.val_losses = metrics.get('val_losses', [])
            trainer.val_accuracies = metrics.get('val_accuracies', [])
            trainer.val_f1_scores = metrics.get('val_f1_scores', [])
            if trainer.val_f1_scores:
                trainer.best_f1 = max(trainer.val_f1_scores)
    else:
        print("No checkpoint found, starting from scratch")
    
    print(f"Resuming training from epoch {start_epoch + 1} to {num_epochs}")
    print(f"Current best F1: {trainer.best_f1:.4f}")
    
    # Training loop
    start_time = time.time()
    patience = 15
    no_improve_count = 0
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_acc, val_f1 = trainer.validate(val_loader)
        
        # Update scheduler
        trainer.scheduler.step()
        
        # Store metrics
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        trainer.val_accuracies.append(val_acc)
        trainer.val_f1_scores.append(val_f1)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > trainer.best_f1:
            trainer.best_f1 = val_f1
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'config': {
                    'use_paper_config': use_paper_config,
                    'enhanced_complexity': enhanced_complexity,
                    'use_paper_loss': use_paper_loss
                }
            }, 'results/best_enhanced_model.pth')
            print(f"Saved new best model with F1: {val_f1:.4f}")
        else:
            no_improve_count += 1
        
        # Early stopping check
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'metrics': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'val_accuracies': trainer.val_accuracies,
                    'val_f1_scores': trainer.val_f1_scores
                }
            }, f'results/checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation F1 score: {trainer.best_f1:.4f}")

if __name__ == '__main__':
    main()
