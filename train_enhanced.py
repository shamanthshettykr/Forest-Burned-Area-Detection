"""
Enhanced training script for DARU-Net with paper-compliant configuration
and increased complexity for better accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import train_test_split

# Import our modules
from darunet import DARU_Net, PaperL2Loss
from data_preprocessing import DataPreprocessor

class EnhancedForestBurnedAreaDataset(Dataset):
    """Enhanced dataset with proper preprocessing"""
    
    def __init__(self, s1_paths, s2_paths, mask_paths, use_paper_config=True, transform=None):
        self.s1_paths = s1_paths
        self.s2_paths = s2_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocessor = DataPreprocessor(use_paper_config=use_paper_config)
        
    def __len__(self):
        return len(self.s1_paths)
    
    def __getitem__(self, idx):
        try:
            # Load Sentinel-1 image
            s1_img = cv2.imread(str(self.s1_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if s1_img is None:
                raise ValueError(f"Could not load S1 image: {self.s1_paths[idx]}")
            
            # Load Sentinel-2 image
            s2_img = cv2.imread(str(self.s2_paths[idx]))
            if s2_img is None:
                raise ValueError(f"Could not load S2 image: {self.s2_paths[idx]}")
            
            # Load mask
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {self.mask_paths[idx]}")
            
            # Preprocess images
            s1_processed = self.preprocessor.preprocess_sentinel1(s1_img)
            s2_processed = self.preprocessor.preprocess_sentinel2(s2_img)
            
            # Ensure proper dimensions
            s1_tensor = torch.from_numpy(s1_processed).float().unsqueeze(0)  # [1, H, W]
            s2_tensor = torch.from_numpy(s2_processed).float().permute(2, 0, 1)  # [C, H, W]
            
            # Process mask
            mask = (mask > 127).astype(np.float32)  # Binary threshold
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]
            
            return s1_tensor, s2_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error processing sample at index {idx}: {str(e)}")
            # Return dummy data
            s1_channels = 1
            s2_channels = 4 if self.preprocessor.use_paper_config else 12
            s1_tensor = torch.zeros((s1_channels, 256, 256), dtype=torch.float32)
            s2_tensor = torch.zeros((s2_channels, 256, 256), dtype=torch.float32)
            mask_tensor = torch.zeros((1, 256, 256), dtype=torch.float32)
            return s1_tensor, s2_tensor, mask_tensor

class EnhancedTrainer:
    """Enhanced trainer with better metrics and loss functions"""
    
    def __init__(self, model, device, use_paper_loss=True, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Choose loss function
        if use_paper_loss:
            self.criterion = PaperL2Loss()
        else:
            self.criterion = nn.NLLLoss()
        
        # Use AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.best_f1 = 0.0
    
    def calculate_metrics(self, outputs, targets):
        """Calculate accuracy and F1 score"""
        with torch.no_grad():
            if self.model.use_log_softmax:
                # Convert log probabilities to class predictions
                _, predictions = torch.max(outputs, dim=1)
            else:
                # For softmax outputs
                _, predictions = torch.max(outputs, dim=1)
            
            # Convert targets to binary
            if targets.dim() == 4:  # [B, 1, H, W]
                targets = targets.squeeze(1)  # [B, H, W]
            targets = (targets > 0.5).long()
            
            # Calculate accuracy
            correct = (predictions == targets).float()
            accuracy = correct.mean().item() * 100
            
            # Calculate F1 score
            tp = ((predictions == 1) & (targets == 1)).float().sum()
            fp = ((predictions == 1) & (targets == 0)).float().sum()
            fn = ((predictions == 0) & (targets == 1)).float().sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            return accuracy, f1.item(), precision.item(), recall.item()
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (s1_imgs, s2_imgs, masks) in enumerate(progress_bar):
            # Move data to device
            s1_imgs = s1_imgs.to(self.device)
            s2_imgs = s2_imgs.to(self.device)
            masks = masks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(s1_imgs, s2_imgs)
            
            # Calculate loss
            if isinstance(self.criterion, PaperL2Loss):
                loss = self.criterion(outputs, masks)
            else:
                targets = (masks.squeeze(1) > 0.5).long()
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate metrics
            accuracy, f1, precision, recall = self.calculate_metrics(outputs, masks)
            
            # Update totals
            total_loss += loss.item()
            total_accuracy += accuracy
            total_f1 += f1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2f}%',
                'f1': f'{f1:.4f}'
            })
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        avg_f1 = total_f1 / len(train_loader)
        
        return avg_loss, avg_accuracy, avg_f1
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        
        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in tqdm(val_loader, desc="Validation"):
                # Move data to device
                s1_imgs = s1_imgs.to(self.device)
                s2_imgs = s2_imgs.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(s1_imgs, s2_imgs)
                
                # Calculate loss
                if isinstance(self.criterion, PaperL2Loss):
                    loss = self.criterion(outputs, masks)
                else:
                    targets = (masks.squeeze(1) > 0.5).long()
                    loss = self.criterion(outputs, targets)
                
                # Calculate metrics
                accuracy, f1, _, _ = self.calculate_metrics(outputs, masks)
                
                # Update totals
                total_loss += loss.item()
                total_accuracy += accuracy
                total_f1 += f1
        
        # Calculate averages
        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        avg_f1 = total_f1 / len(val_loader)
        
        return avg_loss, avg_accuracy, avg_f1

def main():
    """Main training function"""
    # Configuration
    use_paper_config = True  # Use paper specifications
    enhanced_complexity = True  # Use enhanced complexity for better accuracy
    use_paper_loss = True  # Use L2 loss as per paper
    num_epochs = 50
    batch_size = 4
    learning_rate = 5e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Data paths
    s1_dir = Path('Major Project - Copy/data/sentinel1')
    s2_dir = Path('Major Project - Copy/data/sentinel2')
    mask_dir = Path('Major Project - Copy/data/masks')
    
    # Get file paths
    s1_paths = sorted(list(s1_dir.glob('*.png')))
    s2_paths = []
    mask_paths = []
    
    # Match corresponding files
    for s1_path in s1_paths:
        s2_path = s2_dir / s1_path.name.replace('_s1_', '_s2_')
        mask_path = mask_dir / s1_path.name
        
        if s2_path.exists() and mask_path.exists():
            s2_paths.append(s2_path)
            mask_paths.append(mask_path)
        else:
            s1_paths.remove(s1_path)
    
    print(f"Found {len(s1_paths)} valid image triplets")
    
    # Split data
    s1_train, s1_temp, s2_train, s2_temp, mask_train, mask_temp = train_test_split(
        s1_paths, s2_paths, mask_paths, test_size=0.3, random_state=42
    )
    
    s1_val, s1_test, s2_val, s2_test, mask_val, mask_test = train_test_split(
        s1_temp, s2_temp, mask_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training samples: {len(s1_train)}")
    print(f"Validation samples: {len(s1_val)}")
    print(f"Test samples: {len(s1_test)}")
    
    # Create datasets
    train_dataset = EnhancedForestBurnedAreaDataset(
        s1_train, s2_train, mask_train, use_paper_config=use_paper_config
    )
    val_dataset = EnhancedForestBurnedAreaDataset(
        s1_val, s2_val, mask_val, use_paper_config=use_paper_config
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = DARU_Net(
        use_paper_config=use_paper_config,
        use_log_softmax=not use_paper_loss,  # Use log_softmax for NLL, regular softmax for L2
        enhanced_complexity=enhanced_complexity
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = EnhancedTrainer(model, device, use_paper_loss=use_paper_loss, learning_rate=learning_rate)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Configuration: Paper={use_paper_config}, Enhanced={enhanced_complexity}, L2 Loss={use_paper_loss}")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_f1': val_f1,
                'config': {
                    'use_paper_config': use_paper_config,
                    'enhanced_complexity': enhanced_complexity,
                    'use_paper_loss': use_paper_loss
                }
            }, 'results/best_enhanced_model.pth')
            print(f"Saved new best model with F1: {val_f1:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'val_accuracies': trainer.val_accuracies,
                    'val_f1_scores': trainer.val_f1_scores
                }
            }, f'results/checkpoint_epoch_{epoch+1}.pth')
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation F1 score: {trainer.best_f1:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(trainer.val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(trainer.val_f1_scores, label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1 Score')
    
    plt.tight_layout()
    plt.savefig('results/enhanced_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training metrics saved to results/enhanced_training_metrics.png")

if __name__ == '__main__':
    main()
