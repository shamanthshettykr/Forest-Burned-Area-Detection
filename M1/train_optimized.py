import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import time
from pathlib import Path
from darunet import DARU_Net
from data_preprocessing import DataPreprocessor
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Simple dataset for Sentinel-1 and Sentinel-2 images
class ForestBurnedAreaDataset(Dataset):
    def __init__(self, s1_paths, s2_paths, mask_paths, transform=None):
        self.s1_paths = s1_paths
        self.s2_paths = s2_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocessor = DataPreprocessor()

    def __len__(self):
        return len(self.s1_paths)

    def __getitem__(self, idx):
        try:
            # Load Sentinel-1 image
            s1_img = cv2.imread(str(self.s1_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if s1_img is None:
                print(f"Warning: Could not read Sentinel-1 image: {self.s1_paths[idx]}")
                # Create a dummy image
                s1_img = np.zeros((256, 256), dtype=np.uint8)

            s1_processed = self.preprocessor.preprocess_sentinel1(s1_img)

            # Load Sentinel-2 image
            s2_img = cv2.imread(str(self.s2_paths[idx]))
            if s2_img is None:
                print(f"Warning: Could not read Sentinel-2 image: {self.s2_paths[idx]}")
                # Create a dummy image with 3 channels
                s2_img = np.zeros((256, 256, 3), dtype=np.uint8)

            s2_processed = self.preprocessor.preprocess_sentinel2(s2_img)

            # Load mask
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask image: {self.mask_paths[idx]}")
                # Create a dummy mask
                mask = np.zeros((256, 256), dtype=np.uint8)

            # Convert to tensors
            s1_tensor = torch.from_numpy(s1_processed).float().unsqueeze(0) / 255.0

            # Handle different channel configurations for S2
            if len(s2_processed.shape) == 3:
                s2_tensor = torch.from_numpy(s2_processed.transpose(2, 0, 1)).float() / 255.0
            else:
                s2_tensor = torch.from_numpy(s2_processed).float().unsqueeze(0) / 255.0

            # Convert mask to class indices (0 for background, 1 for burned area)
            mask_tensor = torch.from_numpy((mask > 127).astype(np.int64))

            # Apply transforms if specified
            if self.transform:
                s1_tensor = self.transform(s1_tensor)
                s2_tensor = self.transform(s2_tensor)

            return s1_tensor, s2_tensor, mask_tensor

        except Exception as e:
            print(f"Error processing sample: {self.s1_paths[idx]}, {self.s2_paths[idx]}, {self.mask_paths[idx]}")
            print(f"Error details: {str(e)}")

            # Return dummy tensors in case of error
            s1_tensor = torch.zeros((1, 256, 256), dtype=torch.float32)
            s2_tensor = torch.zeros((4, 256, 256), dtype=torch.float32)  # 4 channels for S2
            mask_tensor = torch.zeros((256, 256), dtype=torch.int64)

            return s1_tensor, s2_tensor, mask_tensor

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] tensor of model predictions (log probabilities)
        # targets: [B, H, W] tensor of ground truth class indices

        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Get probabilities from log probabilities
        probs = torch.exp(inputs)

        # Calculate focal loss
        pt = probs * targets_one_hot + (1 - probs) * (1 - targets_one_hot)
        alpha_t = self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)

        loss = -focal_weight * inputs * targets_one_hot
        return loss.sum(dim=1).mean()

# Dice Loss for better boundary detection
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] tensor of model predictions (log probabilities)
        # targets: [B, H, W] tensor of ground truth class indices

        # Get probabilities from log probabilities
        probs = torch.exp(inputs)

        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Flatten predictions and targets
        probs_flat = probs.view(-1)
        targets_flat = targets_one_hot.view(-1)

        # Calculate Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice

# Simple training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    # Initialize metrics tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_dice_scores = []
    best_val_dice = 0.0

    # Track training time
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (s1_imgs, s2_imgs, masks) in enumerate(train_loader):
            # Move data to device
            s1_imgs = s1_imgs.to(device)
            s2_imgs = s2_imgs.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(s1_imgs, s2_imgs)
            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            train_loss += loss.item() * s1_imgs.size(0)

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        # Update learning rate
        scheduler.step()

        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total_pixels = 0
        dice_sum = 0.0

        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in val_loader:
                # Move data to device
                s1_imgs = s1_imgs.to(device)
                s2_imgs = s2_imgs.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(s1_imgs, s2_imgs)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * s1_imgs.size(0)

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == masks).sum().item()
                total_pixels += masks.numel()

                # Calculate Dice score
                predicted_mask = (predicted == 1)
                target_mask = (masks == 1)

                intersection = (predicted_mask & target_mask).sum().item()
                union = predicted_mask.sum().item() + target_mask.sum().item()

                dice = (2. * intersection + 1.0) / (union + 1.0)
                dice_sum += dice

        # Calculate average validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        val_accuracy = correct / total_pixels
        val_accuracies.append(val_accuracy)

        val_dice = dice_sum / len(val_loader)
        val_dice_scores.append(val_dice)

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with dice score: {val_dice:.4f}")

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Dice: {val_dice:.4f}')
        print('-' * 60)

    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f}s, Average time per epoch: {total_time/num_epochs:.2f}s")

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    return model, train_losses, val_losses, val_accuracies, val_dice_scores

# Main function
def main():
    # Set random seed
    set_seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set data directories
    s1_dir = 'data/sentinel1'
    s2_dir = 'data/sentinel2'
    mask_dir = 'data/masks'

    # Check if directories exist
    for directory in [s1_dir, s2_dir, mask_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

    # Get file paths
    s1_paths = list(Path(s1_dir).glob('*.png'))

    # If no files found, create dummy data for testing
    if len(s1_paths) == 0:
        print("No data found. Creating dummy data for testing...")
        # Create 10 dummy samples
        for i in range(10):
            # Create dummy Sentinel-1 image
            s1_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite(f"{s1_dir}/dummy_s1_{i}.png", s1_img)

            # Create dummy Sentinel-2 image with 4 channels (RGB + NIR)
            s2_img = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
            # OpenCV can't save 4-channel images directly, so we'll save as RGB
            cv2.imwrite(f"{s2_dir}/dummy_s2_{i}.png", s2_img[:, :, :3])

            # Create dummy mask
            mask = np.zeros((256, 256), dtype=np.uint8)
            # Add some random burned areas
            mask[np.random.randint(0, 256, 100), np.random.randint(0, 256, 100)] = 255
            cv2.imwrite(f"{mask_dir}/dummy_mask_{i}.png", mask)

        # Update file paths
        s1_paths = list(Path(s1_dir).glob('*.png'))

    # Get corresponding S2 and mask paths
    s2_paths = []
    mask_paths = []

    for s1_path in s1_paths:
        s1_name = s1_path.name
        # Try different naming patterns
        s2_name = s1_name.replace('_s1_', '_s2_')
        if not os.path.exists(Path(s2_dir) / s2_name):
            s2_name = s1_name  # Try same name

        mask_name = s1_name
        if not os.path.exists(Path(mask_dir) / mask_name):
            mask_name = s1_name.replace('_s1_', '_mask_')

        s2_paths.append(Path(s2_dir) / s2_name)
        mask_paths.append(Path(mask_dir) / mask_name)

    print(f"Found {len(s1_paths)} image pairs")

    # Split data into train, validation, and test sets (80/10/10 split)
    # First split: 80% train, 20% temp (which will be split into 10% val, 10% test)
    s1_train, s1_temp, s2_train, s2_temp, mask_train, mask_temp = train_test_split(
        s1_paths, s2_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Second split: Split the 20% temp into 10% val and 10% test
    s1_val, s1_test, s2_val, s2_test, mask_val, mask_test = train_test_split(
        s1_temp, s2_temp, mask_temp, test_size=0.5, random_state=42
    )

    # Create datasets
    print("Creating datasets...")
    print(f"Data split - Train: {len(s1_train)} ({len(s1_train)/len(s1_paths)*100:.1f}%), "
          f"Val: {len(s1_val)} ({len(s1_val)/len(s1_paths)*100:.1f}%), "
          f"Test: {len(s1_test)} ({len(s1_test)/len(s1_paths)*100:.1f}%)")

    train_dataset = ForestBurnedAreaDataset(s1_train, s2_train, mask_train)
    val_dataset = ForestBurnedAreaDataset(s1_val, s2_val, mask_val)
    test_dataset = ForestBurnedAreaDataset(s1_test, s2_test, mask_test)

    # Initialize model
    model = DARU_Net().to(device)

    # Use a small batch size for systems with limited memory
    batch_size = 2
    print(f"Using batch size: {batch_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # No parallel loading to save memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # No parallel loading to save memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # No parallel loading to save memory
    )

    # Initialize loss function
    criterion = CombinedLoss(focal_weight=0.7, dice_weight=0.3)

    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Initialize learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Train model
    print("Starting training...")
    # Reduce number of epochs for testing on systems with limited resources
    num_epochs = 5 if not torch.cuda.is_available() else 50
    print(f"Training for {num_epochs} epochs")

    model, train_losses, val_losses, val_accuracies, val_dice_scores = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=num_epochs
    )

    # Plot training and validation metrics
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(val_dice_scores, label='Val Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.title('Validation Dice Score')

    plt.tight_layout()
    plt.savefig('training_metrics.png')

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_dice_scores': val_dice_scores
    }, 'final_model.pth')

    print("\nTraining complete! Final model saved to final_model.pth")

if __name__ == '__main__':
    main()
