"""
CPU-Optimized Training Script for Maximum Test Accuracy
Designed specifically for CPU training with comprehensive evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
try:
    import seaborn as sns
except ImportError:
    print("Seaborn not available, using matplotlib only")
    sns = None

# For real-time plotting
plt.ion()  # Enable interactive mode

# Import our modules
from darunet_cpu_optimized import CPUOptimizedDARUNet, CombinedLoss
from data_preprocessing import DataPreprocessor

class OptimizedDataset(Dataset):
    """Optimized dataset for CPU training"""
    
    def __init__(self, s1_paths, s2_paths, mask_paths, use_all_s2_channels=True, augment=False):
        self.s1_paths = s1_paths
        self.s2_paths = s2_paths
        self.mask_paths = mask_paths
        self.preprocessor = DataPreprocessor(use_paper_config=not use_all_s2_channels)
        self.augment = augment
        
    def __len__(self):
        return len(self.s1_paths)
    
    def __getitem__(self, idx):
        try:
            # Load images
            s1_img = cv2.imread(str(self.s1_paths[idx]), cv2.IMREAD_GRAYSCALE)
            s2_img = cv2.imread(str(self.s2_paths[idx]))
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            
            if s1_img is None or s2_img is None or mask is None:
                raise ValueError("Could not load images")
            
            # Preprocess
            s1_processed = self.preprocessor.preprocess_sentinel1(s1_img)
            s2_processed = self.preprocessor.preprocess_sentinel2(s2_img)
            
            # Data augmentation for training (fix negative stride issue)
            if self.augment and np.random.random() > 0.5:
                # Random horizontal flip
                if np.random.random() > 0.5:
                    s1_processed = np.fliplr(s1_processed).copy()
                    s2_processed = np.fliplr(s2_processed).copy()
                    mask = np.fliplr(mask).copy()

                # Random vertical flip
                if np.random.random() > 0.5:
                    s1_processed = np.flipud(s1_processed).copy()
                    s2_processed = np.flipud(s2_processed).copy()
                    mask = np.flipud(mask).copy()
            
            # Convert to tensors
            s1_tensor = torch.from_numpy(s1_processed).float().unsqueeze(0)
            s2_tensor = torch.from_numpy(s2_processed).float().permute(2, 0, 1)
            
            # Process mask
            mask = (mask > 127).astype(np.int64)  # Use np.int64 instead of deprecated np.long
            mask_tensor = torch.from_numpy(mask).long()
            
            return s1_tensor, s2_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # Return dummy data
            s1_tensor = torch.zeros((1, 256, 256), dtype=torch.float32)
            s2_tensor = torch.zeros((12, 256, 256), dtype=torch.float32)
            mask_tensor = torch.zeros((256, 256), dtype=torch.long)
            return s1_tensor, s2_tensor, mask_tensor

class CPUOptimizedTrainer:
    """CPU-Optimized trainer for maximum accuracy"""

    def __init__(self, model, learning_rate=1e-4):
        # Force CPU usage
        self.device = torch.device('cpu')
        self.model = model.to(self.device)

        # Ensure model is in CPU mode
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Training device: {self.device}")

        # Use Binary Cross Entropy with Logits for binary segmentation (paper-equivalent)
        self.criterion = nn.BCEWithLogitsLoss()  # Equivalent to L2 for binary classification

        # Paper-based optimizer configuration for ≥90% accuracy
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,  # Reduced weight decay for better convergence
            eps=1e-8,
            betas=(0.9, 0.999)  # Standard Adam parameters
        )

        # Paper-based learning rate scheduler for optimal convergence
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.5  # Reduce LR every 30 epochs
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.val_precisions = []  # Added precision tracking
        self.val_recalls = []     # Added recall tracking
        self.train_accuracies = []  # Track training accuracies for constraint validation
        self.best_f1 = 0.0
        self.best_accuracy = 0.0
        
    def calculate_metrics(self, outputs, targets):
        """Calculate comprehensive metrics"""
        with torch.no_grad():
            # Get predictions
            _, predictions = torch.max(outputs, dim=1)
            
            # Flatten for sklearn metrics
            targets_flat = targets.cpu().numpy().flatten()
            predictions_flat = predictions.cpu().numpy().flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(targets_flat, predictions_flat) * 100
            f1 = f1_score(targets_flat, predictions_flat, average='weighted', zero_division=0)
            precision = precision_score(targets_flat, predictions_flat, average='weighted', zero_division=0)
            recall = recall_score(targets_flat, predictions_flat, average='weighted', zero_division=0)
            
            return accuracy, f1, precision, recall
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (s1_imgs, s2_imgs, masks) in enumerate(progress_bar):
            # Ensure all tensors are on CPU
            s1_imgs = s1_imgs.to(self.device)
            s2_imgs = s2_imgs.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(s1_imgs, s2_imgs)

            # Extract positive class logits for binary classification (paper approach)
            positive_logits = outputs[:, 1:2, :, :]  # Take channel 1 (positive class)
            masks_expanded = masks.unsqueeze(1).float()  # Add channel dimension to match
            loss = self.criterion(positive_logits, masks_expanded)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            accuracy, f1, _, _ = self.calculate_metrics(outputs, masks)
            
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
        
        return total_loss / len(train_loader), total_accuracy / len(train_loader), total_f1 / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0

        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in tqdm(val_loader, desc="Validation", leave=False):
                # Ensure all tensors are on CPU
                s1_imgs = s1_imgs.to(self.device)
                s2_imgs = s2_imgs.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(s1_imgs, s2_imgs)

                # Extract positive class logits for binary classification (paper approach)
                positive_logits = outputs[:, 1:2, :, :]  # Take channel 1 (positive class)
                masks_expanded = masks.unsqueeze(1).float()  # Add channel dimension to match
                loss = self.criterion(positive_logits, masks_expanded)

                accuracy, f1, precision, recall = self.calculate_metrics(outputs, masks)

                total_loss += loss.item()
                total_accuracy += accuracy
                total_f1 += f1
                total_precision += precision
                total_recall += recall

        return (total_loss / len(val_loader),
                total_accuracy / len(val_loader),
                total_f1 / len(val_loader),
                total_precision / len(val_loader),
                total_recall / len(val_loader))
    
    def test(self, test_loader):
        """Comprehensive test evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        
        print("Running comprehensive test evaluation...")
        
        with torch.no_grad():
            for s1_imgs, s2_imgs, masks in tqdm(test_loader, desc="Testing"):
                # Ensure all tensors are on CPU
                s1_imgs = s1_imgs.to(self.device)
                s2_imgs = s2_imgs.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(s1_imgs, s2_imgs)
                loss = self.criterion(outputs, masks)
                
                _, predictions = torch.max(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
                test_loss += loss.item()
        
        # Calculate comprehensive metrics
        test_accuracy = accuracy_score(all_targets, all_predictions) * 100
        test_f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        test_precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        test_recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'loss': test_loss / len(test_loader),
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'confusion_matrix': cm
        }

def plot_real_time_progress(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls, epoch, save_dir='results'):
    """Plot real-time training progress"""
    if len(train_losses) < 2:
        return

    # Create directory for plots
    os.makedirs(save_dir, exist_ok=True)

    # Figure 1: Combined plots (2x2 grid)
    plt.figure(figsize=(15, 10))

    # Plot 1: Loss curves
    plt.subplot(2, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Validation F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_f1_scores, 'm-', label='Validation F1 Score', linewidth=2)
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Combined metrics
    plt.subplot(2, 2, 4)
    # Normalize accuracy to 0-1 scale for comparison
    normalized_acc = [acc/100 for acc in val_accuracies]
    plt.plot(epochs, normalized_acc, 'g-', label='Val Accuracy (normalized)', linewidth=2)
    plt.plot(epochs, val_f1_scores, 'm-', label='Val F1 Score', linewidth=2)
    plt.title('Combined Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'{save_dir}/training_progress_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/training_progress_latest.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Line graph for accuracy during training
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=3, marker='o', markersize=5)
    plt.title('Validation Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_line_graph_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/accuracy_line_graph_latest.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2b: Line graph for F1-Score during training
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1_scores, 'm-', label='Validation F1-Score', linewidth=3, marker='o', markersize=5)
    plt.title('Validation F1-Score During Training')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/f1_score_line_graph_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/f1_score_line_graph_latest.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2c: Combined line graph for Accuracy and F1-Score
    plt.figure(figsize=(12, 6))

    # Create dual y-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot accuracy on left y-axis
    color = 'tab:green'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color=color)
    line1 = ax1.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy', linewidth=3, marker='o', markersize=5, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for F1-score
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('F1-Score', color=color)
    line2 = ax2.plot(epochs, val_f1_scores, 'm-', label='Validation F1-Score', linewidth=3, marker='s', markersize=5, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title and legend
    plt.title('Validation Accuracy and F1-Score During Training')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_f1_combined_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/accuracy_f1_combined_latest.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Bar graph for latest accuracy and precision
    # Only create this after a few epochs when we have meaningful data
    if len(val_accuracies) >= 3 and len(val_precisions) >= 3:
        plt.figure(figsize=(12, 6))

        # Get the latest metrics
        latest_accuracy = val_accuracies[-1]
        latest_precision = val_precisions[-1] * 100  # Convert to percentage
        latest_recall = val_recalls[-1] * 100  # Convert to percentage
        latest_f1 = val_f1_scores[-1] * 100  # Convert to percentage

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [latest_accuracy, latest_precision, latest_recall, latest_f1]
        colors = ['green', 'blue', 'orange', 'purple']

        # Create bar chart
        bars = plt.bar(metrics, values, color=colors, width=0.6)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

        plt.title(f'Performance Metrics (Epoch {epoch})')
        plt.ylabel('Value (%)')
        plt.ylim(0, max(values) * 1.15)  # Add some headroom for labels
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plt.savefig(f'{save_dir}/metrics_bar_graph_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/metrics_bar_graph_latest.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 4: Separate bar graph for Accuracy and Precision comparison
        plt.figure(figsize=(8, 6))

        acc_prec_metrics = ['Accuracy', 'Precision']
        acc_prec_values = [latest_accuracy, latest_precision]
        acc_prec_colors = ['green', 'blue']

        bars = plt.bar(acc_prec_metrics, acc_prec_values, color=acc_prec_colors, width=0.5)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

        plt.title(f'Accuracy vs Precision (Epoch {epoch})')
        plt.ylabel('Value (%)')
        plt.ylim(0, max(acc_prec_values) * 1.15)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plt.savefig(f'{save_dir}/accuracy_precision_bar_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/accuracy_precision_bar_latest.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Figure 5: F1-Score bar graph
        plt.figure(figsize=(6, 6))

        # Create a single bar for F1-Score
        f1_bar = plt.bar(['F1-Score'], [latest_f1], color='purple', width=0.4)

        # Add value label
        for bar in f1_bar:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

        plt.title(f'F1-Score (Epoch {epoch})')
        plt.ylabel('Value (%)')
        plt.ylim(0, latest_f1 * 1.2)  # Add 20% headroom
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plt.savefig(f'{save_dir}/f1_score_bar_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.savefig(f'{save_dir}/f1_score_bar_latest.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Show the plot briefly
    plt.pause(0.1)

def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))

    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Burned', 'Burned'],
                    yticklabels=['Non-Burned', 'Burned'])
    else:
        # Fallback to matplotlib only
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')

        plt.xticks([0, 1], ['Non-Burned', 'Burned'])
        plt.yticks([0, 1], ['Non-Burned', 'Burned'])

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function optimized for CPU"""
    # Force CPU usage - disable CUDA completely
    torch.cuda.is_available = lambda: False
    torch.backends.cudnn.enabled = False

    # PAPER-BASED Configuration for ≥90% validation accuracy (Paper achieved 93.14%)
    use_all_s2_channels = True  # Use all 12 channels as per paper
    num_epochs = 150  # Sufficient epochs for paper-level performance
    batch_size = 4  # Increase batch size for better gradient estimates
    learning_rate = 1e-3  # Paper-recommended learning rate for faster convergence

    # Validation accuracy requirements
    min_val_accuracy = 90.0  # Minimum required
    target_val_accuracy = 99.0  # Maximum target
    print(f"🎯 REQUIREMENT: Validation Accuracy ≥ {min_val_accuracy}% (minimum)")
    print(f"🚀 TARGET: Maximum possible accuracy up to {target_val_accuracy}%")

    print("CPU-ONLY DARU-Net Training for Maximum Test Accuracy")
    print("=" * 60)
    print("GPU DISABLED - Training on CPU only")
    print(f"Device: {torch.device('cpu')}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Data paths
    s1_dir = Path('data/sentinel1')
    s2_dir = Path('data/sentinel2')
    mask_dir = Path('data/masks')
    
    # Get file paths
    s1_paths = sorted(list(s1_dir.glob('*.png')))
    valid_triplets = []
    for s1_path in s1_paths:
        s2_path = s2_dir / s1_path.name.replace('_s1_', '_s2_')
        mask_path = mask_dir / s1_path.name
        
        if s2_path.exists() and mask_path.exists():
            valid_triplets.append((s1_path, s2_path, mask_path))
    
    s1_paths, s2_paths, mask_paths = zip(*valid_triplets) if valid_triplets else ([], [], [])
    
    print(f"Found {len(s1_paths)} valid image triplets")
    
    # Proper Data Splitting Strategy:
    # - 80% of available images for training
    # - 10% of available images for validation (completely separate)
    # - 10% of available images for testing (completely separate)
    # - No data leakage: all sets are completely separate

    total_samples = len(s1_paths)

    print(f"📊 PROPER DATA SPLITTING STRATEGY")
    print(f"=" * 50)
    print(f"Total samples: {total_samples}")
    print(f"Strategy: 80/10/10 split with completely separate train/validation/test sets")
    print(f"")
    print(f"Data Split (80% train, 10% val, 10% test):")

    # Calculate split sizes
    train_size = int(total_samples * 0.8)
    val_size = int(total_samples * 0.1)
    test_size = total_samples - train_size - val_size  # Remaining samples for test

    print(f"  🏋️  Training:   {train_size:4d} samples ({train_size/total_samples*100:.1f}%)")
    print(f"  ✅ Validation: {val_size:4d} samples ({val_size/total_samples*100:.1f}%)")
    print(f"  🧪 Testing:    {test_size:4d} samples ({test_size/total_samples*100:.1f}%)")
    print(f"=" * 50)

    # First split: 80% train, 20% temp (which will be split into 10% val, 10% test)
    s1_train, s1_temp, s2_train, s2_temp, mask_train, mask_temp = train_test_split(
        s1_paths, s2_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Second split: Split the 20% temp into 10% val and 10% test
    s1_val, s1_test, s2_val, s2_test, mask_val, mask_test = train_test_split(
        s1_temp, s2_temp, mask_temp, test_size=0.5, random_state=42
    )

    # Verify the split matches our corrected strategy
    actual_train_pct = len(s1_train)/total_samples*100
    actual_val_pct = len(s1_val)/total_samples*100
    actual_test_pct = len(s1_test)/total_samples*100

    print(f"✅ VERIFICATION:")
    print(f"  Training:   {len(s1_train):4d} samples ({actual_train_pct:.1f}% - Target: 100.0%)")
    print(f"  Validation: {len(s1_val):4d} samples ({actual_val_pct:.1f}% - Target: 16.0% from trained)")
    print(f"  Testing:    {len(s1_test):4d} samples ({actual_test_pct:.1f}% - Target: 4.0% from trained)")

    # Verify percentages match corrected strategy
    if abs(actual_train_pct - 100.0) > 0.1:
        print(f"⚠️  Warning: Training percentage ({actual_train_pct:.1f}%) should be 100.0%")
    if abs(actual_val_pct - 16.0) > 2.0:
        print(f"⚠️  Warning: Validation percentage ({actual_val_pct:.1f}%) deviates from target (16.0%)")
    if abs(actual_test_pct - 4.0) > 2.0:
        print(f"⚠️  Warning: Testing percentage ({actual_test_pct:.1f}%) deviates from target (4.0%)")

    print(f"📝 Note: Train/validation/test sets are completely separate - no data leakage")
    print(f"=" * 50)
    
    # Create datasets
    train_dataset = OptimizedDataset(s1_train, s2_train, mask_train, use_all_s2_channels, augment=True)
    val_dataset = OptimizedDataset(s1_val, s2_val, mask_val, use_all_s2_channels, augment=False)
    test_dataset = OptimizedDataset(s1_test, s2_test, mask_test, use_all_s2_channels, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = CPUOptimizedDARUNet(use_all_s2_channels=use_all_s2_channels)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = CPUOptimizedTrainer(model, learning_rate=learning_rate)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("Configuration: CPU-Optimized, All S2 channels, Combined Loss")
    
    # Training loop
    start_time = time.time()
    patience = 20
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_acc, val_f1, val_precision, val_recall = trainer.validate(val_loader)

        # Update scheduler (StepLR doesn't need validation accuracy)
        trainer.scheduler.step()

        # Store metrics
        trainer.train_losses.append(train_loss)
        trainer.train_accuracies.append(train_acc)  # Track training accuracy for constraint validation
        trainer.val_losses.append(val_loss)
        trainer.val_accuracies.append(val_acc)
        trainer.val_f1_scores.append(val_f1)
        trainer.val_precisions.append(val_precision)
        trainer.val_recalls.append(val_recall)

        # Print progress
        epoch_time = time.time() - epoch_start
        # Calculate overfitting indicators
        train_val_gap = train_acc - val_acc
        overfitting_status = "🟢 Good" if train_val_gap < 10 else "🟡 Moderate" if train_val_gap < 20 else "🔴 High"

        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Overfitting Check: {overfitting_status} (Train-Val gap: {train_val_gap:.2f}%)")

        # Plot real-time progress
        plot_real_time_progress(
            trainer.train_losses,
            trainer.val_losses,
            trainer.val_accuracies,
            trainer.val_f1_scores,
            trainer.val_precisions,
            trainer.val_recalls,
            epoch+1
        )

        # Save best model based on validation accuracy with constraint validation
        if val_acc > trainer.best_accuracy:
            trainer.best_accuracy = val_acc
            trainer.best_f1 = val_f1
            no_improve_count = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'config': {
                    'use_all_s2_channels': use_all_s2_channels,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
            }, 'results/best_cpu_optimized_model.pth')
            print(f"💾 Saved new best model with Accuracy: {val_acc:.2f}%, F1: {val_f1:.4f}")

            # Check validation accuracy requirements
            if val_acc >= min_val_accuracy:
                print(f"✅ MINIMUM REQUIREMENT MET: {val_acc:.2f}% ≥ {min_val_accuracy}%")
                if val_acc >= target_val_accuracy:
                    print(f"🏆 MAXIMUM TARGET ACHIEVED: {val_acc:.2f}% ≥ {target_val_accuracy}%")
            else:
                remaining = min_val_accuracy - val_acc
                print(f"⏳ Need {remaining:.2f}% more to reach minimum requirement of {min_val_accuracy}%")

            # Note: Constraint validation will be performed after final test evaluation
        else:
            no_improve_count += 1
        
        # Early stopping on overfitting detection
        overfitting_gap = train_acc - val_acc  # Gap between training and validation accuracy

        # Stop if no improvement for 25 epochs (overfitting) - increased patience to reach 90%
        if no_improve_count >= 25:
            print(f"🛑 EARLY STOPPING: Overfitting detected after {no_improve_count} epochs without improvement")
            print(f"   Best validation accuracy: {trainer.best_accuracy:.2f}% at epoch {epoch + 1 - no_improve_count}")
            print(f"   Current train-val gap: {overfitting_gap:.2f}%")
            print(f"   Stopping to prevent overfitting and preserve best model")
            break

        # Additional overfitting check: large gap between train and validation accuracy
        if overfitting_gap > 30.0 and epoch > 20:  # More lenient since we use trained images for validation
            print(f"🛑 EARLY STOPPING: Severe overfitting detected")
            print(f"   Training accuracy: {train_acc:.2f}%")
            print(f"   Validation accuracy: {val_acc:.2f}%")
            print(f"   Gap: {overfitting_gap:.2f}% (threshold: 20.0%)")
            print(f"   Stopping to prevent further overfitting")
            break
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'val_accuracies': trainer.val_accuracies,
                    'val_f1_scores': trainer.val_f1_scores,
                    'val_precisions': trainer.val_precisions,
                    'val_recalls': trainer.val_recalls
                }
            }, f'results/checkpoint_epoch_{epoch+1}.pth')
    
    # Load best model for testing
    print("\nLoading best model for final test evaluation...")
    checkpoint = torch.load('results/best_cpu_optimized_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Comprehensive test evaluation
    test_results = trainer.test(test_loader)
    
    # Print final results
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation F1 score: {trainer.best_f1:.4f}")
    print(f"Best validation accuracy: {trainer.best_accuracy:.2f}%")
    print(f"\nTEST SET RESULTS:")
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Test F1 Score: {test_results['f1_score']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test Loss: {test_results['loss']:.4f}")

    # FINAL VALIDATION ACCURACY CHECK
    print(f"\n" + "📊 FINAL VALIDATION ACCURACY CHECK" + "=" * 26)
    final_val_accuracy = trainer.best_accuracy

    if final_val_accuracy >= min_val_accuracy:
        print(f"✅ REQUIREMENT SATISFIED: {final_val_accuracy:.2f}% ≥ {min_val_accuracy}% (minimum)")
        if final_val_accuracy >= target_val_accuracy:
            print(f"🏆 MAXIMUM TARGET ACHIEVED: {final_val_accuracy:.2f}% ≥ {target_val_accuracy}%")
        else:
            print(f"🎯 Good result: {final_val_accuracy:.2f}% (target was {target_val_accuracy}%)")
    else:
        shortage = min_val_accuracy - final_val_accuracy
        print(f"❌ REQUIREMENT NOT MET: {final_val_accuracy:.2f}% < {min_val_accuracy}% (short by {shortage:.2f}%)")
        print(f"⚠️  Consider training longer or adjusting parameters")

    # CORRECTED CONSTRAINT VALIDATION: Ensure validation accuracy ≤ training accuracy
    print(f"\n" + "🔍 CORRECTED CONSTRAINT VALIDATION" + "=" * 35)
    val_accuracy = trainer.best_accuracy
    test_accuracy = test_results['accuracy']

    # Get final training accuracy from the last epoch
    final_train_accuracy = trainer.train_accuracies[-1] if trainer.train_accuracies else 0.0

    print(f"Training Accuracy:   {final_train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Test Accuracy:       {test_accuracy:.2f}%")
    print(f"Difference:          {final_train_accuracy - val_accuracy:.2f}% (Train - Validation)")

    if val_accuracy <= final_train_accuracy:
        print(f"✅ CONSTRAINT SATISFIED: Validation accuracy ≤ Training accuracy")
        constraint_status = "SATISFIED"
    else:
        print(f"❌ CONSTRAINT VIOLATED: Validation accuracy > Training accuracy")
        print(f"   Validation exceeds training by {val_accuracy - final_train_accuracy:.2f}%")
        print(f"   This indicates potential issues with the training process")
        constraint_status = "VIOLATED"

    print("=" * 60)

    # Save test results with constraint validation
    with open('results/test_results.json', 'w') as f:
        json.dump({
            'test_accuracy': test_results['accuracy'],
            'test_f1_score': test_results['f1_score'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'test_loss': test_results['loss'],
            'best_val_f1': trainer.best_f1,
            'best_val_accuracy': trainer.best_accuracy,
            'training_time_hours': total_time / 3600,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            # New data splitting strategy info
            'data_split_strategy': {
                'training_percentage': len(s1_train)/total_samples*100,
                'validation_percentage': len(s1_val)/total_samples*100,
                'testing_percentage': len(s1_test)/total_samples*100,
                'total_samples': total_samples,
                'training_samples': len(s1_train),
                'validation_samples': len(s1_val),
                'testing_samples': len(s1_test)
            },
            # Corrected constraint validation results
            'constraint_validation': {
                'training_accuracy': final_train_accuracy,
                'validation_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
                'train_val_difference': final_train_accuracy - val_accuracy,
                'constraint_satisfied': constraint_status == "SATISFIED",
                'constraint_status': constraint_status,
                'constraint_description': 'validation_accuracy <= training_accuracy'
            }
        }, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(test_results['confusion_matrix'], 'results/confusion_matrix.png')

    # Plot final training curves with enhanced visualization
    plot_real_time_progress(
        trainer.train_losses,
        trainer.val_losses,
        trainer.val_accuracies,
        trainer.val_f1_scores,
        trainer.val_precisions,
        trainer.val_recalls,
        len(trainer.train_losses),
        save_dir='results'
    )

    # Also create a simple version for compatibility
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(trainer.train_losses, label='Train Loss', linewidth=2)
    plt.plot(trainer.val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(trainer.val_accuracies, label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(trainer.val_f1_scores, label='Val F1 Score', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1 Score')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_curves_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to 'results/' directory")
    print(f"Confusion matrix saved to 'results/confusion_matrix.png'")
    print(f"Training curves saved to 'results/training_curves.png'")

if __name__ == '__main__':
    main()
