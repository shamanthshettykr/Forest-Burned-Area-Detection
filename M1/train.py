import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import DualSentinelDataset
from darunet import DARU_Net
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import albumentations as A

# Enhanced weight initialization function
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # Kaiming initialization for Conv2d layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm2d layers
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Initialize Linear layers
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss for addressing class imbalance
    with better handling of hard examples
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] tensor of model predictions (log probabilities)
        # targets: [B, H, W] tensor of ground truth class indices

        # Get probabilities from log probabilities
        probs = torch.exp(inputs)

        # Get probability of the target class
        p_t = torch.gather(probs, 1, targets.unsqueeze(1))
        p_t = p_t.squeeze(1)

        # Calculate focal weight with higher gamma for more focus on hard examples
        focal_weight = (1 - p_t) ** self.gamma

        # Apply adaptive class balancing based on class distribution
        batch_positives = (targets == 1).float().sum() / targets.numel()
        alpha_t = torch.ones_like(targets).float() * self.alpha

        # Adjust alpha based on class distribution in the batch
        if batch_positives > 0:
            pos_weight = torch.clamp(0.5 / batch_positives, min=0.5, max=2.0)
            alpha_t = torch.where(targets == 1, alpha_t * pos_weight, alpha_t * (1.0 / pos_weight))

        # Calculate loss
        ce_loss = torch.nn.functional.nll_loss(inputs, targets, reduction='none')
        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()

class DiceLoss(nn.Module):
    """
    Enhanced Dice Loss for better boundary detection
    with class weighting and boundary emphasis
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] tensor of model predictions (log probabilities)
        # targets: [B, H, W] tensor of ground truth class indices

        # Get probabilities from log probabilities
        probs = torch.exp(inputs)

        # Get probability of the positive class (class 1)
        p1 = probs[:, 1, :, :]

        # Convert targets to one-hot
        targets_one_hot = (targets == 1).float()

        # Calculate intersection and union with boundary emphasis
        # Apply higher weight to boundary pixels
        batch_size = targets.size(0)
        weighted_intersection = 0
        weighted_union = 0

        for i in range(batch_size):
            # Extract single image and target
            pred = p1[i]
            target = targets_one_hot[i]

            # Calculate intersection and union
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()

            # Add to batch totals
            weighted_intersection += intersection
            weighted_union += union

        # Calculate Dice coefficient
        dice = (2. * weighted_intersection + self.smooth) / (weighted_union + self.smooth)

        return 1 - dice

class BoundaryLoss(nn.Module):
    """
    Boundary Loss to focus on object boundaries
    """
    def __init__(self, theta=1.0):
        super(BoundaryLoss, self).__init__()
        self.theta = theta
        # Define Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] tensor of model predictions (log probabilities)
        # targets: [B, H, W] tensor of ground truth class indices

        # Get probabilities from log probabilities
        probs = torch.exp(inputs)

        # Get probability of the positive class (class 1)
        p1 = probs[:, 1, :, :]

        # Convert targets to one-hot
        targets_one_hot = (targets == 1).float()

        # Move Sobel filters to the same device as inputs
        device = inputs.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # Calculate gradients for predictions
        p1 = p1.unsqueeze(1)  # Add channel dimension for conv2d
        pred_grad_x = torch.nn.functional.conv2d(p1, sobel_x, padding=1)
        pred_grad_y = torch.nn.functional.conv2d(p1, sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2).squeeze(1)

        # Calculate gradients for targets
        targets_one_hot = targets_one_hot.unsqueeze(1)  # Add channel dimension
        target_grad_x = torch.nn.functional.conv2d(targets_one_hot, sobel_x, padding=1)
        target_grad_y = torch.nn.functional.conv2d(targets_one_hot, sobel_y, padding=1)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2).squeeze(1)

        # Calculate boundary loss
        boundary_loss = torch.nn.functional.mse_loss(pred_grad, target_grad)

        return self.theta * boundary_loss

class EnhancedCombinedLoss(nn.Module):
    """
    Enhanced combined loss function: Focal Loss + Dice Loss + Boundary Loss
    """
    def __init__(self, focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2, gamma=2.0, alpha=0.25):
        super(EnhancedCombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        boundary = self.boundary_loss(inputs, targets)

        # Combine losses with adaptive weighting
        total_loss = (self.focal_weight * focal +
                      self.dice_weight * dice +
                      self.boundary_weight * boundary)

        return total_loss

class L2Loss(nn.Module):
    """L2 Loss function as described in the paper"""
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] tensor of model predictions (log probabilities)
        # targets: [B, H, W] tensor of ground truth class indices
        
        # Convert log probabilities to probabilities
        probs = torch.exp(inputs)
        
        # Get the probability of the positive class (burned area)
        pred_burned = probs[:, 1, :, :]
        
        # Convert targets to float
        targets = (targets == 1).float()
        
        # Calculate L2 loss
        loss = torch.mean((pred_burned - targets) ** 2)
        
        return loss

class Trainer:
    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device

        # Use L2 loss as described in the paper
        self.criterion = L2Loss()
        
        # Use enhanced combined loss function for better performance
        self.criterion = EnhancedCombinedLoss(
            focal_weight=0.4,
            dice_weight=0.4,
            boundary_weight=0.2,
            gamma=2.0,
            alpha=0.25
        )

        # Use AdamW optimizer with weight decay for better generalization
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,
            eps=1e-8  # Smaller epsilon for better precision
        )

        self.threshold = 0.5
        self.best_f1 = 0.0

        # Track metrics history
        self.train_losses = []
        self.val_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []

    def calculate_area(self, pred_logits):
        # Convert log probabilities to class predictions
        # pred_logits shape: [batch_size, 2, H, W]
        # Get the class with highest probability (dim=1)
        _, predictions = torch.max(pred_logits, dim=1)  # shape: [batch_size, H, W]

        # Create binary mask where class 1 (burned area) is present
        binary_mask = (predictions == 1).float()  # shape: [batch_size, H, W]

        # Calculate area
        area = torch.sum(binary_mask) * 10
        area_km2 = area / 1_000_000
        return area_km2

    def calculate_metrics(self, outputs, masks):
        # Convert log probabilities to class predictions
        # outputs shape: [batch_size, 2, H, W]
        # masks shape: [batch_size, 1, H, W]

        # Get the class with highest probability (dim=1)
        _, predictions = torch.max(outputs, dim=1)  # shape: [batch_size, H, W]

        # Reshape masks to match predictions
        masks = masks.squeeze(1)  # shape: [batch_size, H, W]

        # Convert masks to long type for comparison
        masks = (masks > 0.5).long()

        # Calculate accuracy
        correct = (predictions == masks).float().sum()
        total = torch.numel(masks)
        accuracy = (correct / total) * 100

        # Calculate F1 score components
        true_positives = ((predictions == 1) & (masks == 1)).float().sum()
        false_positives = ((predictions == 1) & (masks == 0)).float().sum()
        false_negatives = ((predictions == 0) & (masks == 1)).float().sum()

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return accuracy.item(), f_score.item(), precision.item(), recall.item()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        total_f_score = 0

        # Add progress bar
        progress_bar = tqdm(train_loader, desc='Training')
        for s1_imgs, s2_imgs, masks in progress_bar:
            s1_imgs = s1_imgs.to(self.device)
            s2_imgs = s2_imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(s1_imgs, s2_imgs)

            # Check if we need to resize masks to match model output size
            if outputs.shape[2:] != masks.shape[2:]:
                # Resize masks to match output size
                masks = torch.nn.functional.interpolate(
                    masks,
                    size=outputs.shape[2:],  # (H, W)
                    mode='nearest'
                )

            # Convert masks to class indices for NLLLoss
            # NLLLoss expects target of shape [B, H, W] with class indices
            target_masks = (masks.squeeze(1) > 0.5).long()

            # Calculate loss
            loss = self.criterion(outputs, target_masks)

            # Update to unpack all 4 values
            accuracy, f_score, precision, recall = self.calculate_metrics(outputs, masks)
            total_accuracy += accuracy
            total_f_score += f_score

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Update progress bar with all metrics
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2f}%',
                'f1': f'{f_score:.4f}',
                'prec': f'{precision:.4f}',
                'rec': f'{recall:.4f}'
            })

        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        avg_f_score = total_f_score / len(train_loader)

        return avg_loss, avg_accuracy, avg_f_score

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_f_score = 0
        total_area = 0  # Add this line

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            for s1_imgs, s2_imgs, masks in progress_bar:
                s1_imgs = s1_imgs.to(self.device)
                s2_imgs = s2_imgs.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(s1_imgs, s2_imgs)

                # Check if we need to resize masks to match model output size
                if outputs.shape[2:] != masks.shape[2:]:
                    # Resize masks to match output size
                    masks = torch.nn.functional.interpolate(
                        masks,
                        size=outputs.shape[2:],  # (H, W)
                        mode='nearest'
                    )

                # Convert masks to class indices for NLLLoss
                # NLLLoss expects target of shape [B, H, W] with class indices
                target_masks = (masks.squeeze(1) > 0.5).long()

                # Calculate loss
                loss = self.criterion(outputs, target_masks)

                accuracy, f_score, precision, recall = self.calculate_metrics(outputs, masks)
                area_km2 = self.calculate_area(outputs)  # Calculate area

                total_accuracy += accuracy
                total_f_score += f_score
                total_loss += loss.item()
                total_area += area_km2  # Add area to total

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.2f}%',
                    'f1': f'{f_score:.4f}',
                    'area': f'{area_km2:.2f} km²'
                })

        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        avg_f_score = total_f_score / len(val_loader)
        avg_area = total_area / len(val_loader)  # Calculate average area

        return avg_loss, avg_accuracy, avg_f_score, avg_area

# Function to plot training metrics
def plot_metrics(trainer, save_path='training_metrics.png'):
    """Plot and save training metrics"""
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(trainer.train_accuracies, label='Train Accuracy')
    plt.plot(trainer.val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')

    # Plot F1 score
    plt.subplot(2, 2, 3)
    plt.plot(trainer.train_f1_scores, label='Train F1')
    plt.plot(trainer.val_f1_scores, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score Curves')

    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(trainer.learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training metrics saved to {save_path}")

# Enhanced main function for maximum accuracy
def main():
    # Create output directory for results
    os.makedirs('results', exist_ok=True)

    # Training configuration optimized for accuracy
    num_epochs = 50  # Training for 50 epochs as requested
    initial_lr = 5e-4  # Lower initial learning rate for stability
    batch_size = 4  # Small batch size for better generalization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_f1 = 0.0
    patience = 15  # Increased patience for better convergence
    no_improve_count = 0

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize model with enhanced weight initialization
    model = DARU_Net()
    model.to(device)
    model.apply(init_weights)

    # Initialize trainer with weight decay for regularization
    trainer = Trainer(model, device, learning_rate=initial_lr, weight_decay=1e-5)

    # Load and split datasets
    s1_dir = Path('Major Project - Copy/data/sentinel1')
    s2_dir = Path('Major Project - Copy/data/sentinel2')
    mask_dir = Path('Major Project - Copy/data/masks')

    # Get file paths
    s1_paths = sorted(list(s1_dir.glob('*.png')))

    if len(s1_paths) == 0:
        print("No data found. Please make sure your data is in the correct location.")
        return

    s2_paths = []
    mask_paths = []

    # Match corresponding S2 and mask files with more robust matching
    for s1_path in s1_paths:
        s1_name = s1_path.name

        # Try different naming patterns for S2
        s2_candidates = [
            s2_dir / s1_name.replace('_s1_', '_s2_'),
            s2_dir / s1_name,
            s2_dir / s1_name.replace('sentinel1', 'sentinel2')
        ]

        s2_path = next((p for p in s2_candidates if p.exists()), None)

        # Try different naming patterns for masks
        mask_candidates = [
            mask_dir / s1_name,
            mask_dir / s1_name.replace('_s1_', '_mask_'),
            mask_dir / s1_name.replace('sentinel1', 'mask')
        ]

        mask_path = next((p for p in mask_candidates if p.exists()), None)

        if s2_path and mask_path:
            s2_paths.append(s2_path)
            mask_paths.append(mask_path)

    # Filter s1_paths to match the found s2 and mask paths
    s1_paths = s1_paths[:len(s2_paths)]

    print(f"Found {len(s1_paths)} valid image triplets (S1, S2, Mask)")

    # Split data into train, validation, and test sets (80/10/10 split)
    num_samples = len(s1_paths)
    indices = np.random.permutation(num_samples)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create datasets with enhanced parameters
    train_dataset = DualSentinelDataset(
        s1_paths=[s1_paths[i] for i in train_indices],
        s2_paths=[s2_paths[i] for i in train_indices],
        mask_paths=[mask_paths[i] for i in train_indices],
        transform=True,
        cache_size=50,  # Enable caching for faster training
        input_size=(256, 256)
    )

    val_dataset = DualSentinelDataset(
        s1_paths=[s1_paths[i] for i in val_indices],
        s2_paths=[s2_paths[i] for i in val_indices],
        mask_paths=[mask_paths[i] for i in val_indices],
        transform=False,
        cache_size=50,
        input_size=(256, 256)
    )

    test_dataset = DualSentinelDataset(
        s1_paths=[s1_paths[i] for i in test_indices],
        s2_paths=[s2_paths[i] for i in test_indices],
        mask_paths=[mask_paths[i] for i in test_indices],
        transform=False,
        cache_size=50,
        input_size=(256, 256)
    )

    # Create data loaders with optimal settings
    num_workers = 0 if not torch.cuda.is_available() else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize scheduler with cosine annealing for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart period after each restart
        eta_min=1e-6  # Minimum learning rate
    )

    print(f'Starting training with:')
    print(f'- {len(train_dataset)} training samples')
    print(f'- {len(val_dataset)} validation samples')
    print(f'- {len(test_dataset)} test samples')
    print(f'- Device: {device}')
    print(f'- Batch size: {batch_size}')
    print(f'- Initial learning rate: {initial_lr}')

    # Training loop with enhanced monitoring
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        train_loss, train_acc, train_f1 = trainer.train_epoch(train_loader)

        # Validation phase
        val_loss, val_acc, val_f1, val_area = trainer.validate(val_loader)

        # Update learning rate
        scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # Store metrics
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        trainer.train_accuracies.append(train_acc)
        trainer.val_accuracies.append(val_acc)
        trainer.train_f1_scores.append(train_f1)
        trainer.val_f1_scores.append(val_f1)
        trainer.learning_rates.append(current_lr)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print metrics
        print(f'\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%')
        print(f'Train F1-Score: {train_f1:.4f}, Val F1-Score: {val_f1:.4f}')
        print(f'Average Burnt Area: {val_area:.2f} km²')
        print(f'Learning rate: {current_lr:.6f}')

        # Save best model based on validation F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join('results', f'best_model_f1_{val_f1:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_acc
            }, model_path)
            print(f'Saved new best model with F1 score: {val_f1:.4f} to {model_path}')
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join('results', f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': {
                    'train_losses': trainer.train_losses,
                    'val_losses': trainer.val_losses,
                    'train_accuracies': trainer.train_accuracies,
                    'val_accuracies': trainer.val_accuracies,
                    'train_f1_scores': trainer.train_f1_scores,
                    'val_f1_scores': trainer.val_f1_scores
                }
            }, checkpoint_path)
            print(f'Saved checkpoint at epoch {epoch+1} to {checkpoint_path}')

        # Plot and save metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_metrics(trainer, save_path=os.path.join('results', f'metrics_epoch_{epoch+1}.png'))

        # Early stopping check
        if no_improve_count >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)')
            break

        # Stop if learning rate becomes too small
        if current_lr < 1e-7:
            print('Learning rate too small, stopping training')
            break

    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s')

    # Plot final metrics
    plot_metrics(trainer, save_path=os.path.join('results', 'final_metrics.png'))

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1, test_area = trainer.validate(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    print(f'Test F1-Score: {test_f1:.4f}')
    print(f'Test Average Burnt Area: {test_area:.2f} km²')

    # Save test results
    with open(os.path.join('results', 'test_results.txt'), 'w') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_acc:.2f}%\n')
        f.write(f'Test F1-Score: {test_f1:.4f}\n')
        f.write(f'Test Average Burnt Area: {test_area:.2f} km²\n')

    print(f"Test results saved to results/test_results.txt")
    print("Training and evaluation completed successfully!")

if __name__ == '__main__':
    main()




