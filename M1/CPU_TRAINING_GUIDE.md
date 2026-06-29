# CPU-Only DARU-Net Training Guide

## Overview
This guide provides a complete CPU-optimized training setup for DARU-Net to achieve maximum test accuracy without GPU dependency.

## Key Features

### 🚀 CPU-Optimized Architecture
- **Reduced Model Complexity**: Optimized for CPU efficiency while maintaining accuracy
- **Efficient CSAR Blocks**: Simplified attention mechanisms for faster CPU computation
- **BatchNorm Instead of GroupNorm**: Avoids batch size issues during validation
- **Optimized Convolution Blocks**: Streamlined for CPU performance

### 📊 Maximum Test Accuracy Configuration
- **90% Training Data**: Uses maximum available data for training
- **All Sentinel-2 Channels**: 12 channels for better feature representation
- **Data Augmentation**: Horizontal/vertical flips for better generalization
- **Combined Loss Function**: Focal Loss + Dice Loss for handling class imbalance

### 🔧 CPU-Specific Optimizations
- **Forced CPU Usage**: Multiple methods to ensure GPU is completely disabled
- **Batch Size 1**: Optimal for CPU training and memory efficiency
- **Efficient Data Loading**: Optimized preprocessing for CPU
- **Memory Management**: Reduced memory footprint for CPU training

## File Structure

```
Major Project - Copy/
├── darunet_cpu_optimized.py      # CPU-optimized DARU-Net model
├── train_cpu_optimized.py        # CPU-optimized training script
├── start_cpu_training.py         # Simple script to start training
├── data_preprocessing.py         # Enhanced data preprocessing
├── results/                      # Training outputs
│   ├── best_cpu_optimized_model.pth
│   ├── test_results.json
│   ├── confusion_matrix.png
│   └── training_curves.png
└── data/                         # Dataset
    ├── sentinel1/
    ├── sentinel2/
    └── masks/
```

## How to Start Training

### Method 1: Simple Start (Recommended)
```bash
cd "Major Project - Copy"
python start_cpu_training.py
```

### Method 2: Direct Training
```bash
cd "Major Project - Copy"
python train_cpu_optimized.py
```

## Training Configuration

### Model Configuration
```python
use_all_s2_channels = True    # Use all 12 Sentinel-2 channels
num_epochs = 50              # Reduced for CPU efficiency
batch_size = 1               # Optimal for CPU
learning_rate = 1e-4         # Stable learning rate
```

### Data Split (Optimized for Maximum Accuracy)
- **Training**: 90% of data (~3,114 samples)
- **Validation**: 5% of data (~173 samples)
- **Testing**: 5% of data (~173 samples)

### Loss Function
- **Combined Loss**: 70% Focal Loss + 30% Dice Loss
- **Focal Loss**: Handles class imbalance (α=1, γ=2)
- **Dice Loss**: Improves segmentation performance

### Optimizer
- **AdamW**: Weight decay = 1e-3 for regularization
- **Learning Rate Scheduler**: ReduceLROnPlateau with patience=10
- **Gradient Clipping**: Max norm = 1.0 for stability

## Expected Performance

### Training Time (CPU)
- **Estimated Time**: 6-10 hours for 50 epochs
- **Per Epoch**: ~7-12 minutes depending on CPU
- **Early Stopping**: Patience = 20 epochs

### Expected Accuracy
- **Target Test Accuracy**: 85-92%
- **Target F1 Score**: 0.80-0.90
- **Validation Accuracy**: Should reach 88-95%

## Monitoring Training

### Real-time Metrics
- Training/Validation Loss
- Training/Validation Accuracy
- Training/Validation F1 Score
- Learning Rate Updates

### Automatic Saving
- **Best Model**: Saved when validation F1 improves
- **Checkpoints**: Every 20 epochs
- **Early Stopping**: If no improvement for 20 epochs

## Final Evaluation

### Comprehensive Test Metrics
- **Test Accuracy**: Overall pixel-wise accuracy
- **F1 Score**: Weighted F1 for both classes
- **Precision/Recall**: Per-class performance
- **Confusion Matrix**: Visual performance analysis

### Output Files
- `test_results.json`: Complete numerical results
- `confusion_matrix.png`: Visual confusion matrix
- `training_curves.png`: Training progress plots

## Troubleshooting

### Common Issues

#### 1. GPU Still Being Used
```python
# The script automatically disables GPU with:
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False
torch.backends.cudnn.enabled = False
```

#### 2. Memory Issues
- Batch size is set to 1 for minimal memory usage
- Model uses efficient BatchNorm instead of GroupNorm
- Data loading is optimized for CPU

#### 3. Slow Training
- This is expected on CPU
- Training time: 6-10 hours is normal
- Consider reducing epochs if needed

#### 4. Low Accuracy
- Ensure all 12 Sentinel-2 channels are used
- Check data preprocessing quality
- Verify 90% training split is working

## Model Architecture Details

### CPU-Optimized CSAR Block
```python
- Simplified channel attention (reduction_ratio=8)
- Efficient spatial attention (7x7 kernel)
- BatchNorm for stability
- Dropout for regularization
- Residual connections
```

### Encoder-Decoder Structure
```python
- Dual encoders: Sentinel-1 (1 channel) + Sentinel-2 (12 channels)
- 5 levels: [32, 64, 128, 256, 512] filters
- Skip connections with feature fusion
- CSAR blocks at each level
```

### Output Layer
```python
- Final convolution: 32 → 16 → 2 channels
- LogSoftmax activation
- 256x256 output resolution
```

## Best Practices for CPU Training

1. **Close Other Applications**: Free up CPU resources
2. **Monitor Temperature**: Ensure adequate cooling
3. **Stable Power**: Use consistent power supply
4. **Patience**: CPU training takes time but works well
5. **Regular Checkpoints**: Training saves progress automatically

## Expected Results

After successful training, you should see:
- Test accuracy > 85%
- F1 score > 0.80
- Well-balanced confusion matrix
- Smooth training curves
- Comprehensive evaluation metrics

The CPU-optimized approach prioritizes achieving the best possible test accuracy while ensuring stable, reliable training on CPU hardware.
