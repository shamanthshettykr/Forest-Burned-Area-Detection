"""
Simple script to start CPU-only training for DARU-Net
Ensures GPU is completely disabled and training runs on CPU only
"""

import torch
import os
import sys

def main():
    """Start CPU-only training"""
    
    # Force CPU usage - multiple methods to ensure GPU is disabled
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide all CUDA devices
    torch.cuda.is_available = lambda: False  # Override CUDA availability
    torch.backends.cudnn.enabled = False     # Disable cuDNN
    
    print("=" * 60)
    print("STARTING CPU-ONLY DARU-NET TRAINING")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.device('cpu')}")
    print("=" * 60)
    
    # Import and run the training
    try:
        from train_cpu_optimized import main as train_main
        print("Starting training...")
        train_main()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed!")
        sys.exit(1)
