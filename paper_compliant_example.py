"""
Example script demonstrating how to use DARU-Net with paper-compliant configuration.

This script shows:
1. How to initialize the model with paper specifications
2. How to use the L2 loss function as described in the paper
3. How to switch between softmax and log_softmax outputs
4. Comparison between paper config and enhanced config
"""

import torch
import torch.nn as nn
import torch.optim as optim
from darunet import DARU_Net, PaperL2Loss
import numpy as np

def create_sample_data(batch_size=2, height=256, width=256):
    """Create sample data for testing"""
    # Sentinel-1 data (VH polarization) - 1 channel
    s1_data = torch.randn(batch_size, 1, height, width)
    
    # Sentinel-2 data (RGB + NIR) - 4 channels for paper config
    s2_data_paper = torch.randn(batch_size, 4, height, width)
    
    # Sentinel-2 data (all channels) - 12 channels for enhanced config
    s2_data_enhanced = torch.randn(batch_size, 12, height, width)
    
    # Binary masks (ground truth)
    masks = torch.randint(0, 2, (batch_size, height, width)).float()
    
    return s1_data, s2_data_paper, s2_data_enhanced, masks

def demonstrate_paper_config():
    """Demonstrate the paper-compliant configuration"""
    print("=== Paper-Compliant DARU-Net Configuration ===")
    
    # Initialize model with paper configuration
    model_paper = DARU_Net(use_paper_config=True, use_log_softmax=False)
    
    print(f"Model parameters: {sum(p.numel() for p in model_paper.parameters()):,}")
    
    # Create sample data
    s1_data, s2_data_paper, _, masks = create_sample_data()
    
    # Forward pass
    with torch.no_grad():
        outputs = model_paper(s1_data, s2_data_paper)
        print(f"Output shape: {outputs.shape}")
        print(f"Output type: {'Softmax probabilities' if not model_paper.use_log_softmax else 'Log probabilities'}")
    
    # Initialize L2 loss as per paper
    criterion = PaperL2Loss()
    
    # Calculate loss
    loss = criterion(outputs, masks)
    print(f"L2 Loss (as per paper): {loss.item():.6f}")
    
    return model_paper

def demonstrate_enhanced_config():
    """Demonstrate the enhanced configuration"""
    print("\n=== Enhanced DARU-Net Configuration ===")
    
    # Initialize model with enhanced configuration
    model_enhanced = DARU_Net(use_paper_config=False, use_log_softmax=True)
    
    print(f"Model parameters: {sum(p.numel() for p in model_enhanced.parameters()):,}")
    
    # Create sample data
    s1_data, _, s2_data_enhanced, masks = create_sample_data()
    
    # Forward pass
    with torch.no_grad():
        outputs = model_enhanced(s1_data, s2_data_enhanced)
        print(f"Output shape: {outputs.shape}")
        print(f"Output type: {'Log probabilities' if model_enhanced.use_log_softmax else 'Softmax probabilities'}")
    
    # Use NLLLoss with log_softmax outputs
    criterion = nn.NLLLoss()
    targets = masks.long()  # Convert to class indices
    
    loss = criterion(outputs, targets)
    print(f"NLL Loss (for enhanced config): {loss.item():.6f}")
    
    return model_enhanced

def compare_configurations():
    """Compare paper vs enhanced configurations"""
    print("\n=== Configuration Comparison ===")
    
    # Paper config
    model_paper = DARU_Net(use_paper_config=True)
    paper_params = sum(p.numel() for p in model_paper.parameters())
    
    # Enhanced config
    model_enhanced = DARU_Net(use_paper_config=False)
    enhanced_params = sum(p.numel() for p in model_enhanced.parameters())
    
    print(f"Paper config parameters: {paper_params:,}")
    print(f"Enhanced config parameters: {enhanced_params:,}")
    print(f"Parameter ratio (Enhanced/Paper): {enhanced_params/paper_params:.2f}x")
    
    # Filter comparison
    print("\nFilter configurations:")
    print("Paper config filters: [16, 32, 64, 128, 256]")
    print("Enhanced config filters: [32, 64, 128, 256, 512]")
    
    print("\nInput channels:")
    print("Paper config - Sentinel-2: 4 channels (RGB + NIR)")
    print("Enhanced config - Sentinel-2: 12 channels (all bands)")

def training_example():
    """Example of training with paper-compliant configuration"""
    print("\n=== Training Example (Paper Configuration) ===")
    
    # Initialize model and loss
    model = DARU_Net(use_paper_config=True, use_log_softmax=False)
    criterion = PaperL2Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create sample data
    s1_data, s2_data, _, masks = create_sample_data()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    outputs = model(s1_data, s2_data)
    loss = criterion(outputs, masks)
    
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.6f}")
    print("Training step completed successfully!")

def main():
    """Main function to run all demonstrations"""
    print("DARU-Net Paper-Compliant Implementation Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstrate configurations
    model_paper = demonstrate_paper_config()
    model_enhanced = demonstrate_enhanced_config()
    
    # Compare configurations
    compare_configurations()
    
    # Training example
    training_example()
    
    print("\n=== Summary ===")
    print("✅ Paper-compliant configuration implemented")
    print("✅ L2 loss function as per paper equation (2)")
    print("✅ Softmax activation (instead of sigmoid)")
    print("✅ Filter numbers: 16, 32, 64, 128, 256 (as per paper)")
    print("✅ Sentinel-2 input: 4 channels (RGB + NIR)")
    print("✅ Enhanced configuration available as alternative")

if __name__ == "__main__":
    main()
