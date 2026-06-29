#!/usr/bin/env python3
"""
Script to check the best accuracy achieved during training
"""

import torch
import json
import os

def check_best_accuracy():
    """Check the best accuracy from the saved model"""
    
    # Check if the best model file exists
    model_path = 'results/best_cpu_optimized_model.pth'
    
    if not os.path.exists(model_path):
        print("❌ Best model file not found!")
        return
    
    try:
        # Load the checkpoint
        print("📊 Loading best model checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("\n" + "=" * 60)
        print("BEST TRAINING RESULTS")
        print("=" * 60)
        
        # Extract information from checkpoint
        if 'val_accuracy' in checkpoint:
            print(f"🎯 Best Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")
        
        if 'val_f1' in checkpoint:
            print(f"🎯 Best Validation F1 Score: {checkpoint['val_f1']:.4f}")
        
        if 'epoch' in checkpoint:
            print(f"📈 Best model saved at Epoch: {checkpoint['epoch'] + 1}")
        
        if 'test_accuracy' in checkpoint:
            print(f"🧪 Test Accuracy: {checkpoint['test_accuracy']:.2f}%")
        
        if 'test_f1_score' in checkpoint:
            print(f"🧪 Test F1 Score: {checkpoint['test_f1_score']:.4f}")
        
        if 'test_precision' in checkpoint:
            print(f"🧪 Test Precision: {checkpoint['test_precision']:.4f}")
        
        if 'test_recall' in checkpoint:
            print(f"🧪 Test Recall: {checkpoint['test_recall']:.4f}")
        
        # Check for training configuration
        if 'training_config' in checkpoint:
            config = checkpoint['training_config']
            print(f"\n📋 Training Configuration:")
            print(f"   - Epochs: {config.get('num_epochs', 'N/A')}")
            print(f"   - Batch Size: {config.get('batch_size', 'N/A')}")
            print(f"   - Learning Rate: {config.get('learning_rate', 'N/A')}")
        
        # Check for model parameters
        if 'model_parameters' in checkpoint:
            params = checkpoint['model_parameters']
            print(f"   - Model Parameters: {params:,}")
        
        # Check for training time
        if 'training_time_hours' in checkpoint:
            hours = checkpoint['training_time_hours']
            print(f"   - Training Time: {hours:.2f} hours")
        
        print("\n" + "=" * 60)
        
        # Show all available keys for debugging
        print("\n🔍 Available keys in checkpoint:")
        for key in sorted(checkpoint.keys()):
            if key != 'model_state_dict' and key != 'optimizer_state_dict':
                print(f"   - {key}: {type(checkpoint[key])}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {str(e)}")
        return None

def check_latest_results():
    """Check for any JSON results files"""
    results_dir = 'results'
    
    # Look for JSON files
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if json_files:
        print(f"\n📄 Found {len(json_files)} JSON result files:")
        for json_file in json_files:
            print(f"   - {json_file}")
            try:
                with open(os.path.join(results_dir, json_file), 'r') as f:
                    data = json.load(f)
                    if 'test_accuracy' in data:
                        print(f"     Test Accuracy: {data['test_accuracy']:.2f}%")
                    if 'best_val_accuracy' in data:
                        print(f"     Best Val Accuracy: {data['best_val_accuracy']:.2f}%")
            except Exception as e:
                print(f"     Error reading {json_file}: {str(e)}")

if __name__ == "__main__":
    print("🚀 Checking Best Training Results")
    print("=" * 60)
    
    # Check the best model
    checkpoint = check_best_accuracy()
    
    # Check for JSON results
    check_latest_results()
    
    # Summary
    if checkpoint:
        print("\n✅ Successfully loaded training results!")
        
        # Extract the key metrics
        val_acc = checkpoint.get('val_accuracy', 'N/A')
        val_f1 = checkpoint.get('val_f1', 'N/A')
        test_acc = checkpoint.get('test_accuracy', 'N/A')
        test_f1 = checkpoint.get('test_f1_score', 'N/A')
        
        print(f"\n🏆 SUMMARY:")
        print(f"   Best Validation Accuracy: {val_acc}")
        print(f"   Best Validation F1: {val_f1}")
        print(f"   Final Test Accuracy: {test_acc}")
        print(f"   Final Test F1: {test_f1}")
    else:
        print("\n❌ Could not load training results.")
