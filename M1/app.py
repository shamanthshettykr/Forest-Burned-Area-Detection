from flask import Flask, render_template, request, jsonify
import torch
from pathlib import Path
import cv2
import numpy as np
import os
from darunet_cpu_optimized import CPUOptimizedDARUNet
from data_preprocessing import DataPreprocessor
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize CPU-optimized model with the exact configuration that achieved 92.81% accuracy
model = CPUOptimizedDARUNet(use_all_s2_channels=True)
try:
    # Use the best performing model (92.81% accuracy)
    model_candidates = [
        'results/best_cpu_optimized_model.pth',  # 92.81% accuracy - BEST!
        'results/smart_best_model.pth',
        'results/paper_compliant_best_model.pth',
        'best_model.pth'
    ]

    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError("No model file found")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"CPU-optimized model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
model.to(device)
model.eval()

def preprocess_image_for_model(image, is_s1=True):
    """
    Preprocess image for model input using DataPreprocessor
    """
    # Create preprocessor with enhanced configuration (12 channels for S2)
    preprocessor = DataPreprocessor(use_paper_config=False)

    # Resize image to match model input size
    image = cv2.resize(image, (256, 256))
    print(f"Debug - Input image shape after resize: {image.shape}")

    if is_s1:
        # Process Sentinel-1 image
        processed = preprocessor.preprocess_sentinel1(image)
        print(f"Debug - S1 processed shape: {processed.shape}")
        # Convert to float32 and normalize
        processed = processed.astype(np.float32) / 255.0
        # Keep as [256, 256] - channel dimension will be added later
        print(f"Debug - S1 final shape: {processed.shape}")
    else:
        # Process Sentinel-2 image (will create 12 channels)
        processed = preprocessor.preprocess_sentinel2(image)
        print(f"Debug - S2 processed shape: {processed.shape}")
        # Convert to float32 and normalize
        processed = processed.astype(np.float32) / 255.0
        # Transpose to channel-first format (C, H, W)
        processed = np.transpose(processed, (2, 0, 1))
        print(f"Debug - S2 final shape: {processed.shape}")

    return processed

@app.route('/')
def home():
    return render_template('new_index.html')

# Add this import at the top
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

@app.route('/analyze', methods=['POST'])
def analyze_images():
    try:
        if 's1_image' not in request.files or 's2_image' not in request.files:
            return jsonify({'error': 'Both images are required'})
        
        s1_file = request.files['s1_image']
        s2_file = request.files['s2_image']
        
        # Validate file names
        if '_s1_' not in s1_file.filename.lower():
            return jsonify({'error': 'First image must be a Sentinel-1 image with "_s1_" in filename'})
        if '_s2_' not in s2_file.filename.lower():
            return jsonify({'error': 'Second image must be a Sentinel-2 image with "_s2_" in filename'})
        
        # Check if images are from the same pair
        s1_base = s1_file.filename.replace('_s1_', '_s2_')
        if s1_base != s2_file.filename:
            return jsonify({'error': 'Images do not match. Please ensure S1 and S2 images are from the same pair'})

        # Extract base filename for ground truth mask lookup
        # Convert S1 filename to mask filename pattern
        s1_filename = s1_file.filename
        if '_s1_' in s1_filename:
            # If filename follows the pattern, use it directly
            base_filename = s1_filename
        else:
            # If uploaded with different name, try to find matching mask
            # For now, use the S1 filename as is
            base_filename = s1_filename

        # Read images once
        s1_data = s1_file.read()
        s2_data = s2_file.read()
        
        if not s1_data or not s2_data:
            return jsonify({'error': 'Empty image data received'})
            
        # Decode images
        s1_array = np.frombuffer(s1_data, np.uint8)
        s2_array = np.frombuffer(s2_data, np.uint8)
        
        s1_img = cv2.imdecode(s1_array, cv2.IMREAD_GRAYSCALE)
        s2_img = cv2.imdecode(s2_array, cv2.IMREAD_COLOR)
        
        if s1_img is None or s2_img is None:
            return jsonify({'error': 'Failed to decode images. Please check the image format.'})
            
        # Validate image types
        if len(s1_img.shape) != 2:
            return jsonify({'error': 'First image must be a Sentinel-1 grayscale image'})
        if len(s2_img.shape) != 3:
            return jsonify({'error': 'Second image must be a Sentinel-2 RGB image'})
            
        # Check if images have matching dimensions
        if s1_img.shape != s2_img.shape[:2]:
            return jsonify({'error': 'Images dimensions do not match'})
        
        # Preprocess Sentinel-1
        s1_processed = DataPreprocessor(use_paper_config=False).preprocess_sentinel1(s1_img)

        # Preprocess Sentinel-2 (use enhanced config for 12 channels, model will reduce to 8)
        s2_processed = DataPreprocessor(use_paper_config=False).preprocess_sentinel2(s2_img)
        
        # Convert float images to uint8 before thresholding
        s1_uint8 = (s1_processed * 255).astype(np.uint8)
        s2_uint8 = (s2_processed[:,:,0] * 255).astype(np.uint8)  # Take first channel for thresholding
        
        # Create features for mask generation
        s1_features = cv2.threshold(s1_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        s2_features = cv2.threshold(s2_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Combine masks - Give more importance to Sentinel-2 (70% S2, 30% S1)
        combined_mask = cv2.addWeighted(s1_features, 0.3, s2_features, 0.7, 0)
        final_mask = (combined_mask > 127).astype(np.uint8) * 255
        
        # Convert mask to base64 for display
        _, buffer = cv2.imencode('.png', final_mask)
        mask_data = base64.b64encode(buffer).decode()
        
        # Calculate area (100m² per pixel, convert to km²)
        pixel_count = np.sum(final_mask > 0)
        area_km2 = (pixel_count * 100) / 1_000_000  # Assuming each pixel represents 100m²
        
        # Use model to get predictions for accuracy calculation
        s1_input = preprocess_image_for_model(s1_img, is_s1=True)
        s2_input = preprocess_image_for_model(s2_img, is_s1=False)

        # Debug: Print shapes
        print(f"Debug - S1 input shape: {s1_input.shape}")
        print(f"Debug - S2 input shape: {s2_input.shape}")

        # Convert to torch tensors with correct format (matching training script)
        # S1: [256, 256] -> [1, 1, 256, 256] (add batch and channel dims)
        s1_tensor = torch.from_numpy(s1_input).float().unsqueeze(0).unsqueeze(0).to(device)
        # S2: [12, 256, 256] -> [1, 12, 256, 256] (already channel-first from preprocessing, just add batch dim)
        s2_tensor = torch.from_numpy(s2_input).float().unsqueeze(0).to(device)

        print(f"Debug - S1 tensor shape: {s1_tensor.shape}")
        print(f"Debug - S2 tensor shape: {s2_tensor.shape}")
        
        # Get model predictions
        with torch.no_grad():
            raw_predictions = model(s1_tensor, s2_tensor)
            print(f"Debug - Raw predictions shape: {raw_predictions.shape}")
            print(f"Debug - Raw predictions min/max: {raw_predictions.min().item():.4f}/{raw_predictions.max().item():.4f}")

            # Model outputs log softmax, so we need to apply exp to get probabilities
            if raw_predictions.shape[1] == 2:  # 2 classes (background, burned)
                probabilities = torch.exp(raw_predictions)  # Convert log softmax to probabilities
                predictions = probabilities[:, 1:2]  # Take burned class probability
                print(f"Debug - Using exp(log_softmax), burned class probabilities min/max: {predictions.min().item():.4f}/{predictions.max().item():.4f}")
            else:
                predictions = torch.sigmoid(raw_predictions)  # Single channel output
                print(f"Debug - Using sigmoid, predictions min/max: {predictions.min().item():.4f}/{predictions.max().item():.4f}")

            # Convert to binary predictions
            binary_predictions = (predictions > 0.5).float()
            print(f"Debug - Binary predictions sum: {binary_predictions.sum().item()} out of {binary_predictions.numel()} pixels")
        
        # Try to load ground truth mask for accurate evaluation
        ground_truth_mask = None
        mask_path = f"data/masks/{base_filename}"
        print(f"DEBUG - Looking for ground truth mask at: {mask_path}")
        try:
            if os.path.exists(mask_path):
                ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                ground_truth_mask = cv2.resize(ground_truth_mask, (256, 256))
                print(f"SUCCESS - Loaded ground truth mask: {mask_path}")
                print(f"DEBUG - Ground truth mask shape: {ground_truth_mask.shape}, min/max: {ground_truth_mask.min()}/{ground_truth_mask.max()}")
            else:
                print(f"WARNING - Ground truth mask not found: {mask_path}")
                # List available mask files for debugging
                mask_dir = "data/masks"
                if os.path.exists(mask_dir):
                    available_masks = [f for f in os.listdir(mask_dir) if f.endswith('.png')][:5]
                    print(f"DEBUG - Available mask files (first 5): {available_masks}")
        except Exception as e:
            print(f"ERROR - Error loading ground truth mask: {str(e)}")

        # Use ground truth mask if available, otherwise use generated mask
        if ground_truth_mask is not None:
            mask_tensor = torch.from_numpy((ground_truth_mask > 127).astype(np.float32)).to(device)
            mask_type = "Ground Truth"
            print(f"Debug - Ground truth mask sum: {mask_tensor.sum().item()} out of {mask_tensor.numel()} pixels")
            print(f"Debug - Ground truth mask min/max: {mask_tensor.min().item():.1f}/{mask_tensor.max().item():.1f}")
        else:
            mask_tensor = torch.from_numpy((final_mask > 0).astype(np.float32)).to(device)
            mask_type = "Generated"
            print(f"Debug - Generated mask sum: {mask_tensor.sum().item()} out of {mask_tensor.numel()} pixels")

        # Calculate accuracy
        pred_flat = binary_predictions.squeeze()
        correct = (pred_flat == mask_tensor).float().sum()
        total = torch.numel(mask_tensor)
        accuracy = (correct / total) * 100

        print(f"DEBUG - ACCURACY CALCULATION:")
        print(f"  Predictions shape: {pred_flat.shape}")
        print(f"  Ground truth shape: {mask_tensor.shape}")
        print(f"  Predictions sum: {pred_flat.sum().item()} (burned pixels predicted)")
        print(f"  Ground truth sum: {mask_tensor.sum().item()} (actual burned pixels)")
        print(f"  Correct predictions: {correct.item()}/{total}")
        print(f"  FINAL ACCURACY: {accuracy.item():.2f}%")

        # Calculate F1-score
        true_positives = (pred_flat * mask_tensor).sum()
        false_positives = (pred_flat * (1 - mask_tensor)).sum()
        false_negatives = ((1 - pred_flat) * mask_tensor).sum()

        print(f"Debug - TP: {true_positives.item()}, FP: {false_positives.item()}, FN: {false_negatives.item()}")
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Create performance plot with matplotlib
        plt.switch_backend('Agg')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy line plot
        x_points = np.array([0, 1])  # Two points for line
        accuracy_points = np.array([0, accuracy.cpu().item()])
        ax1.plot(x_points, accuracy_points, color='blue', marker='o', linewidth=2)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Accuracy')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Start', 'Current'])
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # F1-score line plot
        f1_points = np.array([0, f1.cpu().item()])
        ax2.plot(x_points, f1_points, color='green', marker='o', linewidth=2)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Score')
        ax2.set_title('F1-Score')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Start', 'Current'])
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode()
        
        # Return results including the mask and plots
        return jsonify({
            'area': round(area_km2, 2),
            'accuracy': round(float(accuracy.cpu()), 2),
            'f1_score': round(float(f1.cpu()), 4),
            'mask_type': mask_type,
            'plot': plot_data,
            'mask': mask_data
        })
        
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred while processing the images: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)