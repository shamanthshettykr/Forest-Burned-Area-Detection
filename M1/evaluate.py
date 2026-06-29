import torch
from darunet import DARU_Net
from dataset import DualSentinelDataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DARU_Net()
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    model.to(device)
    model.eval()
    
    # Load test dataset
    # Update paths to use absolute paths
    s1_paths = sorted(list(Path('e:/Major Project/data/sentinel1').glob('*.png')))
    s2_paths = [Path('e:/Major Project/data/sentinel2') / p.name.replace('_s1_', '_s2_') for p in s1_paths]
    mask_paths = [Path('e:/Major Project/data/masks') / p.name for p in s1_paths]
    
    # Use 20% for testing
    split_idx = int(0.8 * len(s1_paths))
    test_dataset = DualSentinelDataset(
        s1_paths=s1_paths[split_idx:],
        s2_paths=s2_paths[split_idx:],
        mask_paths=mask_paths[split_idx:],
        transform=False
    )
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # After loading paths
    print(f"Found {len(s1_paths)} Sentinel-1 images")
    print(f"Found {len(s2_paths)} Sentinel-2 images")
    print(f"Found {len(mask_paths)} mask images")
    
    # After creating test dataset
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of batches: {len(test_loader)}")
    
    total_accuracy = 0
    total_f1 = 0
    total_area = 0
    num_batches = 0
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        for batch_idx, (s1_imgs, s2_imgs, masks) in enumerate(test_loader):
            print(f"Processing batch {batch_idx+1}/{len(test_loader)}", end='\r')
            
            s1_imgs = s1_imgs.to(device)
            s2_imgs = s2_imgs.to(device)
            masks = masks.to(device)
            
            outputs = model(s1_imgs, s2_imgs)
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct = (predictions == masks).float().sum()
            total = torch.numel(masks)
            accuracy = (correct / total) * 100
            
            # Calculate F1-score
            true_positives = (predictions * masks).sum()
            false_positives = (predictions * (1 - masks)).sum()
            false_negatives = ((1 - predictions) * masks).sum()
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Calculate area (10m² per pixel, convert to km²)
            def calculate_area(predictions):
                pixel_count = torch.sum(predictions > 0.5)
                area_m2 = pixel_count * 10  # Each pixel is 10m²
                area_km2 = area_m2 / 1_000_000  # Convert to km²
                return area_km2
            
            # In the evaluation loop:
            area = calculate_area(predictions)
            
            total_accuracy += accuracy.item()
            total_f1 += f1.item()
            total_area += area.item()
            num_batches += 1
    
    print("\nCalculating final metrics...")
    avg_accuracy = total_accuracy / num_batches
    avg_f1 = total_f1 / num_batches
    avg_area = total_area / num_batches
    
    print(f'Test Accuracy: {avg_accuracy:.2f}%')
    print(f'Test F1-Score: {avg_f1:.4f}')
    print(f'Average Area: {avg_area:.2f} km²')

if __name__ == '__main__':
    evaluate_model()