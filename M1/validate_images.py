import cv2
from pathlib import Path
import os

def validate_and_clean_dataset():
    # Get all image paths
    s1_dir = Path('e:/Major Project/data/sentinel1')
    s2_dir = Path('e:/Major Project/data/sentinel2')
    mask_dir = Path('e:/Major Project/data/masks')
    
    # Get all s1 images
    s1_paths = sorted(list(s1_dir.glob('*.png')))
    valid_pairs = []
    corrupted = []
    
    print("Validating image pairs...")
    for s1_path in s1_paths:
        # Get corresponding s2 and mask paths
        s2_path = s2_dir / s1_path.name.replace('_s1_', '_s2_')
        mask_path = mask_dir / s1_path.name
        
        # Check if all files exist and can be opened
        try:
            s1_img = cv2.imread(str(s1_path), cv2.IMREAD_GRAYSCALE)
            s2_img = cv2.imread(str(s2_path))
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if s1_img is None or s2_img is None or mask_img is None:
                raise Exception("Failed to load image")
                
            valid_pairs.append((s1_path, s2_path, mask_path))
        except Exception as e:
            corrupted.append((s1_path, s2_path, mask_path))
            print(f"\nCorrupted set found:")
            print(f"S1: {s1_path}")
            print(f"S2: {s2_path}")
            print(f"Mask: {mask_path}")
    
    print(f"\nValidation complete:")
    print(f"Valid pairs: {len(valid_pairs)}")
    print(f"Corrupted pairs: {len(corrupted)}")
    
    return valid_pairs, corrupted

if __name__ == '__main__':
    valid_pairs, corrupted = validate_and_clean_dataset()