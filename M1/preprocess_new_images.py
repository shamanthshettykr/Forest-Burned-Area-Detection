"""
Enhanced Image Preprocessing and Mask Generation Script
Processes new Sentinel-1 and Sentinel-2 images and generates masks
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import json
from data_preprocessing import DataPreprocessor, process_image_pairs, create_combined_mask

def count_files_in_directory(directory):
    """Count files in a directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith('.png')])

def validate_image_pairs(s1_dir, s2_dir):
    """Validate that Sentinel-1 and Sentinel-2 images are properly paired"""
    s1_files = set(f.name for f in Path(s1_dir).glob('*.png'))
    s2_files = set(f.name.replace('_s2_', '_s1_') for f in Path(s2_dir).glob('*.png'))
    
    # Find matching pairs
    matching_pairs = s1_files.intersection(s2_files)
    
    # Find unmatched files
    unmatched_s1 = s1_files - matching_pairs
    unmatched_s2 = s2_files - matching_pairs
    
    print(f"📊 Image Pairing Analysis:")
    print(f"   Sentinel-1 files: {len(s1_files)}")
    print(f"   Sentinel-2 files: {len(s2_files)}")
    print(f"   Matching pairs: {len(matching_pairs)}")
    print(f"   Unmatched S1: {len(unmatched_s1)}")
    print(f"   Unmatched S2: {len(unmatched_s2)}")
    
    if unmatched_s1:
        print(f"   First 5 unmatched S1: {list(unmatched_s1)[:5]}")
    if unmatched_s2:
        print(f"   First 5 unmatched S2: {list(unmatched_s2)[:5]}")
    
    return matching_pairs, unmatched_s1, unmatched_s2

def check_image_quality(image_path, min_size=(64, 64), max_size=(2048, 2048)):
    """Check if an image meets quality requirements"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, "Cannot read image"
        
        h, w = img.shape[:2]
        
        # Check size constraints
        if h < min_size[0] or w < min_size[1]:
            return False, f"Image too small: {w}x{h}"
        
        if h > max_size[0] or w > max_size[1]:
            return False, f"Image too large: {w}x{h}"
        
        # Check for corrupted data
        if np.all(img == 0):
            return False, "Image is all black"
        
        if np.all(img == 255):
            return False, "Image is all white"
        
        # Check for reasonable variance
        if np.std(img) < 1.0:
            return False, "Image has very low variance"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def quality_check_dataset(s1_dir, s2_dir, mask_dir=None):
    """Perform comprehensive quality check on the dataset"""
    print("\n🔍 Performing dataset quality check...")
    
    s1_files = list(Path(s1_dir).glob('*.png'))
    s2_files = list(Path(s2_dir).glob('*.png'))
    
    # Check Sentinel-1 images
    print("\n📡 Checking Sentinel-1 images...")
    s1_issues = []
    for s1_file in tqdm(s1_files[:100], desc="S1 Quality Check"):  # Check first 100
        is_valid, message = check_image_quality(s1_file)
        if not is_valid:
            s1_issues.append((s1_file.name, message))
    
    # Check Sentinel-2 images
    print("\n🛰️ Checking Sentinel-2 images...")
    s2_issues = []
    for s2_file in tqdm(s2_files[:100], desc="S2 Quality Check"):  # Check first 100
        is_valid, message = check_image_quality(s2_file)
        if not is_valid:
            s2_issues.append((s2_file.name, message))
    
    # Check masks if directory exists
    mask_issues = []
    if mask_dir and os.path.exists(mask_dir):
        print("\n🎭 Checking mask images...")
        mask_files = list(Path(mask_dir).glob('*.png'))
        for mask_file in tqdm(mask_files[:100], desc="Mask Quality Check"):
            is_valid, message = check_image_quality(mask_file)
            if not is_valid:
                mask_issues.append((mask_file.name, message))
    
    # Report results
    print(f"\n📋 Quality Check Results:")
    print(f"   S1 issues: {len(s1_issues)}")
    print(f"   S2 issues: {len(s2_issues)}")
    print(f"   Mask issues: {len(mask_issues)}")
    
    if s1_issues:
        print(f"   First 3 S1 issues: {s1_issues[:3]}")
    if s2_issues:
        print(f"   First 3 S2 issues: {s2_issues[:3]}")
    if mask_issues:
        print(f"   First 3 mask issues: {mask_issues[:3]}")
    
    return s1_issues, s2_issues, mask_issues

def preprocess_and_generate_masks():
    """Main function to preprocess images and generate masks"""
    print("🚀 Starting Enhanced Image Preprocessing and Mask Generation")
    print("=" * 70)
    
    # Define directories
    s1_dir = 'data/sentinel1'
    s2_dir = 'data/sentinel2'
    mask_dir = 'data/masks'
    
    # Create output directory
    os.makedirs(mask_dir, exist_ok=True)
    
    # Initial file counts
    print("\n📊 Initial Dataset Statistics:")
    s1_count = count_files_in_directory(s1_dir)
    s2_count = count_files_in_directory(s2_dir)
    mask_count = count_files_in_directory(mask_dir)
    
    print(f"   Sentinel-1 images: {s1_count}")
    print(f"   Sentinel-2 images: {s2_count}")
    print(f"   Existing masks: {mask_count}")
    
    # Validate image pairs
    matching_pairs, unmatched_s1, unmatched_s2 = validate_image_pairs(s1_dir, s2_dir)
    
    # Quality check
    s1_issues, s2_issues, mask_issues = quality_check_dataset(s1_dir, s2_dir, mask_dir)
    
    # Calculate how many new masks need to be generated
    new_masks_needed = len(matching_pairs) - mask_count
    print(f"\n🎯 Masks to generate: {max(0, new_masks_needed)}")
    
    if new_masks_needed <= 0:
        print("✅ All masks already exist!")
        return
    
    # Start mask generation
    print(f"\n🏭 Starting mask generation for {len(matching_pairs)} image pairs...")
    start_time = time.time()
    
    try:
        # Process image pairs with enhanced settings
        process_image_pairs(
            s1_dir=s1_dir,
            s2_dir=s2_dir,
            output_dir=mask_dir,
            batch_size=100,  # Process in larger batches
            resume_from=True  # Resume from existing masks
        )
        
        # Final statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        final_mask_count = count_files_in_directory(mask_dir)
        new_masks_generated = final_mask_count - mask_count
        
        print(f"\n🎉 Mask Generation Completed!")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   New masks generated: {new_masks_generated}")
        print(f"   Total masks: {final_mask_count}")
        print(f"   Success rate: {(new_masks_generated / max(1, new_masks_needed)) * 100:.1f}%")
        
        # Save processing report
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time_seconds': processing_time,
            'initial_counts': {
                'sentinel1': s1_count,
                'sentinel2': s2_count,
                'masks': mask_count
            },
            'final_counts': {
                'masks': final_mask_count
            },
            'new_masks_generated': new_masks_generated,
            'matching_pairs': len(matching_pairs),
            'quality_issues': {
                'sentinel1': len(s1_issues),
                'sentinel2': len(s2_issues),
                'masks': len(mask_issues)
            },
            'unmatched_files': {
                'sentinel1': len(unmatched_s1),
                'sentinel2': len(unmatched_s2)
            }
        }
        
        os.makedirs('results', exist_ok=True)
        with open('results/preprocessing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Processing report saved to results/preprocessing_report.json")
        
    except Exception as e:
        print(f"❌ Error during mask generation: {str(e)}")
        raise

def validate_final_dataset():
    """Validate the final dataset after preprocessing"""
    print("\n🔍 Final Dataset Validation...")
    
    s1_dir = 'data/sentinel1'
    s2_dir = 'data/sentinel2'
    mask_dir = 'data/masks'
    
    # Count final files
    s1_count = count_files_in_directory(s1_dir)
    s2_count = count_files_in_directory(s2_dir)
    mask_count = count_files_in_directory(mask_dir)
    
    # Find complete triplets
    s1_files = set(f.stem for f in Path(s1_dir).glob('*.png'))
    s2_files = set(f.stem.replace('_s2_', '_s1_') for f in Path(s2_dir).glob('*.png'))
    mask_files = set(f.stem for f in Path(mask_dir).glob('*.png'))
    
    complete_triplets = s1_files.intersection(s2_files).intersection(mask_files)
    
    print(f"📊 Final Dataset Statistics:")
    print(f"   Sentinel-1 images: {s1_count}")
    print(f"   Sentinel-2 images: {s2_count}")
    print(f"   Mask images: {mask_count}")
    print(f"   Complete triplets: {len(complete_triplets)}")
    print(f"   Dataset completeness: {(len(complete_triplets) / max(1, s1_count)) * 100:.1f}%")
    
    return len(complete_triplets)

if __name__ == "__main__":
    try:
        # Run preprocessing and mask generation
        preprocess_and_generate_masks()
        
        # Validate final dataset
        complete_triplets = validate_final_dataset()
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"🎯 Ready for training with {complete_triplets} complete image triplets")
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {str(e)}")
        raise
