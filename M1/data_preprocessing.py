import cv2
import numpy as np
import os
from pathlib import Path
from scipy.ndimage import median_filter
from scipy import ndimage
from PIL import Image

__all__ = ['DataPreprocessor']

class DataPreprocessor:
    def __init__(self, use_paper_config=True):
        """
        Initialize DataPreprocessor with configuration options

        Args:
            use_paper_config (bool): If True, use paper specifications (4 channels for S2)
                                   If False, use enhanced configuration (12 channels for S2)
        """
        self.use_paper_config = use_paper_config
        self.s2_channels = 4 if use_paper_config else 12

    def preprocess_sentinel1(self, image):
        """
        Enhanced preprocessing for Sentinel-1 imagery with robust error handling
        """
        # Check if image is None
        if image is None:
            return np.zeros((256, 256), dtype=np.uint8)

        try:
            # Ensure image is in the correct format
            if image.dtype == np.float64:
                image = image.astype(np.float32)

            # Convert to uint8 if not already
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply median filtering for noise reduction
            filtered = cv2.medianBlur(image, 3)

            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)

            # Additional enhancement for better feature extraction
            # Apply Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

            # Apply histogram equalization for better contrast
            enhanced = cv2.equalizeHist(enhanced)

            return enhanced

        except Exception as e:
            print(f"Error in preprocess_sentinel1: {str(e)}")
            # Return original image if processing fails
            if image.dtype != np.uint8:
                return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return image

    def preprocess_sentinel2(self, image):
        """
        Enhanced preprocessing for Sentinel-2 imagery with configurable channel support
        """
        # Check if image is None
        if image is None:
            return np.zeros((256, 256, self.s2_channels), dtype=np.uint8)

        try:
            # Ensure image is in the correct format
            if image.dtype == np.float64:
                image = image.astype(np.float32)

            if self.use_paper_config:
                # Paper configuration: RGB + NIR (4 channels)
                return self._preprocess_s2_paper_config(image)
            else:
                # Enhanced configuration: All 12 bands
                return self._preprocess_s2_enhanced_config(image)

        except Exception as e:
            print(f"Error in preprocess_sentinel2: {str(e)}")
            return np.zeros((256, 256, self.s2_channels), dtype=np.uint8)

    def _preprocess_s2_paper_config(self, image):
        """Process Sentinel-2 for paper configuration (RGB + NIR)"""
        # Check if image has multiple channels
        if len(image.shape) > 2 and image.shape[2] >= 3:
            # Extract RGB channels
            rgb = image[:, :, :3].copy()

            # Convert to uint8 if not already
            if rgb.dtype != np.uint8:
                for i in range(3):
                    rgb[:, :, i] = cv2.normalize(rgb[:, :, i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create a 4-channel output (RGB + NIR)
            output = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            output[:, :, :3] = rgb

            # Add a NIR channel (use red channel as approximation if not available)
            if image.shape[2] >= 4:
                nir = image[:, :, 3]
                if nir.dtype != np.uint8:
                    nir = cv2.normalize(nir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                output[:, :, 3] = nir
            else:
                output[:, :, 3] = rgb[:, :, 0]  # Use red channel as NIR approximation

            return output
        else:
            # Single channel image - convert to 4 channels
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create a 4-channel output (grayscale repeated in RGB + NIR)
            output = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            output[:, :, 0] = image  # R
            output[:, :, 1] = image  # G
            output[:, :, 2] = image  # B
            output[:, :, 3] = image  # NIR

            return output

    def _preprocess_s2_enhanced_config(self, image):
        """Process Sentinel-2 for enhanced configuration (12 bands)"""
        if len(image.shape) > 2 and image.shape[2] >= 3:
            # If we have at least 3 channels, create 12 channels
            output = np.zeros((image.shape[0], image.shape[1], 12), dtype=np.uint8)

            # Copy available channels
            available_channels = min(image.shape[2], 12)
            for i in range(available_channels):
                if image[:, :, i].dtype != np.uint8:
                    output[:, :, i] = cv2.normalize(image[:, :, i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    output[:, :, i] = image[:, :, i]

            # Fill remaining channels with duplicates of existing channels
            for i in range(available_channels, 12):
                output[:, :, i] = output[:, :, i % available_channels]

            return output
        else:
            # Single channel image - replicate to 12 channels
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            output = np.zeros((image.shape[0], image.shape[1], 12), dtype=np.uint8)
            for i in range(12):
                output[:, :, i] = image

            return output



def create_combined_mask(s1_path, s2_path):
    """
    Create optimized binary mask by combining Sentinel-1 and Sentinel-2 information
    with advanced feature extraction and fusion techniques
    """
    # Create preprocessor instance
    preprocessor = DataPreprocessor()

    # Read and preprocess Sentinel-1
    s1_img = cv2.imread(str(s1_path), cv2.IMREAD_GRAYSCALE)
    if s1_img is None:
        print(f"Error: Could not read Sentinel-1 image: {s1_path}")
        return None
    s1_processed = preprocessor.preprocess_sentinel1(s1_img)

    # Read and preprocess Sentinel-2
    s2_img = cv2.imread(str(s2_path))
    if s2_img is None:
        print(f"Error: Could not read Sentinel-2 image: {s2_path}")
        return None

    # Process Sentinel-2 image
    s2_processed = preprocessor.preprocess_sentinel2(s2_img)

    # Advanced feature extraction for Sentinel-1
    # Use adaptive thresholding instead of global Otsu for better local detail
    s1_adaptive = cv2.adaptiveThreshold(
        s1_processed.astype(np.uint8),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # Constant subtracted from mean
    )

    # Edge detection for better boundary definition
    s1_edges = cv2.Canny(s1_processed.astype(np.uint8), 50, 150)

    # Combine adaptive threshold and edges for S1 features
    s1_features = cv2.bitwise_or(s1_adaptive, s1_edges)

    # Advanced feature extraction for Sentinel-2
    if len(s2_processed.shape) > 2:
        # If we have the burn index channel (4th channel)
        if s2_processed.shape[2] >= 4:
            # Use the burn index directly
            burn_index = s2_processed[:, :, 3]

            # Threshold the burn index with Otsu's method
            _, s2_burn_mask = cv2.threshold(
                burn_index,
                0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Convert RGB to grayscale for texture analysis
            s2_gray = cv2.cvtColor(s2_processed[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            # Just use grayscale conversion
            s2_gray = cv2.cvtColor(s2_processed.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            s2_burn_mask = np.zeros_like(s2_gray)
    else:
        s2_gray = s2_processed.astype(np.uint8)
        s2_burn_mask = np.zeros_like(s2_gray)

    # Apply adaptive thresholding to S2 grayscale
    s2_adaptive = cv2.adaptiveThreshold(
        s2_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Combine S2 adaptive threshold with burn mask if available
    if np.sum(s2_burn_mask) > 0:
        s2_features = cv2.bitwise_or(s2_adaptive, s2_burn_mask)
    else:
        s2_features = s2_adaptive

    # Advanced mask fusion with morphological operations
    # Create initial combined mask with weighted addition - Give more importance to Sentinel-2
    combined_mask = cv2.addWeighted(s1_features, 0.3, s2_features, 0.7, 0)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Final thresholding with hysteresis to reduce noise
    _, mask_low = cv2.threshold(combined_mask, 100, 255, cv2.THRESH_BINARY)
    _, mask_high = cv2.threshold(combined_mask, 150, 255, cv2.THRESH_BINARY)

    # Use high threshold mask as seeds and grow regions using low threshold mask
    final_mask = cv2.bitwise_and(mask_low, cv2.dilate(mask_high, kernel, iterations=2))

    return final_mask

def process_image_pairs(s1_dir, s2_dir, output_dir, batch_size=50, resume_from=None):
    """
    Enhanced processing of Sentinel-1 and Sentinel-2 image pairs to create masks
    with batch processing and resume capability
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of Sentinel-1 images
    s1_files = list(Path(s1_dir).glob('*.png'))
    s2_files = list(Path(s2_dir).glob('*.png'))

    print(f"Found {len(s1_files)} Sentinel-1 images")
    print(f"Found {len(s2_files)} Sentinel-2 images")

    # Create a mapping for faster lookup
    s2_mapping = {}
    for s2_file in s2_files:
        s1_name = s2_file.name.replace('_s2_', '_s1_')
        s2_mapping[s1_name] = s2_file

    # Process only matching pairs
    matching_pairs = []
    for s1_file in s1_files:
        if s1_file.name in s2_mapping:
            matching_pairs.append((s1_file, s2_mapping[s1_file.name]))

    print(f"Found {len(matching_pairs)} matching pairs")

    # Check for existing masks to resume processing
    existing_masks = set()
    if resume_from is not None:
        existing_masks = {p.name for p in Path(output_dir).glob('*.png')}
        print(f"Found {len(existing_masks)} existing masks, resuming from where left off")

    processed_count = 0
    error_count = 0
    error_files = []
    skipped_count = 0

    # Process in batches for better memory management
    for i in range(0, len(matching_pairs), batch_size):
        batch = matching_pairs[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(matching_pairs) + batch_size - 1)//batch_size}")

        for s1_path, s2_path in batch:
            try:
                # Skip if mask already exists and we're resuming
                mask_name = s1_path.name
                if resume_from and mask_name in existing_masks:
                    skipped_count += 1
                    continue

                # Create combined mask using both paths
                mask = create_combined_mask(str(s1_path), str(s2_path))
                if mask is None:
                    print(f"Error: Could not create mask for pair: {s1_path.name}")
                    error_files.append(str(s1_path))
                    error_count += 1
                    continue

                # Save mask
                output_path = Path(output_dir) / mask_name
                cv2.imwrite(str(output_path), mask)
                processed_count += 1

                if processed_count % 10 == 0:
                    print(f'Processed {processed_count} masks...')

            except Exception as e:
                print(f"Error processing {s1_path.name}: {str(e)}")
                error_files.append(str(s1_path))
                error_count += 1

    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed_count} pairs")
    print(f"Errors encountered: {error_count} pairs")

    if error_count > 0:
        print("\nFiles with errors:")
        for f in error_files:
            print(f"- {f}")

def visualize_results(s1_path, s2_path, mask_path):
    """Visualize original images and generated mask"""
    s1_img = cv2.imread(str(s1_path), cv2.IMREAD_GRAYSCALE)
    s2_img = cv2.imread(str(s2_path))  # Read RGB channels
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Convert grayscale to 3-channel BGR for consistent dimensions
    s1_vis = cv2.cvtColor(s1_img, cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Create visualization (all images now have 3 channels)
    fig = np.hstack([
        s1_vis,                    # Sentinel-1 (3 channels)
        s2_img[:, :, [2,1,0]],     # Sentinel-2 RGB (3 channels)
        mask_vis                    # Mask (3 channels)
    ])

    # Save the visualization
    output_path = os.path.join(os.path.dirname(mask_path), 'visualization.png')
    cv2.imwrite(output_path, fig)
    print(f'Visualization saved to: {output_path}')

def count_files():
    s1_dir = 'e:\\Major Project\\data\\sentinel1'
    s2_dir = 'e:\\Major Project\\data\\sentinel2'
    mask_dir = 'e:\\Major Project\\data\\masks'

    # Check if directories exist
    if not os.path.exists(s1_dir):
        print(f"Error: Sentinel-1 directory not found: {s1_dir}")
        return
    if not os.path.exists(s2_dir):
        print(f"Error: Sentinel-2 directory not found: {s2_dir}")
        return
    if not os.path.exists(mask_dir):
        print(f"Error: Masks directory not found: {mask_dir}")
        return

    s1_count = len(list(Path(s1_dir).glob('*.png')))
    s2_count = len(list(Path(s2_dir).glob('*.png')))
    mask_count = len(list(Path(mask_dir).glob('*.png')))

    print("\n=== File Count Summary ===")
    print("-------------------------")
    print(f"Sentinel-1 images: {s1_count:5d}")
    print(f"Sentinel-2 images: {s2_count:5d}")
    print(f"Mask files:       {mask_count:5d}")
    print("-------------------------")

if __name__ == '__main__':
    s1_dir = 'e:\\Major Project\\data\\sentinel1'
    s2_dir = 'e:\\Major Project\\data\\sentinel2'
    output_dir = 'e:\\Major Project\\data\\masks'

    # Process all image pairs
    process_image_pairs(s1_dir, s2_dir, output_dir)

    # Visualize first pair of images and their mask
    first_s1 = next(Path(s1_dir).glob('*.png'))
    first_s2 = Path(s2_dir) / first_s1.name.replace('_s1_', '_s2_')
    first_mask = Path(output_dir) / first_s1.name

    # Print paths for debugging
    print(f"\nAttempting to visualize:")
    print(f"S1: {first_s1}")
    print(f"S2: {first_s2}")
    print(f"Mask: {first_mask}")

    visualize_results(first_s1, first_s2, first_mask)

    # Show final file counts
    count_files()