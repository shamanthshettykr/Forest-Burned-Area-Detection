import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random
from data_preprocessing import DataPreprocessor
import albumentations as A

class DualSentinelDataset(Dataset):
    def __init__(self, s1_paths, s2_paths, mask_paths, transform=True, cache_size=0, input_size=(256, 256)):
        # Validate paths before creating dataset
        self.s1_paths = []
        self.s2_paths = []
        self.mask_paths = []
        self.input_size = input_size

        # Create a simple cache to store preprocessed images
        self.cache = {}
        self.cache_size = min(cache_size, len(s1_paths)) if cache_size > 0 else 0
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"Validating {len(s1_paths)} image pairs...")
        valid_count = 0

        for s1, s2, mask in zip(s1_paths, s2_paths, mask_paths):
            try:
                # Convert to Path objects if they're strings
                from pathlib import Path
                s1_path = Path(s1) if isinstance(s1, str) else s1
                s2_path = Path(s2) if isinstance(s2, str) else s2
                mask_path = Path(mask) if isinstance(mask, str) else mask

                # Quick validation check
                if not (s1_path.exists() and s2_path.exists() and mask_path.exists()):
                    continue

                self.s1_paths.append(str(s1_path))
                self.s2_paths.append(str(s2_path))
                self.mask_paths.append(str(mask_path))
                valid_count += 1

            except Exception as e:
                print(f"Error validating {s1}: {str(e)}")
                continue

        print(f"Found {valid_count} valid image pairs out of {len(s1_paths)}")

        self.transform = transform
        self.preprocessor = DataPreprocessor()

        # Use fixed normalization values instead of computing them
        self.s1_mean, self.s1_std = 0.5, 0.5
        self.s2_mean = [0.5] * 12  # One mean value per band
        self.s2_std = [0.5] * 12   # One std value per band

        # Simple augmentation for training
        if transform:
            self.augmentation = A.Compose([
                # Basic geometric transformations
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                # Always resize to target size at the end
                A.Resize(height=input_size[0], width=input_size[1], p=1.0),
            ])

            # Separate augmentation for validation (only resize)
            self.val_transform = A.Compose([
                A.Resize(height=input_size[0], width=input_size[1], p=1.0),
            ])

    def __len__(self):
        return len(self.s1_paths)  # Return the number of valid image pairs

    def __getitem__(self, idx):
        # Check if the sample is in the cache
        if idx in self.cache:
            self.cache_hits += 1
            return self.cache[idx]

        self.cache_misses += 1

        try:
            # Load images with error handling
            s1_img = cv2.imread(str(self.s1_paths[idx]), cv2.IMREAD_GRAYSCALE)
            s2_img = cv2.imread(str(self.s2_paths[idx]))
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

            # Check if any image failed to load
            if s1_img is None or s2_img is None or mask is None:
                print(f"Warning: Failed to load images at index {idx}")
                # Return a dummy sample
                s1_tensor = torch.zeros((1, self.input_size[0], self.input_size[1]), dtype=torch.float32)
                s2_tensor = torch.zeros((12, self.input_size[0], self.input_size[1]), dtype=torch.float32)
                mask_tensor = torch.zeros((1, self.input_size[0], self.input_size[1]), dtype=torch.float32)
                return s1_tensor, s2_tensor, mask_tensor

            # Apply preprocessing using DataPreprocessor
            s1_processed = self.preprocessor.preprocess_sentinel1(s1_img)

            # For now, treat Sentinel-2 as RGB and replicate to 12 channels
            if len(s2_img.shape) == 3 and s2_img.shape[2] == 3:
                # Replicate RGB to 12 channels
                s2_processed = np.repeat(s2_img, 4, axis=2)  # 3 * 4 = 12 channels
            else:
                s2_processed = self.preprocessor.preprocess_sentinel2(s2_img)

            # Convert to float32 and normalize to [0, 1]
            s1_img = s1_processed.astype(np.float32) / 255.0
            s2_img = s2_processed.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0

            # Resize all images to the same size
            s1_img = cv2.resize(s1_img, self.input_size)

            # Resize all 12 bands of s2_img
            s2_resized = np.zeros((self.input_size[1], self.input_size[0], 12), dtype=np.float32)
            for c in range(12):
                s2_resized[:, :, c] = cv2.resize(s2_img[:, :, c], self.input_size)
            s2_img = s2_resized

            # Resize mask
            mask = cv2.resize(mask, self.input_size)

            # Apply simple augmentation if enabled
            if self.transform:
                # Apply augmentation to s1_img
                if random.random() > 0.5:
                    s1_img = cv2.flip(s1_img, 1)  # Horizontal flip
                if random.random() > 0.5:
                    s1_img = cv2.flip(s1_img, 0)  # Vertical flip

                # Apply same augmentation to s2_img
                if random.random() > 0.5:
                    s2_img = cv2.flip(s2_img, 1)  # Horizontal flip
                if random.random() > 0.5:
                    s2_img = cv2.flip(s2_img, 0)  # Vertical flip

                # Apply same augmentation to mask
                if random.random() > 0.5:
                    mask = cv2.flip(mask, 1)  # Horizontal flip
                if random.random() > 0.5:
                    mask = cv2.flip(mask, 0)  # Vertical flip

            # Ensure proper dimensions (Channels First format)
            s1_img = np.expand_dims(s1_img, axis=0)  # [1, H, W]
            s2_img = s2_img.transpose(2, 0, 1)  # [12, H, W]

            # Binarize mask with threshold 0.5
            mask = (mask > 0.5).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)  # [1, H, W]

            # Convert to torch tensors
            s1_tensor = torch.from_numpy(s1_img).float()
            s2_tensor = torch.from_numpy(s2_img).float()
            mask_tensor = torch.from_numpy(mask).float()

            # Store result in cache if enabled
            result = (s1_tensor, s2_tensor, mask_tensor)
            if self.cache_size > 0 and len(self.cache) < self.cache_size:
                self.cache[idx] = result

            return result

        except Exception as e:
            print(f"Error processing sample at index {idx}: {str(e)}")
            # Return a dummy sample in case of error
            s1_tensor = torch.zeros((1, self.input_size[0], self.input_size[1]), dtype=torch.float32)
            s2_tensor = torch.zeros((12, self.input_size[0], self.input_size[1]), dtype=torch.float32)
            mask_tensor = torch.zeros((1, self.input_size[0], self.input_size[1]), dtype=torch.float32)
            return s1_tensor, s2_tensor, mask_tensor

    def get_cache_stats(self):
        """Return cache hit/miss statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }