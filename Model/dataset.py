# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import random

class ThermalSRDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed LST data.
    Reads .npy files from the 'processed_data' directory.
    """
    def __init__(self, processed_dir="/preprocessing/processed_data", patch_size=96, augment=True):
        """
        Args:
            processed_dir (str): Directory containing the processed scene folders.
            patch_size (int): The size of the high-resolution (30m) patch to extract (e.g., 96x96).
                              The low-res patch will be patch_size / 3.
            augment (bool): Whether to apply random horizontal/vertical flips.
        """
        self.patch_size = patch_size
        self.augment = augment
        
        if (patch_size % 3) != 0:
            raise ValueError("patch_size must be divisible by 3 (the upscale factor).")
            
        self.lr_patch_size = patch_size // 3
        
        # Split data into train/val
        all_scene_dirs = sorted(glob.glob(os.path.join(processed_dir, "*")))
        if not all_scene_dirs:
            raise FileNotFoundError(f"No processed data found in {processed_dir}. Run preprocess.py first.")
            
        # Use first 90% for training, last 10% for validation
        # This assumes you are passing 'train' or 'val' to the constructor
        # For simplicity, we'll just use all 500 for now.
        self.scene_dirs = all_scene_dirs
            
        # --- D. Normalization ---

        # LST (Brightness Temperature in K)
        # Copy these into dataset.py
        self.LST_MEAN = 343.7189
        self.LST_STD = 14.3151

        # OLI (Reflectance, 7 channels)
        # Copy these into dataset.py
        self.OLI_MEAN = np.array([0.3277, 0.2869, 0.2712, 0.2586, 0.7287, 0.5524, 0.3482], dtype=np.float32)
        self.OLI_STD = np.array([0.1012, 0.113 , 0.1294, 0.175 , 0.2042, 0.2299, 0.2261], dtype=np.float32)

        # Emissivity (Unitless)
        # Copy these into dataset.py
        self.EPS_MEAN = 0.9823
        self.EPS_STD = 0.0090
        # ---------------------------------------------------------
        
        # --- Reshape stats for broadcasting ---
        # This allows us to normalize (tensor - mean) / std
        # [7, 1, 1] will broadcast over a [7, H, W] tensor
        self.OLI_MEAN = self.OLI_MEAN.reshape(7, 1, 1)
        self.OLI_STD = self.OLI_STD.reshape(7, 1, 1)

    def __len__(self):
        # Return the number of scenes
        return len(self.scene_dirs)

    def __getitem__(self, index):
        scene_dir = self.scene_dirs[index]
        
        try:
            lr_lst = np.load(os.path.join(scene_dir, "lst_100m_input.npy"), mmap_mode='r')
            hr_oli = np.load(os.path.join(scene_dir, "oli_30m_guide.npy"), mmap_mode='r')
            hr_eps = np.load(os.path.join(scene_dir, "eps_30m_guide.npy"), mmap_mode='r')
            hr_target = np.load(os.path.join(scene_dir, "lst_30m_target.npy"), mmap_mode='r')
        except FileNotFoundError:
            return self.__getitem__(np.random.randint(0, len(self.scene_dirs)))

        if hr_eps.ndim == 2:
            hr_eps = np.expand_dims(hr_eps, axis=0) # [1, H, W]
        if hr_target.ndim == 2:
            hr_target = np.expand_dims(hr_target, axis=0) # [1, H, W]
            
        hr_h, hr_w = hr_oli.shape[1], hr_oli.shape[2]
        
        # Check if image is smaller than patch size
        if hr_h < self.patch_size or hr_w < self.patch_size:
            # Handle small images, e.g., by resizing or skipping
            # For now, just take a random patch from a different image
            return self.__getitem__(np.random.randint(0, len(self.scene_dirs)))
            
        rand_h = np.random.randint(0, hr_h - self.patch_size + 1)
        rand_w = np.random.randint(0, hr_w - self.patch_size + 1)
        
        hr_oli_patch = hr_oli[:, rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size]
        hr_eps_patch = hr_eps[:, rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size]
        hr_target_patch = hr_target[:, rand_h:rand_h + self.patch_size, rand_w:rand_w + self.patch_size]

        lr_h_start, lr_w_start = rand_h // 3, rand_w // 3
        lr_lst_patch = lr_lst[lr_h_start:lr_h_start + self.lr_patch_size, lr_w_start:lr_w_start + self.lr_patch_size]
        lr_lst_patch = np.expand_dims(lr_lst_patch, axis=0) # [1, H/3, W/3]

        # Combine all 30m guides
        # [7 OLI channels] + [1 Emissivity channel]
        # We will normalize OLI and EPS separately before concatenating
        
        # --- Convert to Tensors ---
        # We do normalization in NumPy (float32) before converting
        
        # --- Apply Normalization ---
        lr_lst_norm = (lr_lst_patch - self.LST_MEAN) / self.LST_STD
        hr_target_norm = (hr_target_patch - self.LST_MEAN) / self.LST_STD
        hr_oli_norm = (hr_oli_patch - self.OLI_MEAN) / self.OLI_STD
        hr_eps_norm = (hr_eps_patch - self.EPS_MEAN) / self.EPS_STD

        # Concatenate normalized guides
        hr_guide_norm = np.concatenate([hr_oli_norm, hr_eps_norm], axis=0).astype(np.float32)

        # Convert final tensors
        lr_lst = torch.from_numpy(lr_lst_norm.copy())
        hr_guide = torch.from_numpy(hr_guide_norm.copy())
        hr_target = torch.from_numpy(hr_target_norm.copy())

        # --- Data Augmentation ---
        if self.augment:
            if random.random() > 0.5: # Horizontal flip
                lr_lst = torch.fliplr(lr_lst)
                hr_guide = torch.fliplr(hr_guide)
                hr_target = torch.fliplr(hr_target)
            if random.random() > 0.5: # Vertical flip
                lr_lst = torch.flipud(lr_lst)
                hr_guide = torch.flipud(hr_guide)
                hr_target = torch.flipud(hr_target)

        return {
            "lr_lst": lr_lst,       # [1, H/3, W/3]
            "hr_guide": hr_guide,   # [8, H, W]  <-- Note: now 8 channels
            "hr_target": hr_target  # [1, H, W]
        }