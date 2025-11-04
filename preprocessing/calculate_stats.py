import numpy as np
import os
import glob
from tqdm import tqdm

def calculate_normalization_stats(processed_dir=os.path.join("..", "preprocessing", "processed_data", "train")):
    """
    Calculates the mean and standard deviation for all processed .npy files.
    This uses a running "sum" and "sum of squares" to avoid loading all
    data into memory, which is a numerically stable approach.
    """
    
    scene_dirs = sorted(glob.glob(os.path.join(processed_dir, "*")))
    if not scene_dirs:
        print(f"Error: No processed data found in {processed_dir}. Run preprocess.py first.")
        return

    print(f"Calculating statistics over {len(scene_dirs)} scenes...")

    # --- LST (Input and Target) ---
    # We use the same stats for both input and target,
    # so we'll accumulate from both files.
    lst_sum = 0.0
    lst_sq_sum = 0.0
    lst_count = 0

    # --- OLI Guide (Per-Channel) ---
    # Our OLI guide has 7 channels (Bands 1-7)
    oli_sum = np.zeros(7, dtype=np.float64)
    oli_sq_sum = np.zeros(7, dtype=np.float64)
    oli_count = 0

    # --- Emissivity Guide (Single Channel) ---
    eps_sum = 0.0
    eps_sq_sum = 0.0
    eps_count = 0

    for scene_dir in tqdm(scene_dirs, desc="Calculating Stats"):
        try:
            # --- LST Stats ---
            lr_lst = np.load(os.path.join(scene_dir, "lst_100m_input.npy"))
            hr_target = np.load(os.path.join(scene_dir, "lst_30m_target.npy"))
            
            # Add stats from both LR input and HR target
            lst_sum += np.sum(lr_lst)
            lst_sq_sum += np.sum(lr_lst**2)
            lst_count += lr_lst.size
            
            lst_sum += np.sum(hr_target)
            lst_sq_sum += np.sum(hr_target**2)
            lst_count += hr_target.size

            # --- OLI Stats ---
            oli_guide = np.load(os.path.join(scene_dir, "oli_30m_guide.npy")) # Shape [7, H, W]
            
            # Calculate sum and sq_sum per channel
            oli_sum += np.sum(oli_guide, axis=(1, 2))
            oli_sq_sum += np.sum(oli_guide**2, axis=(1, 2))
            oli_count += oli_guide.shape[1] * oli_guide.shape[2]

            # --- Emissivity Stats ---
            eps_guide = np.load(os.path.join(scene_dir, "eps_30m_guide.npy")) # Shape [H, W]
            
            eps_sum += np.sum(eps_guide)
            eps_sq_sum += np.sum(eps_guide**2)
            eps_count += eps_guide.size

        except Exception as e:
            print(f"Warning: Failed to load data from {scene_dir}. Skipping. Error: {e}")

    # --- Final Calculation ---
    
    print("\n--- Statistics Calculation Complete ---")

    # LST
    lst_mean = lst_sum / lst_count
    lst_std = np.sqrt((lst_sq_sum / lst_count) - (lst_mean**2))
    print(f"\n# LST (Brightness Temperature in K)")
    print(f"# Copy these into dataset.py")
    print(f"self.LST_MEAN = {lst_mean:.4f}")
    print(f"self.LST_STD = {lst_std:.4f}")

    # OLI
    oli_mean = oli_sum / oli_count
    oli_std = np.sqrt((oli_sq_sum / oli_count) - (oli_mean**2))
    print(f"\n# OLI (Reflectance, 7 channels)")
    print(f"# Copy these into dataset.py")
    print(f"self.OLI_MEAN = np.array({np.array2string(oli_mean, precision=4, separator=', ')}, dtype=np.float32)")
    print(f"self.OLI_STD = np.array({np.array2string(oli_std, precision=4, separator=', ')}, dtype=np.float32)")

    # Emissivity
    eps_mean = eps_sum / eps_count
    eps_std = np.sqrt((eps_sq_sum / eps_count) - (eps_mean**2))
    print(f"\n# Emissivity (Unitless)")
    print(f"# Copy these into dataset.py")
    print(f"self.EPS_MEAN = {eps_mean:.4f}")
    print(f"self.EPS_STD = {eps_std:.4f}")

if __name__ == "__main__":
    calculate_normalization_stats()