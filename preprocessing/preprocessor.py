# preprocess.py
import numpy as np
from osgeo import gdal
import os
import cv2
import glob
from tqdm import tqdm

# --- Constants ---
# (From our EDA findings)
RADIANCE_MULT_B10 = 3.3420E-04
RADIANCE_ADD_B10 = 0.1
K1_CONSTANT_B10 = 774.8853
K2_CONSTANT_B10 = 1321.0789

# --- Band Indices (CRITICAL: Finalized from check_bands.py) ---
OLI_BANDS_IDX = [1, 2, 3, 4, 5, 6, 7] # Bands 1-7 (B,G,R,NIR,SWIR1,SWIR2, etc.)
OLI_RED_BAND_IDX = 4
OLI_NIR_BAND_IDX = 5
OLI_PAN_BAND_IDX = 8  # Using Pan band for alignment
TIRS_B10_BAND_IDX = 10
# TIRS_B11_BAND_IDX = 11 # Add if using split-window

# --- Data Directories ---
ROOT_DIR = "C:/Users/swast/OneDrive/Desktop/Case Study ML/data" # Point to your data dir
OUTPUT_DIR = "processed_data"

# --- Helper Functions (Updated) ---

def dn_to_bt(tirs_dn, mult=RADIANCE_MULT_B10, add=RADIANCE_ADD_B10, k1=K1_CONSTANT_B10, k2=K2_CONSTANT_B10):
    """ Scales 8-bit DN to 16-bit, then converts to BT in Kelvin. """
    tirs_dn_16bit = tirs_dn.astype(np.float32) * 256.0
    radiance = tirs_dn_16bit * mult + add
    with np.errstate(divide='ignore', invalid='ignore'):
        bt = k2 / np.log((k1 / radiance) + 1)
    bt[radiance <= 0] = 0
    return bt

def calculate_ndvi(red_dn, nir_dn):
    """ Calculates NDVI from 8-bit OLI DNs. """
    red_toa = red_dn.astype(np.float32) / 255.0
    nir_toa = nir_dn.astype(np.float32) / 255.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_toa - red_toa) / (nir_toa + red_toa)
    ndvi[np.isnan(ndvi)] = 0 # Use 0 for NaNs
    ndvi[np.isinf(ndvi)] = 0
    return ndvi

def calculate_emissivity(ndvi):
    """ Estimates 30m Emissivity (epsilon) using the VCM. """
    ndvi_min = 0.2
    ndvi_max = 0.5
    
    pv = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min))**2
    pv[ndvi < ndvi_min] = 0.0
    pv[ndvi > ndvi_max] = 1.0
    
    emissivity = 0.99 * pv + 0.97 * (1 - pv) # e_v=0.99, e_s=0.97
    return emissivity

def align_image(src_img, ref_img):
    """
    Aligns src_img (TIRS) to ref_img (OLI Pan) using ECC.
    Includes the robust try/except fallback.
    """
    ref_norm = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
    src_norm = (src_img - src_img.min()) / (src_img.max() - src_img.min())
    
    ref_norm = ref_norm.astype(np.float32)
    src_norm = src_norm.astype(np.float32)

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
        
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-5)

    try:
        (cc, warp_matrix) = cv2.findTransformECC(
            ref_norm, src_norm, warp_matrix, warp_mode, criteria, None, 5
        )
    except cv2.error:
        # This fallback is essential, as confirmed by our test.
        pass # Silently use the identity matrix

    rows, cols = ref_img.shape
    aligned_img = cv2.warpAffine(
        src_img,
        warp_matrix,
        (cols, rows),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return aligned_img, warp_matrix

def get_band_data(gdal_dataset, band_index):
    """Helper to read a band as a NumPy array."""
    band = gdal_dataset.GetRasterBand(band_index)
    return band.ReadAsArray()

# --- Main Preprocessing Loop ---
def process_all_files():
    # --- Tell GDAL to use exceptions ---
    gdal.UseExceptions()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    search_path = os.path.join(ROOT_DIR, "*/all_bands.tif")
    tif_files = sorted(glob.glob(search_path))
    
    # --- NEW: Limit processing to the first 500 files ---
    tif_files = tif_files[:500]
    # ----------------------------------------------------
    
    if not tif_files:
        print(f"Error: No files found at '{search_path}'. Check ROOT_DIR.")
        return

    print(f"Found {len(tif_files)} files to process (limited to 500).")

    for tif_path in tqdm(tif_files, desc="Preprocessing Scenes"):
        try:
            gdal_ds = gdal.Open(tif_path)

            # --- Extract OLI Bands (30m) ---
            oli_data = []
            for band_idx in OLI_BANDS_IDX:
                oli_data.append(get_band_data(gdal_ds, band_idx))
            
            # Stack and convert to Reflectance [0, 1]
            oli_30m = np.stack(oli_data, axis=0).astype(np.float32)
            oli_30m = oli_30m / 255.0
            
            # --- Extract TIRS & Pan Bands (for alignment) ---
            tirs_b10_30m_unaligned = get_band_data(gdal_ds, TIRS_B10_BAND_IDX).astype(np.float32)
            
            # Use OLI Pan (Band 8) as the alignment reference
            ref_img_30m_pan = get_band_data(gdal_ds, OLI_PAN_BAND_IDX).astype(np.float32)
            
            # --- Step C: Geometric Alignment ---
            tirs_b10_30m_aligned, _ = align_image(tirs_b10_30m_unaligned, ref_img_30m_pan)
            
            # --- Step B: Emissivity Estimation (from OLI DNs) ---
            oli_red_30m_dn = get_band_data(gdal_ds, OLI_RED_BAND_IDX)
            oli_nir_30m_dn = get_band_data(gdal_ds, OLI_NIR_BAND_IDX)
            
            ndvi_30m = calculate_ndvi(oli_red_30m_dn, oli_nir_30m_dn)
            epsilon_30m = calculate_emissivity(ndvi_30m).astype(np.float32)

            # --- Step A: Radiometric Conversion ---
            # This is our LST_30m_Target (using BT as placeholder)
            lst_30m_target = dn_to_bt(tirs_b10_30m_aligned)

            # --- Create the Low-Resolution (100m) Input ---
            # This is the 'Degradation' step. We use a 3x3 average pool.
            h, w = lst_30m_target.shape
            h_lr, w_lr = h // 3, w // 3
            
            # Crop to be divisible by 3
            lst_30m_target_cropped = lst_30m_target[:h_lr * 3, :w_lr * 3]
            
            # Reshape and average
            lst_100m_input = lst_30m_target_cropped.reshape(h_lr, 3, w_lr, 3).mean(axis=(1, 3))
            
            # --- Crop all 30m guides to match ---
            oli_30m_cropped = oli_30m[:, :h_lr * 3, :w_lr * 3]
            epsilon_30m_cropped = epsilon_30m[:h_lr * 3, :w_lr * 3]
            
            # --- Save the processed tensors ---
            scene_id = os.path.basename(os.path.dirname(tif_path))
            out_path_dir = os.path.join(OUTPUT_DIR, scene_id)
            os.makedirs(out_path_dir, exist_ok=True)
            
            np.save(os.path.join(out_path_dir, "lst_100m_input.npy"), lst_100m_input.astype(np.float32))
            np.save(os.path.join(out_path_dir, "oli_30m_guide.npy"), oli_30m_cropped.astype(np.float32))
            np.save(os.path.join(out_path_dir, "eps_30m_guide.npy"), epsilon_30m_cropped.astype(np.float32))
            np.save(os.path.join(out_path_dir, "lst_30m_target.npy"), lst_30m_target_cropped.astype(np.float32))

        except Exception as e:
            print(f"Failed to process {tif_path}: {e}")
            
    print("Preprocessing complete.")
    # TODO: Calculate and save normalization stats (mean/std)

if __name__ == "__main__":
    process_all_files()