# inference.py
import torch
import torch.nn.functional as F
import numpy as np
from osgeo import gdal
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# --- Import our custom modules ---
from model import CSRN

# --- 1. Dynamic Path Handling ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODEL_DIR)

# --- 2. Configuration ---
MODEL_PATH = os.path.join(MODEL_DIR, "checkpoints", "best_model.pth")
DATA_ROOT = os.path.join(PROJECT_ROOT, "preprocessing", "data")
SAMPLE_TIF_PATH = os.path.join(PROJECT_ROOT, "preprocessing", "data", "0000999", "all_bands.tif") # Your test scene
OUTPUT_IMAGE_PATH = os.path.join(MODEL_DIR, "inference_output_30m.tif")
OUTPUT_PLOT_PATH = os.path.join(MODEL_DIR, "inference_comparison.png")

# ---
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#  CRITICAL: Make sure these are your correct stats from calculate_stats.py
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
LST_MEAN = 342.4161
LST_STD = 14.0398
OLI_MEAN = np.array([0.3258, 0.2833, 0.2652, 0.2462, 0.7328, 0.5324, 0.3284], dtype=np.float32)
OLI_STD = np.array([0.1087, 0.1198, 0.1335, 0.1744, 0.2096, 0.2286, 0.2205], dtype=np.float32)
EPS_MEAN = 0.9830
EPS_STD = 0.0087
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ---

# --- Preprocessing Constants ---
RADIANCE_MULT_B10 = 3.3420E-04
RADIANCE_ADD_B10 = 0.1
K1_CONSTANT_B10 = 774.8853
K2_CONSTANT_B10 = 1321.0789
OLI_BANDS_IDX = [1, 2, 3, 4, 5, 6, 7]
OLI_RED_BAND_IDX = 4
OLI_NIR_BAND_IDX = 5
OLI_PAN_BAND_IDX = 8
TIRS_B10_BAND_IDX = 10

# --- Patch size (must match training) ---
PATCH_SIZE_HR = 72
PATCH_SIZE_LR = PATCH_SIZE_HR // 3

# --- Helper Functions ---
def get_band_data(gdal_dataset, band_index):
    band = gdal_dataset.GetRasterBand(band_index)
    return band.ReadAsArray()

def dn_to_bt(tirs_dn):
    tirs_dn_16bit = tirs_dn.astype(np.float32) * 256.0
    radiance = tirs_dn_16bit * RADIANCE_MULT_B10 + RADIANCE_ADD_B10
    with np.errstate(divide='ignore', invalid='ignore'):
        bt = K2_CONSTANT_B10 / np.log((K1_CONSTANT_B10 / radiance) + 1)
    bt[radiance <= 0] = 0
    return bt

def calculate_ndvi(red_dn, nir_dn):
    red_toa = red_dn.astype(np.float32) / 255.0
    nir_toa = nir_dn.astype(np.float32) / 255.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_toa - red_toa) / (nir_toa + red_toa)
    ndvi[np.isnan(ndvi)] = 0
    return ndvi

def calculate_emissivity(ndvi):
    ndvi_min, ndvi_max = 0.2, 0.5
    pv = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min))**2
    pv[ndvi < ndvi_min] = 0.0
    pv[ndvi > ndvi_max] = 1.0
    return 0.99 * pv + 0.97 * (1 - pv)

def align_image(src_img, ref_img):
    ref_norm = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
    src_norm = (src_img - src_img.min()) / (src_img.max() - src_img.min())
    ref_norm = ref_norm.astype(np.float32)
    src_norm = src_norm.astype(np.float32)
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-5)
    try:
        (cc, warp_matrix) = cv2.findTransformECC(ref_norm, src_norm, warp_matrix, warp_mode, criteria, None, 5)
    except cv2.error:
        pass
    rows, cols = ref_img.shape
    return cv2.warpAffine(src_img, warp_matrix, (cols, rows), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

# ---
# vvv MODIFIED FUNCTION vvv
# ---
def save_as_geotiff(array, original_gdal_ds, output_path, crop_shape):
    """Saves a NumPy array as a GeoTIFF, copying georeferencing
       and calculating statistics for proper visualization.
    """
    print(f"Saving GeoTIFF to {output_path}...")
    driver = gdal.GetDriverByName('GTiff')
    h, w = array.shape
    out_ds = driver.Create(output_path, w, h, 1, gdal.GDT_Float32)
    
    gt = original_gdal_ds.GetGeoTransform()
    cropped_gt = (gt[0], gt[1], gt[2], gt[3], gt[4], gt[5])
    
    out_ds.SetGeoTransform(cropped_gt)
    out_ds.SetProjection(original_gdal_ds.GetProjection())
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(0) # Assume 0 K is NoData
    
    # --- NEW: Calculate stats for visualization ---
    # This is what makes QGIS display it correctly on load
    print("Calculating statistics for GeoTIFF...")
    out_band.ComputeStatistics(False) 
    # ---
    
    out_band.FlushCache()
    out_ds = None # Close file
    print("GeoTIFF saved.")
# ---
# ^^^ END MODIFIED FUNCTION ^^^
# ---

def main():
    # --- 1. Setup ---
    gdal.UseExceptions()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    oli_mean_bc = OLI_MEAN.reshape(7, 1, 1)
    oli_std_bc = OLI_STD.reshape(7, 1, 1)

    # --- 2. Load Model ---
    print(f"Loading model from {MODEL_PATH}...")
    model = CSRN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model loaded.")

    # --- 3. Load and Preprocess Full Scene ---
    print(f"Loading sample file: {SAMPLE_TIF_PATH}")
    gdal_ds = gdal.Open(SAMPLE_TIF_PATH)
    
    oli_data = [get_band_data(gdal_ds, idx) for idx in OLI_BANDS_IDX]
    oli_30m = np.stack(oli_data, axis=0).astype(np.float32)
    
    tirs_b10_unaligned = get_band_data(gdal_ds, TIRS_B10_BAND_IDX).astype(np.float32)
    pan_ref = get_band_data(gdal_ds, OLI_PAN_BAND_IDX).astype(np.float32)
    red_dn = get_band_data(gdal_ds, OLI_RED_BAND_IDX)
    nir_dn = get_band_data(gdal_ds, OLI_NIR_BAND_IDX)
    
    tirs_b10_aligned = align_image(tirs_b10_unaligned, pan_ref)
    ndvi_30m = calculate_ndvi(red_dn, nir_dn)
    epsilon_30m = calculate_emissivity(ndvi_30m).astype(np.float32)
    
    lst_30m_target_full = dn_to_bt(tirs_b10_aligned)
    
    # --- 4. Create Model Inputs (Degradation + Cropping) ---
    h, w = lst_30m_target_full.shape
    h_lr, w_lr = h // 3, w // 3
    
    h_crop, w_crop = h_lr * 3, w_lr * 3
    
    lst_30m_target = lst_30m_target_full[:h_crop, :w_crop]
    oli_30m = oli_30m[:, :h_crop, :w_crop]
    epsilon_30m = epsilon_30m[:h_crop, :w_crop]

    lst_100m_input = lst_30m_target.reshape(h_lr, 3, w_lr, 3).mean(axis=(1, 3))
    
    # --- 5. Normalization and Tensor Conversion ---
    print("Normalizing data and preparing tensors...")
    
    oli_30m_reflectance = oli_30m / 255.0
    
    lr_lst_norm = (lst_100m_input - LST_MEAN) / LST_STD
    oli_norm = (oli_30m_reflectance - oli_mean_bc) / oli_std_bc
    eps_norm = (epsilon_30m - EPS_MEAN) / EPS_STD
    
    guide_norm = np.concatenate([oli_norm, np.expand_dims(eps_norm, axis=0)], axis=0)
    
    lr_lst_tensor = torch.from_numpy(lr_lst_norm).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    hr_guide_tensor = torch.from_numpy(guide_norm).unsqueeze(0).to(device, dtype=torch.float32)
    
    hr_target_norm_tensor = (torch.from_numpy(lst_30m_target) - LST_MEAN) / LST_STD
    hr_target_norm_tensor = hr_target_norm_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

    # --- 6. Run Tiled Inference ---
    print("Running tiled model inference...")
    B, C, H_lr, W_lr = lr_lst_tensor.shape
    _, _, H_hr, W_hr = hr_guide_tensor.shape
    pred_hr_norm = torch.zeros(B, 1, H_hr, W_hr).to(device)

    for h_start_lr in tqdm(range(0, H_lr, PATCH_SIZE_LR), desc="Inference Tiles"):
        for w_start_lr in range(0, W_lr, PATCH_SIZE_LR):
            
            h_end_lr = min(h_start_lr + PATCH_SIZE_LR, H_lr)
            w_end_lr = min(w_start_lr + PATCH_SIZE_LR, W_lr)
            lr_patch = lr_lst_tensor[:, :, h_start_lr:h_end_lr, w_start_lr:w_end_lr]
            h_pad_lr = PATCH_SIZE_LR - lr_patch.shape[2]
            w_pad_lr = PATCH_SIZE_LR - lr_patch.shape[3]
            lr_patch_padded = F.pad(lr_patch, (0, w_pad_lr, 0, h_pad_lr), mode='reflect')
            
            h_start_hr = h_start_lr * 3
            w_start_hr = w_start_lr * 3
            h_end_hr = min(h_start_hr + PATCH_SIZE_HR, H_hr)
            w_end_hr = min(w_start_hr + PATCH_SIZE_HR, W_hr)
            hr_patch = hr_guide_tensor[:, :, h_start_hr:h_end_hr, w_start_hr:w_end_hr]
            h_pad_hr = PATCH_SIZE_HR - hr_patch.shape[2]
            w_pad_hr = PATCH_SIZE_HR - hr_patch.shape[3]
            hr_patch_padded = F.pad(hr_patch, (0, w_pad_hr, 0, h_pad_hr), mode='reflect')

            with torch.no_grad():
                pred_patch_norm = model(lr_patch_padded, hr_patch_padded)
            
            h_unpad_hr = h_end_hr - h_start_hr
            w_unpad_hr = w_end_hr - w_start_hr
            pred_hr_norm[:, :, h_start_hr:h_end_hr, w_start_hr:w_end_hr] = \
                pred_patch_norm[:, :, :h_unpad_hr, :w_unpad_hr]

    print("Inference complete.")

    # --- 7. Post-processing (De-normalization) ---
    pred_hr_kelvin = (pred_hr_norm * LST_STD) + LST_MEAN
    
    pred_hr_np = pred_hr_kelvin.squeeze().cpu().numpy()
    hr_target_np = (hr_target_norm_tensor * LST_STD + LST_MEAN).squeeze().cpu().numpy()
    lr_input_np = (lr_lst_tensor * LST_STD + LST_MEAN).squeeze().cpu().numpy()

    # --- 8. Calculate Metrics ---
    print("--- Quantitative Metrics ---")
    
    rmse = np.sqrt(np.mean((pred_hr_np - hr_target_np)**2))
    print(f"RMSE: {rmse:.4f} K")
    
    data_range_norm = hr_target_norm_tensor.max() - hr_target_norm_tensor.min()
    data_range_norm = data_range_norm.item()
    if data_range_norm == 0: data_range_norm = 1.0
    
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range_norm).to(device)
    psnr = psnr_metric(pred_hr_norm, hr_target_norm_tensor)
    print(f"PSNR: {psnr.item():.4f} dB")
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range_norm).to(device)
    ssim = ssim_metric(pred_hr_norm, hr_target_norm_tensor)
    print(f"SSIM: {ssim.item():.4f}")

    # --- 9. Save GeoTIFF Output ---
    save_as_geotiff(pred_hr_np, gdal_ds, OUTPUT_IMAGE_PATH, (h_crop, w_crop))
    
    # --- 
    # --- 10. Save Comparison Plot --- (MODIFIED)
    # ---
    print(f"Saving comparison plot to {OUTPUT_PLOT_PATH}...")
    
    lr_input_upscaled = cv2.resize(
        lr_input_np, 
        (w_crop, h_crop), 
        interpolation=cv2.INTER_NEAREST
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8),layout='constrained')
    
    # --- NEW: Better vmin/vmax for plotting ---
    # Filter out 0 K values before getting percentiles
    valid_temps = hr_target_np[hr_target_np > 150]
    vmin = np.percentile(valid_temps, 5) # Use 5th percentile
    vmax = np.percentile(valid_temps, 95) # Use 95th percentile
    # ---
    
    axes[0].imshow(lr_input_upscaled, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Input LST (100m, Nearest Neighbor)")
    
    im = axes[1].imshow(pred_hr_np, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicted LST (30m) - RMSE: {rmse:.4f} K")
    
    axes[2].imshow(hr_target_np, cmap='hot', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Ground Truth LST (30m)")
    
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Temperature (K)")
    fig.suptitle(f"Inference on Scene: {os.path.basename(os.path.dirname(SAMPLE_TIF_PATH))}", fontsize=16)
    
        
    plt.savefig(OUTPUT_PLOT_PATH)
    print("Plot saved.")
    print("\n--- Inference Complete ---")

if __name__ == "__main__":
    main()