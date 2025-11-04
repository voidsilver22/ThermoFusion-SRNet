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
SAMPLE_TIF_PATH = os.path.join(PROJECT_ROOT, "preprocessing", "data", "0007999", "all_bands.tif") # Your test scene
OUTPUT_IMAGE_PATH = os.path.join(MODEL_DIR, "inference_output_30m.tif")
OUTPUT_PLOT_PATH = os.path.join(MODEL_DIR, "inference_comparison.png")
OUTPUT_DIAG_PATH = os.path.join(MODEL_DIR, "inference_diagnostics.png") # <-- NEW PLOT

# ---
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
#  CRITICAL: Make sure these are your correct stats from calculate_stats.py
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
LST_MEAN = 343.7189
LST_STD = 14.3151


OLI_MEAN = np.array([0.3277, 0.2869, 0.2712, 0.2586, 0.7287, 0.5524, 0.3482], dtype=np.float32)
OLI_STD = np.array([0.1012, 0.113 , 0.1294, 0.175 , 0.2042, 0.2299, 0.2261], dtype=np.float32)

EPS_MEAN = 0.9823
EPS_STD = 0.0090
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
STRIDE_HR = 72
STRIDE_LR = STRIDE_HR // 3

# --- Define overlap (padding) ---
PAD_HR = 8
PAD_LR = (PAD_HR + 2) // 3
PATCH_SIZE_LR = STRIDE_LR + 2 * PAD_LR
PATCH_SIZE_HR = STRIDE_HR + 2 * PAD_HR


# --- Helper Functions (No changes) ---
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

def save_as_geotiff(array, original_gdal_ds, output_path, crop_shape):
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
    out_band.SetNoDataValue(0)
    print("Calculating statistics for GeoTIFF...")
    out_band.ComputeStatistics(False) 
    out_band.FlushCache()
    out_ds = None
    print("GeoTIFF saved.")
# --- End of Helper Functions ---


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
    h_lr_orig, w_lr_orig = h // 3, w // 3
    
    h_crop, w_crop = h_lr_orig * 3, w_lr_orig * 3
    
    lst_30m_target = lst_30m_target_full[:h_crop, :w_crop] # This is our hr_target_np
    oli_30m = oli_30m[:, :h_crop, :w_crop]
    epsilon_30m = epsilon_30m[:h_crop, :w_crop]

    lst_100m_input = lst_30m_target.reshape(h_lr_orig, 3, w_lr_orig, 3).mean(axis=(1, 3))
    
    # --- 5. Normalization and Tensor Conversion ---
    print("Normalizing data and preparing tensors...")
    
    oli_30m_reflectance = oli_30m / 255.0
    
    lr_lst_norm = (lst_100m_input - LST_MEAN) / LST_STD
    oli_norm = (oli_30m_reflectance - oli_mean_bc) / oli_std_bc
    eps_norm = (epsilon_30m - EPS_MEAN) / EPS_STD
    
    guide_norm = np.concatenate([oli_norm, np.expand_dims(eps_norm, axis=0)], axis=0)
    
    lr_lst_tensor_unpadded = torch.from_numpy(lr_lst_norm).unsqueeze(0).unsqueeze(0)
    hr_guide_tensor_unpadded = torch.from_numpy(guide_norm).unsqueeze(0)
    hr_target_tensor = torch.from_numpy(lst_30m_target).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

    B, C_lr, H_lr, W_lr = lr_lst_tensor_unpadded.shape
    _, C_hr, H_hr, W_hr = hr_guide_tensor_unpadded.shape

    lr_lst_tensor = F.pad(lr_lst_tensor_unpadded, (PAD_LR, PAD_LR, PAD_LR, PAD_LR), mode='reflect').to(device, dtype=torch.float32)
    hr_guide_tensor = F.pad(hr_guide_tensor_unpadded, (PAD_HR, PAD_HR, PAD_HR, PAD_HR), mode='reflect').to(device, dtype=torch.float32)

    # --- 6. Run Tiled Inference (with Overlap) ---
    print("Running tiled model inference (with overlap)...")
    pred_hr_norm = torch.zeros(B, 1, H_hr, W_hr).to(device)

    for h_start_lr in tqdm(range(0, H_lr, STRIDE_LR), desc="Inference Tiles"):
        for w_start_lr in range(0, W_lr, STRIDE_LR):
            
            h_in_lr = h_start_lr
            w_in_lr = w_start_lr
            lr_patch = lr_lst_tensor[:, :, h_in_lr : h_in_lr + PATCH_SIZE_LR, 
                                          w_in_lr : w_in_lr + PATCH_SIZE_LR]
            
            h_in_hr = h_start_lr * 3
            w_in_hr = w_start_lr * 3
            hr_patch = hr_guide_tensor[:, :, h_in_hr : h_in_hr + PATCH_SIZE_HR, 
                                            w_in_hr : w_in_hr + PATCH_SIZE_HR]

            with torch.no_grad():
                pred_patch_norm_padded = model(lr_patch, hr_patch)
            
            pred_patch_center = pred_patch_norm_padded[:, :, PAD_HR : -PAD_HR, PAD_HR : -PAD_HR]
            
            h_out_hr = h_start_lr * 3
            w_out_hr = w_start_lr * 3

            h_end_hr = min(h_out_hr + STRIDE_HR, H_hr)
            w_end_hr = min(w_out_hr + STRIDE_HR, W_hr)
            
            pred_hr_norm[:, :, h_out_hr : h_end_hr, 
                                w_out_hr : w_end_hr] = pred_patch_center[:, :, :h_end_hr - h_out_hr, :w_end_hr - w_out_hr]

    print("Inference complete.")

    # --- 7. Post-processing (De-normalization) ---
    pred_hr_kelvin = (pred_hr_norm * LST_STD) + LST_MEAN
    
    pred_hr_np = pred_hr_kelvin.squeeze().cpu().numpy()
    hr_target_np = lst_30m_target # <-- This was the bug fix
    lr_input_np = (lr_lst_tensor_unpadded * LST_STD + LST_MEAN).squeeze().cpu().numpy()

    # --- 8. Calculate Metrics ---
    print("--- Quantitative Metrics ---")
    
    rmse = np.sqrt(np.mean((pred_hr_np - hr_target_np)**2))
    print(f"RMSE: {rmse:.4f} K")
    
    hr_target_norm_tensor = (hr_target_tensor - LST_MEAN) / LST_STD
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
    # --- 10. Save Comparison Plot (MODIFIED to 2x2) ---
    # --- 
    print(f"Saving comparison plot to {OUTPUT_PLOT_PATH}...")
    
    lr_input_upscaled = cv2.resize(
        lr_input_np, 
        (w_crop, h_crop), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create the 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 18), layout="constrained")
    fig.suptitle(f"Inference on Scene: {os.path.basename(os.path.dirname(SAMPLE_TIF_PATH))}", fontsize=20)
    
    # --- Temperature Plots ---
    valid_mask = hr_target_np > 150
    if valid_mask.any():
        vmin = np.percentile(hr_target_np[valid_mask], 5) 
        vmax = np.percentile(hr_target_np[valid_mask], 95)
    else:
        vmin, vmax = 273, 330 # Fallback
    
    # Plot 1: Input
    im_hot = axes[0, 0].imshow(lr_input_upscaled, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Input LST (100m, Nearest Neighbor)")
    
    # Plot 2: Ground Truth
    axes[0, 1].imshow(hr_target_np, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Ground Truth LST (30m)")
    
    # Plot 3: Predicted
    axes[1, 0].imshow(pred_hr_np, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"Predicted LST (30m) - RMSE: {rmse:.4f} K")
    
    # Add a colorbar for the temperature plots
    fig.colorbar(im_hot, ax=axes[0, :].ravel().tolist() + [axes[1,0]], shrink=0.8, label="Temperature (K)")
    
    # --- Plot 4: Error Map (Residuals) ---
    error_map = pred_hr_np - hr_target_np
    # Get a robust error range, ignoring 0K pixels
    if valid_mask.any():
        v_err = np.percentile(np.abs(error_map[valid_mask]), 98) # 98th percentile of error
    else:
        v_err = 5.0 # Fallback
        
    im_err = axes[1, 1].imshow(error_map, cmap='RdBu_r', vmin=-v_err, vmax=v_err)
    axes[1, 1].set_title(f"Error Map (Residuals)")
    fig.colorbar(im_err, ax=axes[1, 1], shrink=1.0, label="Error (K)")
        
    plt.savefig(OUTPUT_PLOT_PATH)
    print("Comparison plot saved.")

    # --- 
    # --- 11. Save Diagnostic Plots (NEW) ---
    # --- 
    print(f"Saving diagnostic plot to {OUTPUT_DIAG_PATH}...")
    fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), layout="constrained")
    fig_diag.suptitle(f"Diagnostic Plots for Scene: {os.path.basename(os.path.dirname(SAMPLE_TIF_PATH))}", fontsize=16)

    # --- Plot 1: Error Histogram ---
    errors_flat = error_map.flatten()
    target_flat = hr_target_np.flatten()
    valid_errors = errors_flat[target_flat > 150]
    
    mean_error = np.mean(valid_errors)
    std_error = np.std(valid_errors)
    
    ax1.hist(valid_errors, bins=100, range=(-5, 5))
    ax1.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean Error: {mean_error:.3f} K')
    ax1.set_title(f"Error Histogram (StdDev: {std_error:.3f} K)")
    ax1.set_xlabel("Error (K)")
    ax1.set_ylabel("Pixel Count")
    ax1.legend()

    # --- Plot 2: Predicted vs. Truth Scatter Plot ---
    pred_flat = pred_hr_np.flatten()
    valid_pred = pred_flat[target_flat > 150]
    valid_truth = target_flat[target_flat > 150]
    
    # Sample 50k points to avoid overplotting
    if len(valid_truth) > 50000:
        sample_indices = np.random.choice(len(valid_truth), 50000, replace=False)
        valid_truth = valid_truth[sample_indices]
        valid_pred = valid_pred[sample_indices]

    hb = ax2.hexbin(valid_truth, valid_pred, gridsize=100, cmap='inferno', mincnt=1)
    fig_diag.colorbar(hb, ax=ax2, label='Pixel Density')
    
    # Add 1:1 line
    lim_min = min(np.min(valid_truth), np.min(valid_pred))
    lim_max = max(np.max(valid_truth), np.max(valid_pred))
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label='1:1 Line')
    
    ax2.set_title("Predicted vs. Ground Truth")
    ax2.set_xlabel("Ground Truth (K)")
    ax2.set_ylabel("Predicted (K)")
    ax2.legend()
    ax2.set_aspect('equal', 'box')

    plt.savefig(OUTPUT_DIAG_PATH)
    print("Diagnostic plot saved.")
    print("\n--- Inference Complete ---")

if __name__ == "__main__":
    main()