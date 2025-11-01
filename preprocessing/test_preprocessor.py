import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import cv2  # <--- ADDED for ECC
from skimage.transform import resize
import warnings

# --- Constants (for Landsat 8 - adjust as needed) ---
# Metadata-derived values, placeholder_paths.
# You MUST get these from your image metadata (e.g., MTL.txt file)
# For TIRS Band 10
RADIANCE_MULT_B10 = 3.3420E-04
RADIANCE_ADD_B10 = 0.1
K1_CONSTANT_B10 = 774.8853
K2_CONSTANT_B10 = 1321.0789

# For OLI Bands 4 (Red) and 5 (NIR)
REFLECT_MULT = 2.0000E-05
REFLECT_ADD = -0.1

def get_band_data(gdal_dataset, band_index):
    """Helper to read a band as a NumPy array."""
    band = gdal_dataset.GetRasterBand(band_index)
    return band.ReadAsArray()

def dn_to_bt(tirs_dn, mult=RADIANCE_MULT_B10, add=RADIANCE_ADD_B10, k1=K1_CONSTANT_B10, k2=K2_CONSTANT_B10):
    """
    Converts TIRS Digital Number (DN) to Brightness Temperature (BT) in Kelvin.
    MODIFIED: Assumes 8-bit DN needs scaling to 16-bit range.
    """
    # --- NEW: Scale 8-bit DN to 16-bit DN ---
    # Test hypothesis: DN_16bit = DN_8bit * 256
    tirs_dn_16bit = tirs_dn.astype(np.float32) * 256.0

    # 1. DN to Radiance
    radiance = tirs_dn_16bit * mult + add
    # 2. Radiance to Brightness Temperature (K)
    # Suppress divide-by-zero warnings for pixels with 0 radiance
    with np.errstate(divide='ignore', invalid='ignore'):
        bt = k2 / np.log((k1 / radiance) + 1)
    bt[radiance <= 0] = 0  # Set invalid pixels to 0 K
    return bt, radiance

def calculate_ndvi(red_dn, nir_dn, mult=REFLECT_MULT, add=REFLECT_ADD):
    """
    Calculates NDVI from OLI DNs.
    MODIFIED: Assumes 8-bit DN (0-255) scales to 0-1 reflectance.
    """
    # --- NEW: Convert 8-bit DN to Reflectance (0-1) ---
    red_toa = red_dn.astype(np.float32) / 255.0
    nir_toa = nir_dn.astype(np.float32) / 255.0
    
    # Calculate NDVI
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_toa - red_toa) / (nir_toa + red_toa)
    
    # Handle NaNs (e.g., water) or invalid values
    ndvi[np.isnan(ndvi)] = -1  # Or another placeholder
    ndvi[np.isinf(ndvi)] = -1
    return ndvi

# --- NEW: ECC ALIGNMENT FUNCTION ---
def align_image(src_img, ref_img):
    """
    Aligns src_img (TIRS) to ref_img (OLI) using Enhanced Correlation
    Coefficient (ECC) from OpenCV.
    
    This finds an AFFINE transformation (translation, rotation, scale, shear)
    for sub-pixel accuracy.
    """
    # --- 1. Prepare images for ECC ---
    # ECC requires float32 and values > 0.
    # We normalize to [0, 1] for stability.
    ref_norm = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
    src_norm = (src_img - src_img.min()) / (src_img.max() - src_img.min())
    
    ref_norm = ref_norm.astype(np.float32)
    src_norm = src_norm.astype(np.float32)

    # --- 2. Define ECC parameters ---
    # We'll solve for an Affine transformation (cv2.MOTION_AFFINE)
    warp_mode = cv2.MOTION_AFFINE
    
    # Initialize the 2x3 warp matrix as an identity matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
        
    # Set termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        5000,  # max iterations
        1e-5   # epsilon (accuracy)
    )

    # --- 3. Run ECC Alignment ---
    try:
        (cc, warp_matrix) = cv2.findTransformECC(
            ref_norm,
            src_norm,
            warp_matrix,
            warp_mode,
            criteria,
            None, # No input mask
            5    # Gaussian blur sigma
        )
    except cv2.error as e:
        print(f"Warning: ECC alignment failed: {e}. Using identity matrix.")
        # Fallback to identity matrix (no transform)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # --- 4. Apply the warp to the ORIGINAL src_img ---
    rows, cols = ref_img.shape
    
    aligned_img = cv2.warpAffine(
        src_img,
        warp_matrix,
        (cols, rows), # dsize (width, height)
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return aligned_img, warp_matrix
# --- END OF NEW FUNCTION ---

def plot_exploration(bt_100m, radiance_100m, ndvi_30m, oli_30m, gdal_ds_100m, gdal_ds_30m):
    """
    Generates the three exploratory plots you specified.
    """
    print("Generating exploratory plots...")
    
    # --- 1. Radiometric Conversion Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Task 1.1: Radiometric Conversion Verification (TIRS Band 10)")
    
    # Filter out invalid/zero values for a cleaner plot
    valid_radiance = radiance_100m[radiance_100m > 0]
    valid_bt = bt_100m[bt_100m > 150] # Filter out physically impossible cold values

    ax1.hist(valid_radiance.flatten(), bins=100, color='orange')
    ax1.set_title("Histogram of TIRS Radiance")
    ax1.set_xlabel("Radiance ($W/m^2/sr/\mu m$)")
    ax1.set_ylabel("Pixel Count")
    
    ax2.hist(valid_bt.flatten(), bins=100, color='skyblue')
    ax2.set_title("Histogram of Brightness Temperature (BT)")
    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Pixel Count")
    
    plt.tight_layout()
    plt.savefig("plot_1_radiometric_check.png")
    print("Saved plot_1_radiometric_check.png")
    plt.close()

    # --- 2. Thermal-Spatial Correlation Plot ---
    # We must resize LST to match NDVI's shape for a 1:1 scatter plot
    bt_30m_resampled = resize(
        bt_100m,
        (ndvi_30m.shape[0], ndvi_30m.shape[1]),
        order=0, # Use Nearest Neighbor to avoid creating new values
        preserve_range=True,
        anti_aliasing=False
    )
    
    # Sample 10,000 pixels to avoid overplotting
    sample_indices = np.random.choice(bt_30m_resampled.size, 10000, replace=False)
    bt_sample = bt_30m_resampled.flat[sample_indices]
    ndvi_sample = ndvi_30m.flat[sample_indices]

    # Filter out invalid values 
    valid_mask = (bt_sample > 150) & (ndvi_sample > -1) # Check for BT > 150 K
    bt_sample = bt_sample[valid_mask]
    ndvi_sample = ndvi_sample[valid_mask]

    plt.figure(figsize=(10, 8))
    plt.hexbin(ndvi_sample, bt_sample, gridsize=100, cmap='inferno', mincnt=1)
    plt.title("Task 1.2: Thermal-Spatial Correlation (BT vs. NDVI)")
    plt.xlabel("NDVI (30m)")
    plt.ylabel("Brightness Temperature (K)")
    cb = plt.colorbar(label='Pixel Density')
    plt.savefig("plot_2_correlation_check.png")
    print("Saved plot_2_correlation_check.png")
    plt.close()

    # --- 3. Alignment Check Plot ---
    # Get geotransform (pixel size and origin)
    # Assumes gdal_ds_30m (OLI) and gdal_ds_100m (TIRS) are available
    gt_30m = gdal_ds_30m.GetGeoTransform()
    gt_100m = gdal_ds_100m.GetGeoTransform()

    # Calculate extents
    ext_30m = [
        gt_30m[0], gt_30m[0] + gdal_ds_30m.RasterXSize * gt_30m[1],
        gt_30m[3] + gdal_ds_30m.RasterYSize * gt_30m[5], gt_30m[3]
    ]
    ext_100m = [
        gt_100m[0], gt_100m[0] + gdal_ds_100m.RasterXSize * gt_100m[1],
        gt_100m[3] + gdal_ds_100m.RasterYSize * gt_100m[5], gt_100m[3]
    ]

    fig, ax = plt.subplots(figsize=(12, 12))
    # Plot OLI band in grayscale
    ax.imshow(oli_30m, cmap='gray', extent=ext_30m)
    
    # Plot TIRS band with high transparency, using its own extent
    ax.imshow(
        bt_100m,
        cmap='hot',
        alpha=0.4,
        extent=ext_100m,
        interpolation='nearest' # Critical to show raw pixels
    )
    
    # Add grid lines for TIRS pixels
    ax.set_xticks(np.arange(ext_100m[0], ext_100m[1], gt_100m[1]), minor=False)
    ax.set_yticks(np.arange(ext_100m[3], ext_100m[2], gt_100m[5]), minor=False)
    ax.grid(which='both', color='cyan', linestyle='-', linewidth=0.5)

    ax.set_title("Task 1.3: Alignment Check (TIRS 100m grid over OLI 30m)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    # Zoom into a small area to see detail
    ax.set_xlim(gt_30m[0], gt_30m[0] + 200 * gt_30m[1]) # Show first 200 OLI pixels
    ax.set_ylim(gt_30m[3] + 200 * gt_30m[5], gt_30m[3])
    
    plt.savefig("plot_3_alignment_check.png")
    print("Saved plot_3_alignment_check.png")
    plt.close()


def main(tif_path):
    """
    Main EDA function.
    
    ASSUMPTIONS:
    - Band 4: OLI Red
    - Band 5: OLI NIR
    - Band 10: TIRS 1
    
    Please adjust these band indices based on your 'all_bands.tif' structure.
    """
    
    # --- FIX: Explicitly tell GDAL to use exceptions ---
    gdal.UseExceptions()
    # ----------------------------------------------------

    print(f"Opening {tif_path}")
    warnings.simplefilter('ignore', gdal.CE_Warning) # Changed from CE_Warning
    
    try:
        gdal_ds = gdal.Open(tif_path)
    except RuntimeError as e:
        # If gdal.Open fails, it will now raise an exception
        print(f"Error: Could not open file '{tif_path}'. GDAL error: {e}")
        return
        
    # --- Define Band Indices (CRITICAL: Adjust this) ---
    OLI_RED_BAND_IDX = 4
    OLI_NIR_BAND_IDX = 5
    TIRS_B10_BAND_IDX = 10
    # ----------------------------------------------------

    # --- Extract Bands ---
    oli_red_30m = get_band_data(gdal_ds, OLI_RED_BAND_IDX)
    oli_nir_30m = get_band_data(gdal_ds, OLI_NIR_BAND_IDX)
    tirs_b10_30m = get_band_data(gdal_ds, TIRS_B10_BAND_IDX)
    
    if oli_red_30m is None or tirs_b10_30m is None:
        print(f"Error: Could not read bands. Check indices. Max bands: {gdal_ds.RasterCount}")
        return

    # --- NEW: Run ECC Alignment Test ---
    # We align the TIRS 30m band (source) to the OLI NIR 30m band (reference)
    # Since they are from the same file, we expect the warp_matrix
    # to be very close to an identity matrix.
    print("Running ECC alignment test...")
    tirs_b10_30m_aligned, warp_matrix = align_image(
        tirs_b10_30m.astype(np.float32), 
        oli_nir_30m.astype(np.float32)
    )
    print("ECC Alignment Complete. Warp Matrix:")
    print(warp_matrix)
    # --- END OF ECC TEST ---


    # --- Simulate 100m TIRS ---
    # We use the *aligned* TIRS band to simulate the 100m input
    downscale_factor = 3
    shape_100m = (
        oli_red_30m.shape[0] // downscale_factor,
        oli_red_30m.shape[1] // downscale_factor
    )
    
    # Use 'resize' with anti-aliasing to simulate the sensor's MTF
    tirs_b10_100m_dn = resize(
        tirs_b10_30m_aligned,  # <--- USE THE ALIGNED VERSION
        shape_100m,
        order=1, # Bilinear
        preserve_range=True,
        anti_aliasing=True
    ).astype(tirs_b10_30m.dtype)

    # --- Perform Physical Conversions ---
    
    # 1. TIRS (Simulated 100m) to BT and Radiance
    bt_100m, radiance_100m = dn_to_bt(tirs_b10_100m_dn)
    
    # 2. OLI (30m) to NDVI
    ndvi_30m = calculate_ndvi(oli_red_30m, oli_nir_30m)

    # --- Generate Plots ---
    # We'll create a dummy 100m geotransform
    gt_30m = gdal_ds.GetGeoTransform()
    gt_100m_sim = list(gt_30m)
    gt_100m_sim[1] = gt_30m[1] * downscale_factor # 30m * 3 = 90m (close enough)
    gt_100m_sim[5] = gt_30m[5] * downscale_factor
    
    # Create a dummy in-memory 100m dataset
    driver_mem = gdal.GetDriverByName('MEM')
    gdal_ds_100m = driver_mem.Create(
        '',
        shape_100m[1],
        shape_100m[0],
        1,
        gdal.GDT_Float32
    )
    gdal_ds_100m.SetGeoTransform(tuple(gt_100m_sim))
    gdal_ds_100m.SetProjection(gdal_ds.GetProjection())
    
    plot_exploration(bt_100m, radiance_100m, ndvi_30m, oli_red_30m, gdal_ds_100m, gdal_ds)
    
    print("EDA complete.")
    gdal_ds = None # Close file
    gdal_ds_100m = None


if __name__ == "__main__":
    # Path to one of your sample files
    SAMPLE_TIF_PATH = "C:/Users/swast/OneDrive/Desktop/Case Study ML/data/0000000/all_bands.tif"
    main(SAMPLE_TIF_PATH)