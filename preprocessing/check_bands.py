# check_bands.py
from osgeo import gdal
import sys

# --- Tell GDAL to use exceptions ---
gdal.UseExceptions()

# --- Path to your TIF file ---
# Make sure this points to your file
TIF_PATH = "C:/Users/swast/OneDrive/Desktop/Case Study ML/data/0000000/all_bands.tif"

try:
    ds = gdal.Open(TIF_PATH)
except RuntimeError as e:
    print(f"Error opening file: {e}")
    sys.exit(1)

print(f"--- Inspecting File: {TIF_PATH} ---")
print(f"Total Bands: {ds.RasterCount}")
print("-" * 30)

for i in range(1, ds.RasterCount + 1):
    band = ds.GetRasterBand(i)
    desc = band.GetDescription()
    if not desc:
        desc = "(No description)"
    
    stats = band.GetStatistics(True, True)
    print(f"Band {i}: Min={stats[0]:.2f}, Max={stats[1]:.2f}, Mean={stats[2]:.2f} | Description: {desc}")

print("-" * 30)
print("Done.")

# Clean up
ds = None