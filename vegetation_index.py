"""
vegetation_index.py
Generates NDVI and GNDVI images from RGB + NIR data
Saves processed, resized, color-mapped images for training and Streamlit use
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Configuration
RAW_ROOT = Path("C:/Users/Jabili N/Music/AgriHackathon_NutrientDeficiency/dataset/raw")
PROC_ROOT = Path("C:/Users/Jabili N/Music/AgriHackathon_NutrientDeficiency/dataset/processed")
FOLDERS = ["wheat_13082019", "wheat_27072019", "wheat_30082019"]
TARGET_H, TARGET_W = 798, 1098
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

def compute_index(nir, band):
    bottom = (nir.astype(float) + band.astype(float)) + 1e-5
    index = (nir.astype(float) - band.astype(float)) / bottom
    index = np.clip(index, -1, 1)
    index = ((index + 1) / 2 * 255).astype(np.uint8)
    return cv2.applyColorMap(index, cv2.COLORMAP_JET)

def generate_indices():
    for folder in FOLDERS:
        rgb_dir = RAW_ROOT / folder / "RGB"
        nir_dir = RAW_ROOT / folder / "NIR"

        ndvi_dir = PROC_ROOT / "NDVI" / folder
        gndvi_dir = PROC_ROOT / "GNDVI" / folder
        ndvi_dir.mkdir(parents=True, exist_ok=True)
        gndvi_dir.mkdir(parents=True, exist_ok=True)

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(IMG_EXTS)])
        nir_files = sorted([f for f in os.listdir(nir_dir) if f.lower().endswith(IMG_EXTS)])

        for rgb_file, nir_file in zip(rgb_files, nir_files):
            rgb = cv2.imread(str(rgb_dir / rgb_file), cv2.IMREAD_COLOR)
            nir = cv2.imread(str(nir_dir / nir_file), cv2.IMREAD_COLOR)

            if rgb is None or nir is None:
                print(f"⚠️ Skipping unreadable: {rgb_file} or {nir_file}")
                continue

            # Resize both
            rgb = cv2.resize(rgb, (TARGET_W, TARGET_H))
            nir = cv2.resize(nir, (TARGET_W, TARGET_H))

            red = rgb[:, :, 2]
            green = rgb[:, :, 1]
            nir_band = nir[:, :, 0]

            ndvi = compute_index(nir_band, red)
            gndvi = compute_index(nir_band, green)

            cv2.imwrite(str(ndvi_dir / rgb_file), ndvi)
            cv2.imwrite(str(gndvi_dir / rgb_file), gndvi)
            print(f"✅ {folder}: Saved NDVI & GNDVI for {rgb_file}")

if __name__ == "__main__":
    generate_indices()
    print("\n🎯 Done! All NDVI and GNDVI images saved in 'processed' folder.")