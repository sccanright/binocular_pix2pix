# =========================================================================
# Analysis Script for Image Quality Metrics (varifocal_pix2pix)
#
# Computes average image similarity metrics (MAE, MSE, PSNR, SSIM, BCE)
# between real and generated images within a given results folder.
#
# Usage:
#     python analyze.py results/your_data_folder
#     → Automatically looks in: your_data_folder/test_latest/images
#     → Handles both sliced and unsliced data structures
#     → Aggregates metrics across all subfolders (e.g., *_sliceYY or YYYY-MM-DD)
#     → Updates a summary log (analysis_log.txt), replacing old results if rerun
#
#  !! Requires folder structure: results/FOLDER/test_latest/images !!
#  !! Image names must follow format: real_B_X.png and fake_B_X.png (X = 1 to 10) !!
#
# Author: Slater Canright | Date: 2025-04-25
# =========================================================================

import torch
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image
import pytorch_msssim
import math
import sys
import time

def load_image(img_path):
    img = Image.open(img_path).convert('L')
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)

def calculate_psnr(mse):
    return 20 * math.log10(1.0 / math.sqrt(mse))

# --- SETUP ---
script_dir = Path(__file__).resolve().parent

if len(sys.argv) < 2:
    print("Usage: python analyze.py results/TEST")
    sys.exit(1)

# Add fixed subpath after the user input
relative_base = Path(sys.argv[1])
root_dir = script_dir / relative_base / "test_latest" / "images"

# --- Initialize lists to accumulate values across all folders ---
mae_losses_all_folders = []
mse_losses_all_folders = []
ssim_values_all_folders = []
psnr_values_all_folders = []
cross_entropy_values_all_folders = []

# Record start time
start_time = time.time()

# --- TRAVERSE BASE FOLDERS (YYYY-MM-DD) ---
for base_folder in root_dir.glob("*/"):  # Look for YYYY-MM-DD style folders
    if base_folder.is_dir():
        for i in range(1, 11):  # Assuming 1 to 10
            real_img_path = base_folder / f"real_B_{i}.png"
            fake_img_path = base_folder / f"fake_B_{i}.png"

            if not real_img_path.exists() or not fake_img_path.exists():
                print(f"Missing image pair in {base_folder.name}, index {i}")
                continue

            real_img = load_image(real_img_path)
            fake_img = load_image(fake_img_path)

            mae_loss = torch.nn.functional.l1_loss(real_img, fake_img)
            mae_losses_all_folders.append(mae_loss.item())
            
            mse_loss = torch.nn.functional.mse_loss(real_img, fake_img)
            mse_losses_all_folders.append(mse_loss.item())

            psnr = calculate_psnr(mse_loss.item())
            psnr_values_all_folders.append(psnr)

            ssim_value_torch = pytorch_msssim.ssim(real_img, fake_img, data_range=1)
            ssim_values_all_folders.append(ssim_value_torch.item())

            cross_entropy = torch.nn.functional.binary_cross_entropy(real_img, fake_img)
            cross_entropy_values_all_folders.append(cross_entropy.item())

# --- AVERAGE METRICS ACROSS ALL FOLDERS ---
if mae_losses_all_folders:
    print(f"\nAnalyzed {len(mae_losses_all_folders)} image pairs across all folders.")
    print(f"Average MAE: {np.mean(mae_losses_all_folders):.6f}")
    print(f"Average MSE: {np.mean(mse_losses_all_folders):.6f}")
    print(f"Average PSNR: {np.mean(psnr_values_all_folders):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_values_all_folders):.6f}")
    print(f"Average Cross Entropy: {np.mean(cross_entropy_values_all_folders):.6f}")
else:
    print("No valid image pairs found.")

# Record end time and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Analysis completed in {elapsed_time:.2f} seconds.")

log_file = script_dir / "results" / "analysis_log.txt"
folder_name = relative_base.name  # "512slicedata01", "1024slicedata01", etc.

entry_header = f"{folder_name}:\n"
entry_stats = (
    f"{entry_header}"
    f"Average MAE: {np.mean(mae_losses_all_folders):.6f}\n"
    f"Average MSE: {np.mean(mse_losses_all_folders):.6f}\n"
    f"Average PSNR: {np.mean(psnr_values_all_folders):.2f} dB\n"
    f"Average SSIM: {np.mean(ssim_values_all_folders):.6f}\n"
    f"Average Cross Entropy: {np.mean(cross_entropy_values_all_folders):.6f}\n\n"
)

# Load existing file if it exists
if log_file.exists():
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    # Remove old block for this folder if it exists
    new_lines = []
    skip = False
    for line in lines:
        if line.strip() == f"{folder_name}:":
            skip = True
            continue
        if skip and line.strip() == "":
            skip = False
            continue
        if not skip:
            new_lines.append(line)

    new_lines.append(entry_stats)

    with open(log_file, "w") as f:
        f.writelines(new_lines)
else:
    # Just write the entry
    with open(log_file, "w") as f:
        f.write(entry_stats)