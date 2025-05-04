# =========================================================================================
# Script to slice and reorganize image data from the DATA folder 
# within the FIELDSET directory of datasets in varifocal_pix2pix.
#
# The script slices images into smaller tiles starting from the vertical centerline, 
# saves them into separate folders based on slice indices, and moves unsliced images 
# to the UN_SLICED folder.
#
# Default usage:
#   python slice.py
#
# Change slice size:
#   python slice.py --size
#
# !! Ensure the directory structure follows the expected format for smooth execution !!
#
# Author: Slater Canright, Date: 4/12/2025
# =========================================================================================

import os
from PIL import Image
import numpy as np
import time
import shutil
import argparse
import re

# 🌱 Root dataset path (constant)
DATA_ROOT = os.path.expanduser('~/varifocal_pix2pix/datasets/FIELDSET/DATA')

# 🧠 Will store your final selected crop values after 'q'
FINAL_CROP = None

def find_all_run_folders():
    """Returns a list of all run_xx folders under properly named YYYY-MM-DD date folders."""
    run_folders = []
    date_folder_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')  # Only allow strict YYYY-MM-DD format

    for date_folder in os.listdir(DATA_ROOT):
        if not date_folder_pattern.match(date_folder):
            continue  # skip anything not exactly YYYY-MM-DD

        date_path = os.path.join(DATA_ROOT, date_folder)
        if not os.path.isdir(date_path):
            continue

        for run_folder in os.listdir(date_path):
            run_path = os.path.join(date_path, run_folder)
            if os.path.isdir(run_path) and run_folder.startswith("run"):
                run_folders.append(run_path)

    return run_folders

def get_png_and_jpgs(run_folder):
    """Returns the PNG and all JPG files in a given run folder."""
    png = next((f for f in os.listdir(run_folder) if f.lower().endswith('.png')), None)
    jpgs = [f for f in sorted(os.listdir(run_folder)) if f.lower().endswith('.jpg')]

    if png and jpgs:
        return os.path.join(run_folder, png), [os.path.join(run_folder, jpg) for jpg in jpgs]
    return None, None

def up_interpolate_jpg(jpg_path, png_path):
    """Upsample JPG to match PNG size."""
    jpg_img = Image.open(jpg_path)
    png_img = Image.open(png_path)
    upsampled_jpg = jpg_img.resize(png_img.size, Image.LANCZOS)
    return upsampled_jpg

def slice_image(image, slice_size):
    """Slices an image symmetrically from the vertical centerline without exceeding bounds."""
    img_width, img_height = image.size
    tile_width, tile_height = slice_size

    slices = []

    # Y-direction slicing (top to bottom, regular loop)
    for y in range(0, img_height - tile_height + 1, tile_height):
        # X-direction slicing (from centerline outward)
        center_x = img_width // 2

        # Start with the center slice
        left = center_x - tile_width // 2
        if left >= 0 and (left + tile_width) <= img_width:
            slices.append(image.crop((left, y, left + tile_width, y + tile_height)))

        # Move outward left and right from center
        offset = 1
        while True:
            placed = False

            # Slice to the right
            right_x = center_x + offset * tile_width - tile_width // 2
            if right_x + tile_width <= img_width:
                slices.append(image.crop((right_x, y, right_x + tile_width, y + tile_height)))
                placed = True

            # Slice to the left
            left_x = center_x - offset * tile_width - tile_width // 2
            if left_x >= 0:
                slices.append(image.crop((left_x, y, left_x + tile_width, y + tile_height)))
                placed = True

            if not placed:
                break  # Stop when neither direction can place a full tile

            offset += 1

    return slices

def move_unsliced_folder(run_folder, un_sliced_dir):
    """Move the entire run folder to the UN_SLICED directory."""
    date_folder = os.path.basename(os.path.dirname(run_folder))
    run_folder_name = os.path.basename(run_folder)
    target_folder = os.path.join(un_sliced_dir, date_folder)
    os.makedirs(target_folder, exist_ok=True)
    shutil.move(run_folder, target_folder)

def save_grouped_slices(slice_dict, date_folder, run_folder_name):
    """Saves grouped slices where each folder contains the same slice index from all images."""
    for idx, slice_group in slice_dict.items():
        slice_folder = os.path.join(date_folder, f"{run_folder_name}_slice{idx + 1}")
        os.makedirs(slice_folder, exist_ok=True)
        for i, (slice_img, original_name) in enumerate(slice_group):
            ext = os.path.splitext(original_name)[1]
            slice_filename = f"{os.path.splitext(original_name)[0]}{ext}"
            slice_img.save(os.path.join(slice_folder, slice_filename))

def slice_images_in_run(run_folder, png_path, jpg_paths, un_sliced_dir, slice_size):
    """Slices all images, saving grouped slices and moving the unsliced folder."""
    date_folder = os.path.dirname(run_folder)
    run_folder_name = os.path.basename(run_folder)
    slice_dict = {}

    # Process JPGs
    for jpg_path in jpg_paths:
        upsampled_jpg = up_interpolate_jpg(jpg_path, png_path)
        slices = slice_image(upsampled_jpg, slice_size)
        for idx, slice_img in enumerate(slices):
            slice_dict.setdefault(idx, []).append((slice_img, os.path.basename(jpg_path)))

    # Process PNG
    png_img = Image.open(png_path)
    png_slices = slice_image(png_img, slice_size)
    for idx, slice_img in enumerate(png_slices):
        slice_dict.setdefault(idx, []).append((slice_img, os.path.basename(png_path)))

    # Save grouped slices
    save_grouped_slices(slice_dict, date_folder, run_folder_name)

    # Move unsliced folder
    move_unsliced_folder(run_folder, un_sliced_dir)

def main():
    parser = argparse.ArgumentParser(description="Slice and organize varifocal image data.")
    parser.add_argument("--size", type=int, default=512, help="Size of square slices (default: 512)")
    args = parser.parse_args()
    slice_size = (args.size, args.size)

    start_time = time.time()
    print(f"📤 Starting up-interpolating and slicing processes with slice size: {slice_size[0]}x{slice_size[1]}")

    run_folders = find_all_run_folders()
    if not run_folders:
        print("⚠️ No run folders found.")
        return

    un_sliced_dir = os.path.expanduser('~/varifocal_pix2pix/datasets/FIELDSET/UN_SLICED')
    os.makedirs(un_sliced_dir, exist_ok=True)

    for run_folder in run_folders:
        png_path, jpg_paths = get_png_and_jpgs(run_folder)
        if not png_path or not jpg_paths:
            print(f"⚠️ Missing PNG or JPG files in {run_folder}")
            continue

        slice_images_in_run(run_folder, png_path, jpg_paths, un_sliced_dir, slice_size)

    elapsed = time.time() - start_time
    print(f"✅ Completed slicing in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
