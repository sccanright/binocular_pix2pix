# =========================================================================
# Crop Script for Focus Stack Alignment (hyperspectral_pix2pix)
#
# Crops JPG focus stack images in the DATA folder to match the
# size of corresponding PNG EDoF images (FIELDSET directory).
#
# Interactive mode (to find crop values):
#     python crop.py --iterative
#     → Enter crop values: left top right bottom
#     → Repeat until satisfied, then press 'q' to quit and save values 
#     → press y/n to run batch cropping
#
# Batch mode (after setting crop values):
#     python crop.py
#
#  !! Requires proper folder structure: DATA/YYYY-MM-DD/run_xx !!
#
# Author: Slater Canright | Date: 2025-04-12
# =========================================================================


import matplotlib
matplotlib.use('TkAgg')  # Or 'Agg' if you're in a headless environment

import os
import random
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import time

# 🌱 Root dataset path (constant)
DATA_ROOT = os.path.expanduser('~/hyperspectral_pix2pix/datasets/FIELDSET/DATA')
UN_CROPPED_DIR = os.path.join(os.path.dirname(DATA_ROOT), 'UN_CROPPED')

# 🧠 Will store your final selected crop values after 'q'
FINAL_CROP = None

def find_all_run_folders():
    """Returns a list of all valid run folders under the DATA folder."""
    run_folders = []
    for date_folder in sorted(os.listdir(DATA_ROOT)):
        if not date_folder[:4].isdigit():
            continue
        date_path = os.path.join(DATA_ROOT, date_folder)
        if os.path.isdir(date_path):
            for run_folder in os.listdir(date_path):
                if 'run' not in run_folder:
                    continue
                run_path = os.path.join(date_path, run_folder)
                if os.path.isdir(run_path):
                    run_folders.append(run_path)
    return run_folders

def get_png_and_jpgs(run_folder):
    png = next((f for f in os.listdir(run_folder) if f.lower().endswith('.png')), None)
    jpgs = sorted([f for f in os.listdir(run_folder) if f.lower().endswith('.jpg')])
    if png and jpgs:
        return os.path.join(run_folder, png), [os.path.join(run_folder, j) for j in jpgs]
    return None, []

def save_crop_values(crop_vals, filename='crop_values.txt'):
    with open(filename, 'w') as f:
        f.write(' '.join(map(str, crop_vals)))
    print(f"💾 Saved crop values to {filename}")

def load_crop_values(filename='crop_values.txt'):
    """Loads crop values from a text file."""
    try:
        with open(filename, 'r') as f:
            crop_values = f.read().strip().split()
            if len(crop_values) == 4:
                return tuple(map(int, crop_values))
            else:
                print("⚠️ Invalid crop values in file. Please set crop values manually.")
                return None
    except FileNotFoundError:
        print(f"⚠️ {filename} not found. Please run iterative mode first.")
        return None

def overlay_cropped_jpg(png_path, jpg_path, crop_box):
    left, top, right, bottom = crop_box
    png_img = Image.open(png_path).convert("RGBA")
    jpg_img = Image.open(jpg_path).convert("RGBA")

    w, h = jpg_img.size
    cropped_jpg = jpg_img.crop((left, top, w - right, h - bottom))
    png_resized = png_img.resize(cropped_jpg.size)

    initial_alpha = 100

    def update(val):
        alpha = slider.val
        jpg_overlay = cropped_jpg.copy()
        jpg_overlay.putalpha(int(alpha))
        composite = Image.alpha_composite(png_resized, jpg_overlay)
        ax.imshow(composite)
        ax.set_title(f"Overlay (L:{left}, T:{top}, R:{right}, B:{bottom}) | Alpha: {int(alpha)}")
        plt.draw()

    fig, ax = plt.subplots(figsize=(10, 10))
    jpg_overlay = cropped_jpg.copy()
    jpg_overlay.putalpha(initial_alpha)
    composite = Image.alpha_composite(png_resized, jpg_overlay)
    ax.imshow(composite)
    ax.set_title(f"Overlay (L:{left}, T:{top}, R:{right}, B:{bottom}) | Alpha: {initial_alpha}")
    ax.axis('off')
    ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
    slider = Slider(ax_slider, 'Transparency', 0, 255, valinit=initial_alpha, valstep=1)
    slider.on_changed(update)
    plt.show(block=True)

def run_iterative_cropper():
    global FINAL_CROP
    run_folders = find_all_run_folders()
    if not run_folders:
        print("No run folders found.")
        return

    run_folder = random.choice(run_folders)
    print("📂 Selected folder:", run_folder)

    png_path, jpg_paths = get_png_and_jpgs(run_folder)
    if not png_path or not jpg_paths:
        print("⚠️ Missing PNG or JPGs in:", run_folder)
        return

    last_crop_vals = None

    while True:
        try:
            crop_input = input("\n✂️  Enter crop values (left top right bottom) or 'q' to quit: ")
            if crop_input.lower() == 'q':
                if last_crop_vals:
                    FINAL_CROP = last_crop_vals
                    print(f"\n✅ Final crop values: {FINAL_CROP}")
                    save_crop_values(FINAL_CROP)

                    do_batch = input("🚀 Do you want to start batch cropping now? (y/n): ").strip().lower()
                    if do_batch == 'y':
                        run_main_cropper()
                    else:
                        print("🛑 Skipping batch cropping. You can run it later with `python crop.py`.")
                else:
                    print("\nNo crop values entered.")
                break

            crop_vals = tuple(map(int, crop_input.strip().split()))
            if len(crop_vals) != 4:
                print("⚠️  Please enter exactly 4 integers.")
                continue

            last_crop_vals = crop_vals
            overlay_cropped_jpg(png_path, jpg_paths[0], crop_vals)

        except Exception as e:
            print("❌ Error:", e)

def run_main_cropper():
    global FINAL_CROP

    # Try to load crop values from file if they are not set yet
    if FINAL_CROP is None:
        FINAL_CROP = load_crop_values()

    if FINAL_CROP is None:
        print("❌ No valid crop values have been set. Please run iterative mode first.")
        return

    left, top, right, bottom = FINAL_CROP
    print(f"📤 Cropping all JPGs with crop values: Left={left}, Top={top}, Right={right}, Bottom={bottom}")

    start_time = time.time()
    total_images = 0

    run_folders = find_all_run_folders()
    if not run_folders:
        print("⚠️ No run folders found.")
        return

    un_cropped_dir = os.path.join(os.path.dirname(DATA_ROOT), 'UN_CROPPED')
    os.makedirs(un_cropped_dir, exist_ok=True)

    for run_folder in run_folders:
        png_path, jpg_paths = get_png_and_jpgs(run_folder)
        if not png_path or not jpg_paths:
            print(f"⚠️ Missing PNG or JPG files in {run_folder}")
            continue

        for jpg_path in jpg_paths:
            try:
                original_jpg = Image.open(jpg_path)
                cropped_jpg = original_jpg.crop((left, top, original_jpg.width - right, original_jpg.height - bottom))

                # Backup original JPG to UN_CROPPED folder with subfolders matching original
                relative_path = os.path.relpath(jpg_path, DATA_ROOT)
                backup_path = os.path.join(un_cropped_dir, os.path.dirname(relative_path))
                os.makedirs(backup_path, exist_ok=True)

                uncropped_filename = os.path.join(backup_path, os.path.basename(jpg_path))
                original_jpg.save(uncropped_filename)

                # Save cropped image (overwrite original)
                cropped_filename = os.path.join(run_folder, os.path.basename(jpg_path))
                cropped_jpg.save(cropped_filename)

                total_images += 1

            except Exception as e:
                print(f"❌ Failed to crop {jpg_path}: {e}")

    elapsed = time.time() - start_time
    print(f"✅ Cropped {total_images} images in {elapsed:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="Crop tool for hyperspectral_pix2pix.")
    parser.add_argument('--iterative', action='store_true', help='Run the interactive crop tuning mode')
    args = parser.parse_args()

    if args.iterative:
        run_iterative_cropper()
    else:
        run_main_cropper()

if __name__ == "__main__":
    main()
