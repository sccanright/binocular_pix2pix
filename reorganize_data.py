# =========================================================================================
# Script to process and reorganize data from the DATA folder 
# within the FIELDSET directory of datasets in varifocal_pix2pix.
#
# The script renames, processes, and splits the data into "train" 
# and "test" folders based on the specified percentage.
#
# Default usage: 
#   python reorganize_data.py
#
# To customize paths and split percentage, use:
#   python reorganize_data.py --root_folder /path/to/data --split_percentage ## --destination_folder /path/to/dest
#
# !! Ensure the directory structure follows the expected format for smooth execution !!
#
# Author: Slater Canright, Date: 1/31/2025
# =========================================================================================

import os
import shutil
import random
import argparse

def reorganize_data(root_folder, split_percentage, destination_folder=None):
    if not os.path.isdir(root_folder):
        Warning(f"{root_folder} is not a valid directory.")
        return

    # Determine FIELDSET folder and DATA folder
    data_folder = os.path.join(root_folder, "DATA")
    
    # Check if the DATA folder exists
    if not os.path.isdir(data_folder):
        print("No new data in need of sorting...")
        return  # Exit if no DATA folder is found

    # If destination folder is provided, use it; otherwise, use root folder's FIELDSET
    if destination_folder:
        train_folder = os.path.join(destination_folder, "train")
        test_folder = os.path.join(destination_folder, "test")
        repeats_folder = os.path.join(destination_folder, "repeats")
        incomplete_folder = os.path.join(destination_folder, "incomplete")
    else:
        train_folder = os.path.join(root_folder, "train")
        test_folder = os.path.join(root_folder, "test")
        repeats_folder = os.path.join(root_folder, "repeats")
        incomplete_folder = os.path.join(root_folder, "incomplete")

    # Create train, test, repeats, and incomplete folders
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(repeats_folder, exist_ok=True)
    os.makedirs(incomplete_folder, exist_ok=True)

    all_runs = []  # List to store all renamed run folders

    # Step 1: Reorganize the run folders
    print("Reorganizing and renaming all run folders...")
    no_file = False
    unwanted_file = False
    for date_folder in os.listdir(data_folder):
        date_path = os.path.join(data_folder, date_folder)

        if os.path.isdir(date_path) and len(date_folder) == 10 and date_folder[4] == '-' and date_folder[7] == '-':
            # Remove .ini files from date folder
            for file in os.listdir(date_path):
                if file.endswith(".ini"):
                    os.remove(os.path.join(date_path, file))
            
            for run_folder in os.listdir(date_path):
                run_path = os.path.join(date_path, run_folder)
                if os.path.isdir(run_path) and run_folder.startswith("run_"):
                    new_name = f"{date_folder}_{run_folder}"
                    new_path = os.path.join(root_folder, new_name)

                    # Check for .jpg and .png files in the run folder
                    jpg_files = [f for f in os.listdir(run_path) if f.endswith(".jpg")]
                    png_files = [f for f in os.listdir(run_path) if f.endswith(".png")]

                    # If the run folder doesn't have exactly 10 .jpg files and 1 .png file
                    if len(jpg_files) != 10 or len(png_files) != 1:
                        if not no_file:
                            print("Missing or extra files found within run folder[s]. Folder[s] moved to 'incomplete'...")
                            no_file = True  # Only print once
                        shutil.move(run_path, os.path.join(incomplete_folder, new_name))
                        
                    else:
                        # Move the folder to FIELDSET and rename it
                        shutil.move(run_path, new_path)
                        all_runs.append(new_path)

            # Remove the date folder if it contains no more run_ folders
            run_folders_remaining = [
                f for f in os.listdir(date_path)
                if os.path.isdir(os.path.join(date_path, f)) and f.startswith("run_")
            ]

            if not run_folders_remaining:
                print(f"Removing {date_path} (no run_ folders remain)...")
                shutil.rmtree(date_path)  # Forcefully remove folder and all contents

    # Remove .ini files from the DATA folder
    for file in os.listdir(data_folder):
        if file.endswith(".ini"):
            os.remove(os.path.join(data_folder, file))

   # Check if there are no YYYY-MM-DD folders left in the DATA folder
    date_folders_remaining = [
        f for f in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, f)) and len(f) == 10 and f[4] == '-'
    ]

    if not date_folders_remaining:
    print(f"DATA folder is now empty. Preserving folder structure...")
    # Remove any leftover files in DATA (e.g., .ini files)
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Step 2: Split the data into train and test sets
    print("Reorganizing and splitting data into train and test sets...")
    test_count = min(100, max(1, round(len(all_runs) * (split_percentage / 100)))) if all_runs else 0
    test_runs = random.sample(all_runs, test_count) if test_count > 0 else []

    # Check for existing runs in train/test folders and move duplicates to 'repeats'
    duplicates_found = False
    for run in all_runs:
        run_name = os.path.basename(run)
        if os.path.exists(os.path.join(train_folder, run_name)) or os.path.exists(os.path.join(test_folder, run_name)):
            if not duplicates_found:
                print("Duplicates found, moving to 'repeats' folder...")
                duplicates_found = True  # Only print once
            shutil.move(run, os.path.join(repeats_folder, run_name))
        else:
            if run in test_runs:
                shutil.move(run, os.path.join(test_folder, run_name))
            else:
                shutil.move(run, os.path.join(train_folder, run_name))

    print("Data has been processed, reorganized, and properly culled if needed!!")

if __name__ == "__main__":
    root_folder_default = os.path.expanduser('~/varifocal_pix2pix/datasets/FIELDSET')

    parser = argparse.ArgumentParser(description="Extract, rename, and split run folders into train, test, and repeats sets.")
    parser.add_argument("--root_folder", help="Path to the FIELDSET folder containing the DATA folder.", default=root_folder_default)
    parser.add_argument("--split_percentage", type=float, help="Percentage of runs to put in test set (default: 5%)", default=5)
    parser.add_argument("--destination_folder", help="Path to save train, test, and repeats folders (optional).", default=None)

    args = parser.parse_args()

    reorganize_data(args.root_folder, args.split_percentage, args.destination_folder)

