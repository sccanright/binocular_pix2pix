import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
import re

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
##############################################################################################################
# Started Changing 
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        
        # Get the paths for each day folder
        self.date_run_folders = sorted([os.path.join(self.dir_AB, d) for d in os.listdir(self.dir_AB) if os.path.isdir(os.path.join(self.dir_AB, d))])  # get date_run folder paths
        assert self.opt.load_size >= self.opt.crop_size   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
        index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
        A (tensor)       -- an image in the input domain
        B (tensor)       -- its corresponding image in the target domain
        A_paths (str)    -- image paths
        B_paths (str)    -- image paths
        """
        # Get the path for the corresponding day folder
        date_run_folder = self.date_run_folders[index]
        
        # Find the A image (assuming it is a .png file in the date_run folder)
        A_path = [os.path.join(date_run_folder, f) for f in os.listdir(date_run_folder) if f.endswith(".png")]
        
        # Find the B images (all .jpg files in the folder)
        B_paths = [os.path.join(date_run_folder, f) for f in os.listdir(date_run_folder) if f.endswith(".jpg")]
        B_paths = sorted(B_paths, key=lambda x: int(re.findall(r'\d+', x.split('/')[-1])[0]))
            
        if len(A_path) != 1 or not B_paths:     # No valid A or B images
            raise ValueError(f"No valid A and/or B images found in {date_run_folder}!!") # 😶

        else:   # There are valid A and B images
            # Process A image
            A = Image.open(A_path[0])
            if A.mode != "RGB": # Incase a certain image gets assigned the wrong type "P" or "L"
                A = A.convert('RGB')
                ###print(f"Image mode: {A.mode}")
                ###print(F"A image from: {np.shape(A)}")
            # Reorder and resize
            A = np.array(A)
            A = torch.from_numpy(A).float()
            A = A.permute(2, 0, 1)
            A = A.unsqueeze(0)  # Add batch dimension
            A = F.interpolate(A, size=(1024, 1024), mode='bilinear', align_corners=False) # Change to size of photos
            A = A.squeeze(0)  # Remove batch dimension
                ###print(F"A image: {A.shape}")

            ### Load the A image (grayscale, if needed)
            # A = Image.open(A_path[0]).convert('L')  # Change to 'L' to avoid using rgb

            # Process B image
            B_images = []
            for B_path in B_paths:
                B_image = Image.open(B_path)  # Load each B image
                    ###print(f"B image: {np.shape(B_image)}")
                # Reorder and resize
                B_image = np.array(B_image)
                B_image = torch.from_numpy(B_image).float()
                B_image = B_image.permute(2, 0, 1)
                B_image = B_image.unsqueeze(0)  # Add batch dimension
                B_image = F.interpolate(B_image, size=(1024, 1024), mode='bilinear', align_corners=False) # Resize to size photos
                B_image = B_image.squeeze(0)  # Remove batch dimension
                    ###print(f"B image resize: {B_image.shape}")
                # B_image = Image.open(B_path).convert('L')   # Load and convert to grayscale
                B_images.append(B_image)

            # Apply the same transformation to both A and B
            A_transform = get_transform(bit_depth=8)
            B_transform = get_transform(bit_depth=8)

            A = A_transform(A)

            B_images = [B_transform(b) for b in B_images]
            B = torch.cat(B_images, dim=1)
            B = B.squeeze(0)
                ###print(F"B stack: {B.shape}")

            # Print the day and run information for user
            ###print(f"Analyzing images in {date_run_folder}")
        
        # Return the dictionary with A, B, and their respective paths
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    
# Ending Changes
##############################################################################################################

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.date_run_folders)