import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted([os.path.join(self.dir_AB, d) for d in os.listdir(self.dir_AB) if os.path.isdir(os.path.join(self.dir_AB, d))])  # get directory paths
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
        AB_path = self.AB_paths[index]
        base_name = os.path.basename(AB_path)
        AB_path = os.path.join(AB_path, base_name)


        
        base_name_2 = base_name.replace('_ms', '')
        A_path = os.path.join(AB_path, base_name_2 + '_RGB.bmp')
        A = Image.open(A_path).convert('L')
        
        # Load hyperspectral images
        B_images = []
        #B_paths = sorted([os.path.join(AB_path, fname) for fname in os.listdir(AB_path) if self.is_image_file(fname)])

        for i in range(1, 11):      ## Change to number images in full capture
            filename = f'{base_name}_{i:02d}.png'
            B_path = os.path.join(AB_path, filename)
            #print(f'b path: {B_path}')
            B_image = Image.open(B_path)  # Convert to grayscale

            # Check min, max, and bit depth
            B_array = np.array(B_image)
            #print(f'B_image {i} - min: {B_array.min()}, max: {B_array.max()}, dtype: {B_array.dtype}, shape: {B_array.shape}')

            B_images.append(B_image)


        # apply the same transform to both A and B
        A_transform = get_transform(bit_depth = 8)
        B_transform = get_transform(bit_depth = 16)

        A = A_transform(A)

        B_images = [B_transform(b) for b in B_images]
        B = torch.stack(B_images, dim=0)
        #print(f'Transformed B shape: {B.shape}')
        #B = A 
        ## Convert the Image objects to numpy arrays
        #A_np = np.array(A)
        #B_np = np.array(B)
        #print(f'b np shape: {B_np.shape}')
        ## Flatten the arrays
        #A_flat = A_np.flatten()
        #B_flat = B_np.flatten()

        #plt.figure(figsize=(12, 6))
        #plt.subplot(1, 2, 1)
        #plt.hist(A_flat, bins=50, color='blue', alpha=0.7)
        #plt.title('Histogram of A')
        #plt.subplot(1, 2, 2)
        #plt.hist(B_flat, bins=50, color='red', alpha=0.7)
        #plt.title('Histogram of B')
        #plt.show()
        
        #plt.figure()
        #b_img = B_np[0, 0, :, :]
        #print(f'b img shape: {b_img.shape}')
        #plt.imshow(b_img, cmap='gray')
        #plt.show
        

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)