import torch
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image
import pytorch_msssim
import math

def load_image(img_path):
    img = Image.open(img_path).convert('L')
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)

def calculate_psnr(mse):
    return 20 * math.log10(1.0 / math.sqrt(mse))

img_dir = Path('/home/scanright/hyperspectral_pix2pix/results/512slicedata01/test_latest/images')   # Change to the path of the test you want to analyze
real_images = list(img_dir.glob('*real_B.png'))
fake_images = [img_dir / f"{img.stem.replace('_real_B', '_fake_B')}.png" for img in real_images]

mae_losses = []
mse_losses = []
ssim_values = []
psnr_values = []
cross_entropy_values = []

for real_img_path, fake_img_path in zip(real_images, fake_images):
    
    real_img = load_image(real_img_path)
    fake_img = load_image(fake_img_path)

    mae_loss = torch.nn.functional.l1_loss(real_img, fake_img)
    mae_losses.append(mae_loss.item())
    
    mse_loss = torch.nn.functional.mse_loss(real_img, fake_img)
    mse_losses.append(mse_loss.item())

    psnr = calculate_psnr(mse_loss.item())
    psnr_values.append(psnr)

    ssim_value_torch = pytorch_msssim.ssim(real_img, fake_img, data_range=1)  # PyTorch SSIM
    ssim_values.append(ssim_value_torch.item())

    cross_entropy = torch.nn.functional.binary_cross_entropy(real_img, fake_img)
    cross_entropy_values.append(cross_entropy.item())

    # do FID
    
    ssim_values.append(ssim_value_torch)
    
average_mse = np.mean(mse_losses)
average_ssim = np.mean(ssim_values)
average_psnr = np.mean(psnr_values)
average_cross_entropy = np.mean(cross_entropy_values)
average_mae = np.mean(mae_losses)

print(f"Average MAE: {average_mae}")
print(f"Average MSE: {average_mse}")
print(f"Average PSNR: {average_psnr}")
print(f"Average SSIM: {average_ssim}")
print(f"Average Cross Entropy: {average_cross_entropy}")
