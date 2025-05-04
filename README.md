# Binocular Camera Pix2Pix Model

## Overview

This repository contains the code, models, and supporting materials for a research project using a modified Pix2Pix model to synthesize depth-aware focal stacks from a single image input. The system uses a binocular (dual-camera) setup along with an EDoF (Extended Depth of Field) lens and a varifocal camera to generate depth-variant imagery. This project builds on the original [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) model, extending it for binocular input and synthetic refocusing.

## Example Results:
Presented and discussed in greater detail in the paper:  
**ADD PAPER AND SUPPLEMENT HERE**  

Data is presented as: EDoF image | Generated images | Ground truth images  

 
- 512x512 Model Results
  ![512slice_supplement](https://github.com/user-attachments/assets/c9c0e727-c447-4e9e-89a0-e993dec5239d)

- 1024x1024 Model Results
  ![1024slice_supplement](https://github.com/user-attachments/assets/8f408938-c8fc-4466-a78d-0acd02b005c9)

- No Slice Model Results
  ![noslice_supplement](https://github.com/user-attachments/assets/4099cac7-156a-4b19-bbd0-424b89ecc72a)


---

## Getting Started
### Installation

- Create and activate virtual environment:

```bash
conda create --name pix2pix python=3.9
conda activate pix2pix
```

- Install dependencies via:
  - requirements.txt

Or:

```bash
conda install torch
conda install torchvision
pip install dominate
pip install visdom
pip install wandb
```

- Clone this repo:

```bash
git clone https://github.com/sccanright/binocular_pix2pix
cd binocular_pix2pix
```

### Download a pretrained model
- Pre-trained models are available for download and unzipping
  - release: [models](https://github.com/sccanright/binocular_pix2pix/releases/tag/models)
    - Make sure it is saved within the binocular_pix2pix folder

```bash
# Download the checkpoints model release:
wget https://github.com/sccanright/binocular_pix2pix/releases/download/models/checkpoints.zip -O checkpoints.zip
unzip checkpoints.zip
``` 

### Test any of the trained models
- Test datasets are available for download and unzipping
  - release: [data](https://github.com/sccanright/binocular_pix2pix/releases/tag/data)
    - Make sure they are saved within the binocular_pix2pix/datasets folder

- Download and unzip the test sets to ./datasets:

```bash
# Download zipped folders
wget https://github.com/sccanright/binocular_pix2pix/releases/download/data/512_SLICED_FIELDSET.zip -O datasets/512_SLICED_FIELDSET.zip
wget https://github.com/sccanright/binocular_pix2pix/releases/download/data/1024_SLICED_FIELDSET.zip -O datasets/1024_SLICED_FIELDSET.zip
wget https://github.com/sccanright/binocular_pix2pix/releases/download/data/NO_SLICE_FIELDSET.zip -O datasets/NO_SLICE_FIELDSET.zip

# Unzip to the correct folder
unzip datasets/512_SLICED_FIELDSET.zip -d datasets/
unzip datasets/1024_SLICED_FIELDSET.zip -d datasets/
unzip datasets/NO_SLICE_FIELDSET.zip -d datasets/
```

- Run the tests:

```bash
python test.py --dataroot ./datasets/512_SLICED_FIELDSET --name 512slicedata01 --model pix2pix --gpu_ids 0 --netG unet_512 --input_nc 3 --output_nc 30

python test.py --dataroot ./datasets/1024_SLICED_FIELDSET --name 1024slicedata01 --model pix2pix --gpu_ids 0 --netG unet_1024 --input_nc 3 --output_nc 30

python test.py --dataroot ./datasets/NO_SLICE_FIELDSET --name noslicedata01 --model pix2pix --gpu_ids 0 --netG unet_1024 --input_nc 3 --output_nc 30
```

- Locate results:
```bash
bash ./results/NAME/test_latest
```

### Training your models

- Train a new model:
(Update netG and input/output channels as needed)
```bash
python train.py --dataroot ./datasets/FIELDSET --name NAMEofMODEL --model pix2pix --gpu_ids 0 --netG unet_1024 --input_nc 3 --output_nc 30
```

## Prerequisites
- Linux, macOS, or Windows with WSL
- Python 3.8+
- CPU or NVIDIA GPU + CUDA CuDNN

## Citation
If you use this code in your research, please cite both this repository and the original Pix2Pix project:

```bash
@misc{canright2025binocularpix2pix,
  author = {Slater Canright},
  title = {Binocular Camera Pix2Pix Model},
  year = {2025},
  url = {https://github.com/sccanright/binocular_pix2pix}
}

@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```

## New Code
- ButtonCapture.py
- crop.py
- slice.py
- reorganize_data.py
- analyze.py

## Optional Files
### Qualitative optimization code
- exposure_test.py
- GPIOTest.py
- GPIOZeroTest.py

### SolidWorks CAD files
- Camera Box
- Box Lid
- EDoF Holder
- EDoF Lid
- Lens Mount - Left
- Lens Mount - Right

### Download Optionals
- SolidWorks and extra code
  - [exras](https://github.com/sccanright/binocular_pix2pix/tree/extras)

```bash
# Clone just the extras branch directly:
git clone --branch extras --single-branch https://github.com/sccanright/binocular_pix2pix
```
  

## Related Projects

- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | [px2pix](https://github.com/phillipi/pix2pix)

## Acknowledgments

Our code comes directly from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), with slight script modifications to function with our expanded data requirements.
