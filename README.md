# Image Super-Resolution using Perceptual Loss with Variational Autoencoder (VAE)

This repository contains an implementation of an image super-resolution method using Variational Autoencoder (VAE) and Perceptual loss in PyTorch. The VAE is trained to generate high-resolution images from low-resolution inputs.

## Overview

The project consists of the following components:

1. **Data Preparation**: Preprocessing of the dataset (Oxford-IIIT Pets) for training and testing the VAE model.

2. **Model Architecture**: Implementation of the VAE model architecture consisting of an encoder and a decoder.

3. **Loss Functions**: Definition of custom loss functions including reconstruction loss and perceptual loss.

4. **Training and Validation**: Training and validation loop to train the VAE model.

5. **Testing**: Evaluation of the trained model using test data, including PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Matplotlib
- NumPy
- scikit-image
- torchvision
- Pillow

## Steps

1. Clone the repository

```bash
git clone https://github.com/nujitha99/PerceptualLossVAE
```

2. Install the required Python packages using pip:

```bash
pip install torch torchvision matplotlib numpy scikit-image Pillow
```

3. Change directory

```bash
cd PerceptualLossVAE
```

4.  Run the Python script
```bash
python main.py
```

## Results
- Average PSNR: 1.32 dB
- Average SSIM: 0.02
