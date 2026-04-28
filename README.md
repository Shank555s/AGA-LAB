# VAE for PlantVillage Crop Disease Image Compression and Synthetic Generation

**AGA Lab CIE Project — April 2026**

---

## Overview

This project implements a **Convolutional Variational Autoencoder (VAE)** trained on the
PlantVillage crop-disease image dataset. The model compresses leaf images into a compact
128-dimensional latent space and generates novel synthetic plant-disease images by sampling
from the learned latent distribution.

The evaluation framework covers three dimensions — reconstruction quality, latent space
structure, and downstream ML utility — mirroring the rigorous three-dimensional assessment
used in the companion DBN synthetic healthcare data project.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download PlantVillage dataset

**Option A — Kaggle CLI (recommended)**
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/plantvillage
```

**Option B — Manual**
Download from https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
and extract to `data/plantvillage/` so that it follows ImageFolder layout:
```
data/plantvillage/
    Apple___Apple_scab/
        image0001.JPG
        ...
    Apple___Black_rot/
        ...
```

**Option C — Synthetic fallback (no download needed)**
If the directory is absent, the code automatically generates a small synthetic
placeholder dataset for development and testing. Set
`USE_SYNTHETIC_FALLBACK = True` in `data/loader.py` (default).

### 3. Run the pipeline

```bash
python main.py
```

---

## Project Structure

```
vae_plantvillage/
│── main.py               Entry point — runs the full pipeline
│── config.py             All hyperparameters and paths
│── requirements.txt      Python dependencies
│── README.md             This file
│
├── data/
│   ├── loader.py         Dataset loading, train/val split
│   └── preprocess.py     Image transforms and augmentation
│
├── models/
│   ├── vae.py            CNN VAE (Encoder, Decoder, ELBO loss)
│   └── autoencoder.py    Deterministic AE (comparison baseline)
│
├── training/
│   └── train.py          Training loops with checkpointing
│
├── evaluation/
│   ├── reconstruction.py MSE, MAE, PSNR, SSIM metrics
│   ├── latent_space.py   t-SNE, PCA, Silhouette, Davies-Bouldin
│   └── classifier.py     TRTR vs TSTR downstream utility
│
├── generation/
│   └── generate.py       Random sampling + latent interpolation
│
├── visualization/
│   └── plots.py          All matplotlib figures
│
└── outputs/
    ├── checkpoints/      Model weights (.pt files)
    ├── reconstructions/  Input vs output comparison grids + loss curves
    ├── generated/        Synthetic images + interpolation strips
    └── latent/           t-SNE and PCA scatter plots + utility chart
```

---

## Key Configuration (config.py)

| Parameter      | Default | Description                              |
|----------------|---------|------------------------------------------|
| `IMAGE_SIZE`   | 128     | Resize all images to 128×128             |
| `LATENT_DIM`   | 128     | Dimensionality of the latent space z     |
| `BATCH_SIZE`   | 32      | Reduce to 16 if GPU memory is limited    |
| `EPOCHS`       | 50      | Training epochs per model                |
| `BETA`         | 1.0     | KL weight (>1 for β-VAE disentanglement) |
| `MAX_CLASSES`  | 10      | Use only the first N disease classes     |
| `LOSS_TYPE`    | "mse"   | Reconstruction loss: "mse" or "bce"     |

---

## Outputs Produced

| File                                  | Description                              |
|---------------------------------------|------------------------------------------|
| `outputs/reconstructions/loss_curves_vae.png` | Training/val loss curves (3 panels) |
| `outputs/reconstructions/reconstructions.png` | Original vs reconstructed image grid |
| `outputs/reconstructions/loss_comparison.png` | VAE vs AE validation loss overlay   |
| `outputs/generated/generated_samples.png`     | 64 synthetic images from prior       |
| `outputs/generated/interpolation.png`         | Latent space interpolation strips    |
| `outputs/latent/latent_pca.png`               | PCA 2D scatter by disease class      |
| `outputs/latent/latent_tsne.png`              | t-SNE 2D scatter by disease class    |
| `outputs/latent/utility_comparison.png`       | TRTR vs TSTR accuracy bar chart      |
| `outputs/checkpoints/vae_checkpoint_best.pt`  | Best VAE weights                     |
| `outputs/checkpoints/ae_checkpoint_best.pt`   | Best AE weights                      |

---

## VAE Architecture

```
Encoder
  Conv(3→32, k=4, s=2)  → 64×64   + BN + LeakyReLU(0.2)
  Conv(32→64, k=4, s=2) → 32×32   + BN + LeakyReLU(0.2)
  Conv(64→128,k=4, s=2) → 16×16   + BN + LeakyReLU(0.2)
  Conv(128→256,k=4,s=2) →  8×8    + BN + LeakyReLU(0.2)
  Flatten → 16,384
  FC → μ (128)     FC → log σ² (128)

Reparameterisation:  z = μ + ε·σ,  ε ~ N(0,I)

Decoder
  FC(128 → 16,384)
  Reshape → 256×8×8
  ConvTranspose(256→128) → 16×16  + BN + ReLU
  ConvTranspose(128→64)  → 32×32  + BN + ReLU
  ConvTranspose(64→32)   → 64×64  + BN + ReLU
  ConvTranspose(32→3)    →128×128 + Sigmoid

Total parameters: ~3.7 million
```

---

## References

- Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.
- Hughes, D.P. & Salathé, M. (2016). An open access repository of images on plant health. arXiv:1511.08060.
- Higgins, I. et al. (2017). β-VAE: Learning Basic Visual Concepts. ICLR.
