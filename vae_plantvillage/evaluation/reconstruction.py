"""
evaluation/reconstruction.py
==============================
Quantitative assessment of reconstruction quality.

Metrics
-------
1. MSE  — Mean Squared Error per pixel (lower is better).
2. MAE  — Mean Absolute Error per pixel (lower is better).
3. PSNR — Peak Signal-to-Noise Ratio in dB (higher is better).
         PSNR = 10 · log₁₀(MAX² / MSE),  MAX=1.0 for [0,1] images.
4. SSIM — Structural Similarity Index (higher is better, max=1.0).
         Computed as a simplified single-scale approximation.

Evaluation protocol (analogous to DBN report Section 8)
---------------------------------------------------------
* Collect all validation images.
* Pass through the encoder → sample z → decoder.
* Compute pixel-wise metrics averaged over the full validation set.
* Also compare VAE vs AE reconstruction quality on the same images.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.vae import VAE
from models.autoencoder import AutoEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Per-image metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def mse_per_image(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    MSE per image, shape (B,).
    Averaged over C, H, W dimensions.
    """
    return ((x_recon - x) ** 2).mean(dim=[1, 2, 3])


def mae_per_image(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (x_recon - x).abs().mean(dim=[1, 2, 3])


def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    PSNR (dB) from per-image MSE.
    PSNR = 10 · log₁₀(MAX² / MSE).
    Returns -inf for perfect reconstruction (mse == 0).
    """
    return 10.0 * torch.log10(max_val ** 2 / (mse + 1e-10))


def ssim_batch(x_recon: torch.Tensor, x: torch.Tensor,
               window_size: int = 11,
               C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    """
    Simplified single-scale SSIM computed with average-pooling windows.
    Returns a scalar tensor (mean over batch and channels).

    For a full multi-scale SSIM use the `piq` library.
    """
    # Compute local statistics via average pooling
    pad = window_size // 2

    mu_x = F.avg_pool2d(x,       window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(x_recon, window_size, stride=1, padding=pad)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sigma_x  = F.avg_pool2d(x       * x,       window_size, 1, pad) - mu_x_sq
    sigma_y  = F.avg_pool2d(x_recon * x_recon, window_size, 1, pad) - mu_y_sq
    sigma_xy = F.avg_pool2d(x       * x_recon, window_size, 1, pad) - mu_xy

    numerator   = (2 * mu_xy + C1)   * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation over a DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_reconstruction(
    model      : torch.nn.Module,
    val_loader : DataLoader,
    model_type : str = "vae",         # "vae" or "ae"
) -> Dict[str, float]:
    """
    Computes average reconstruction metrics over the entire validation set.

    Returns a dict with keys: mse, mae, psnr, ssim.
    """
    model = model.to(config.DEVICE)
    model.eval()

    total_mse  = 0.0
    total_mae  = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_images   = 0

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc=f"Evaluating {model_type.upper()} reconstruction",
                               leave=True, ncols=80):
            images = images.to(config.DEVICE)

            if model_type == "vae":
                x_recon, _, _ = model(images)
            else:  # ae
                x_recon = model(images)

            batch_mse  = mse_per_image(x_recon, images)    # (B,)
            batch_mae  = mae_per_image(x_recon, images)    # (B,)
            batch_psnr = psnr_from_mse(batch_mse)           # (B,)
            batch_ssim = ssim_batch(x_recon, images)        # scalar

            total_mse  += batch_mse.sum().item()
            total_mae  += batch_mae.sum().item()
            total_psnr += batch_psnr.sum().item()
            total_ssim += batch_ssim.item() * images.size(0)
            n_images   += images.size(0)

    results = {
        "mse":  total_mse  / n_images,
        "mae":  total_mae  / n_images,
        "psnr": total_psnr / n_images,
        "ssim": total_ssim / n_images,
    }

    print(f"\n{'─'*50}")
    print(f"  {model_type.upper()} Reconstruction Quality")
    print(f"  MSE  : {results['mse']:.6f}")
    print(f"  MAE  : {results['mae']:.6f}")
    print(f"  PSNR : {results['psnr']:.2f} dB")
    print(f"  SSIM : {results['ssim']:.4f}")
    print(f"{'─'*50}\n")

    return results


def compare_vae_ae(
    vae_metrics : Dict[str, float],
    ae_metrics  : Dict[str, float],
) -> None:
    """
    Prints a side-by-side comparison table of VAE vs AE reconstruction quality.
    Mirrors the methodology comparison table in the DBN report (Section 10.4).
    """
    header = f"{'Metric':<10} {'VAE':>12} {'AE':>12} {'Δ (VAE-AE)':>14}"
    print(f"\n{'═'*50}")
    print("  VAE vs AutoEncoder — Reconstruction Comparison")
    print(f"{'═'*50}")
    print(header)
    print("─" * 50)
    for k in ["mse", "mae", "psnr", "ssim"]:
        delta = vae_metrics[k] - ae_metrics[k]
        sign  = "+" if delta > 0 else ""
        print(f"  {k.upper():<8} {vae_metrics[k]:>12.4f} {ae_metrics[k]:>12.4f} "
              f"{sign}{delta:>12.4f}")
    print(f"{'═'*50}\n")
    print("  Note: For MSE/MAE lower is better; for PSNR/SSIM higher is better.")
    print("  Δ > 0 favours AE on MSE/MAE; Δ > 0 favours VAE on PSNR/SSIM.\n")
