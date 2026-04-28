"""
generation/generate.py
=======================
Synthetic image generation and latent space interpolation.

Functions
---------
1. generate_random_samples
   Sample z ~ N(0, I) and decode.  Produces novel images not in the
   training set, demonstrating the generative capability of the VAE.

2. interpolate_latent_space
   Linear interpolation between two latent codes:
       z(t) = (1 − t) · z_a  +  t · z_b,   t ∈ {0, 1/(K-1), …, 1}
   Decoding z(t) produces a smooth visual transition between two images,
   which is only possible because the VAE enforces a smooth, continuous
   latent space via the KL regularisation term.
   A deterministic AE typically produces artefact-heavy interpolations
   because its latent space has no smoothness constraint.

3. reconstruct_sample_grid
   Encodes a batch of real images and returns side-by-side
   (original, reconstruction) pairs for visual inspection.
"""

import os
from typing import Tuple, Optional

import torch
import numpy as np

import config
from models.vae import VAE


# ─────────────────────────────────────────────────────────────────────────────
# 1. Random generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_samples(
    model    : VAE,
    n_samples: int = config.NUM_GEN_SAMPLES,
    seed     : Optional[int] = None,
) -> torch.Tensor:
    """
    Generates n_samples synthetic images by sampling from the prior.

    Returns:
        images : (n_samples, 3, IMAGE_SIZE, IMAGE_SIZE) tensor in [0, 1]
    """
    model = model.to(config.DEVICE)
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z      = torch.randn(n_samples, model.latent_dim, device=config.DEVICE)
        images = model.decoder(z)

    return images.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Latent space interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_between_images(
    model   : VAE,
    img_a   : torch.Tensor,    # (3, H, W) single image
    img_b   : torch.Tensor,    # (3, H, W) single image
    steps   : int = config.INTERP_STEPS,
) -> torch.Tensor:
    """
    Encodes img_a and img_b to their posterior means,
    then linearly interpolates in latent space and decodes each step.

    Returns:
        frames : (steps, 3, IMAGE_SIZE, IMAGE_SIZE) tensor
                 frames[0] ≈ img_a reconstruction,
                 frames[-1] ≈ img_b reconstruction.
    """
    model = model.to(config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Encode both images to latent means
        za, _ = model.encoder(img_a.unsqueeze(0).to(config.DEVICE))   # (1, LATENT_DIM)
        zb, _ = model.encoder(img_b.unsqueeze(0).to(config.DEVICE))   # (1, LATENT_DIM)

        # Build interpolation steps
        alphas = torch.linspace(0.0, 1.0, steps, device=config.DEVICE)   # (steps,)
        # Broadcast: (steps, 1) * (1, LATENT_DIM)
        z_interp = (1 - alphas.unsqueeze(1)) * za + alphas.unsqueeze(1) * zb  # (steps, LATENT_DIM)

        frames = model.decoder(z_interp)   # (steps, 3, H, W)

    return frames.cpu()


def interpolate_random_pairs(
    model : VAE,
    n_pairs: int = 4,
    steps : int  = config.INTERP_STEPS,
) -> torch.Tensor:
    """
    Generates n_pairs interpolation sequences between random prior samples.
    Returns a (n_pairs * steps, 3, H, W) tensor.
    """
    model = model.to(config.DEVICE)
    model.eval()

    all_frames = []
    with torch.no_grad():
        for _ in range(n_pairs):
            za = torch.randn(1, model.latent_dim, device=config.DEVICE)
            zb = torch.randn(1, model.latent_dim, device=config.DEVICE)
            alphas  = torch.linspace(0, 1, steps, device=config.DEVICE)
            z_interp = (1 - alphas.unsqueeze(1)) * za + alphas.unsqueeze(1) * zb
            frames  = model.decoder(z_interp)
            all_frames.append(frames)

    return torch.cat(all_frames, dim=0).cpu()   # (n_pairs*steps, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reconstruction grid builder
# ─────────────────────────────────────────────────────────────────────────────

def build_reconstruction_grid(
    model  : VAE,
    loader : torch.utils.data.DataLoader,
    n      : int = config.NUM_RECON_DISPLAY,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (originals, reconstructions) each of shape (n, 3, H, W).
    Uses the deterministic posterior mean for reproducibility.
    """
    model = model.to(config.DEVICE)
    model.eval()

    images_all = []
    for batch, _ in loader:
        images_all.append(batch)
        if sum(x.size(0) for x in images_all) >= n:
            break

    images = torch.cat(images_all)[:n].to(config.DEVICE)

    with torch.no_grad():
        recons = model.reconstruct(images)

    return images.cpu(), recons.cpu()
