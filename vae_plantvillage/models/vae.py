"""
models/vae.py
=============
Convolutional Variational Autoencoder (VAE) for 128×128 RGB plant images.

Mathematical formulation
------------------------
A VAE models the joint distribution  p(x, z) = p(x|z) · p(z)  where:
    • z ~ N(0, I)              — isotropic Gaussian prior over latent codes
    • p(x|z) = Decoder(z)     — likelihood of image x given latent code z
    • q(z|x) = N(μ(x), σ²(x)) — approximate posterior (encoder output)

The training objective is the Evidence Lower BOund (ELBO):

    ELBO = E_q[ log p(x|z) ]  −  β · KL( q(z|x) ‖ p(z) )
           ─────────────────     ─────────────────────────
           Reconstruction term      Regularisation term

where KL( N(μ, σ²) ‖ N(0, I) ) = −½ Σ (1 + log σ² − μ² − σ²).

The reparameterisation trick makes the sampling step differentiable:
    z = μ + ε · σ,    ε ~ N(0, I)

Architecture (128×128 input)
----------------------------
Encoder
    Conv(3→32, k=4, s=2, p=1)  → 64×64  + BN + LeakyReLU(0.2)
    Conv(32→64, k=4, s=2, p=1) → 32×32  + BN + LeakyReLU(0.2)
    Conv(64→128,k=4, s=2, p=1) → 16×16  + BN + LeakyReLU(0.2)
    Conv(128→256,k=4,s=2, p=1) →  8×8   + BN + LeakyReLU(0.2)
    Flatten → 16 384
    FC → μ       (LATENT_DIM,)
    FC → log σ²  (LATENT_DIM,)

Decoder (mirror of encoder)
    FC(LATENT_DIM → 16 384)
    Reshape → 256 × 8 × 8
    ConvTranspose(256→128,k=4,s=2,p=1) → 16×16  + BN + ReLU
    ConvTranspose(128→64, k=4,s=2,p=1) → 32×32  + BN + ReLU
    ConvTranspose(64→32,  k=4,s=2,p=1) → 64×64  + BN + ReLU
    ConvTranspose(32→3,   k=4,s=2,p=1) → 128×128 + Sigmoid  (output ∈ [0,1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import config


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Maps an input image x  (B, 3, 128, 128)
    to the parameters of the variational posterior:
        μ        (B, LATENT_DIM)
        log σ²   (B, LATENT_DIM)
    """

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()

        # ── Convolutional feature extractor ───────────────────────────────────
        self.conv_layers = nn.Sequential(
            # Layer 1: 3 × 128 × 128 → 32 × 64 × 64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 32 × 64 × 64 → 64 × 32 × 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64 × 32 × 32 → 128 × 16 × 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 128 × 16 × 16 → 256 × 8 × 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Latent projection heads ────────────────────────────────────────────
        self.fc_mu      = nn.Linear(config.FLAT_DIM, latent_dim)
        self.fc_log_var = nn.Linear(config.FLAT_DIM, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 128, 128) image tensor in [0, 1]
        Returns:
            mu:      (B, LATENT_DIM)
            log_var: (B, LATENT_DIM)
        """
        h = self.conv_layers(x)          # (B, 256, 8, 8)
        h = h.view(h.size(0), -1)        # (B, 16384)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Maps a latent code z  (B, LATENT_DIM)
    back to a reconstructed image  (B, 3, 128, 128)  with pixel values in [0, 1].
    """

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()

        # Project latent code back to feature-map dimensionality
        self.fc = nn.Linear(latent_dim, config.FLAT_DIM)

        # ── Transposed convolutional upsampler ────────────────────────────────
        self.deconv_layers = nn.Sequential(
            # Layer 1: 256 × 8 × 8 → 128 × 16 × 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 2: 128 × 16 × 16 → 64 × 32 × 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: 64 × 32 × 32 → 32 × 64 × 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 4: 32 × 64 × 64 → 3 × 128 × 128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),   # constrain output to [0, 1] to match input range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, LATENT_DIM) latent code
        Returns:
            x_recon: (B, 3, 128, 128) reconstructed image in [0, 1]
        """
        h = self.fc(z)                                                         # (B, 16384)
        h = h.view(h.size(0),
                   config.ENCODER_LAST_CHANNELS,
                   config.ENCODER_FEATURE_MAP_SIZE,
                   config.ENCODER_FEATURE_MAP_SIZE)                            # (B, 256, 8, 8)
        x_recon = self.deconv_layers(h)                                        # (B, 3, 128, 128)
        return x_recon


# ─────────────────────────────────────────────────────────────────────────────
# VAE  (Encoder + Reparameterisation + Decoder)
# ─────────────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """
    Variational Autoencoder combining Encoder, reparameterisation trick,
    and Decoder into a single nn.Module.

    Forward pass returns:
        x_recon   — reconstructed image
        mu        — posterior mean
        log_var   — posterior log-variance
    """

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    # ── Reparameterisation trick ───────────────────────────────────────────────
    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample z = μ + ε · exp(½ log σ²)  with  ε ~ N(0, I).

        During inference (model.eval()) this still samples; call
        model.encoder(x)[0]  directly to get the deterministic mean.
        """
        std = torch.exp(0.5 * log_var)      # σ = exp(½ log σ²)
        eps = torch.randn_like(std)          # ε ~ N(0, I), same shape as std
        return mu + eps * std                # z with gradient through μ and σ

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var  = self.encoder(x)
        z            = self.reparameterise(mu, log_var)
        x_recon      = self.decoder(z)
        return x_recon, mu, log_var

    def generate(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample n_samples images from the prior p(z) = N(0, I)."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction using the posterior mean (no sampling)."""
        with torch.no_grad():
            mu, _ = self.encoder(x)
            return self.decoder(mu)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the posterior mean as the latent representation."""
        with torch.no_grad():
            mu, _ = self.encoder(x)
            return mu


# ─────────────────────────────────────────────────────────────────────────────
# ELBO Loss
# ─────────────────────────────────────────────────────────────────────────────

def vae_loss(x_recon: torch.Tensor,
             x: torch.Tensor,
             mu: torch.Tensor,
             log_var: torch.Tensor,
             loss_type: str = "mse",
             beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the negative ELBO:
        L = Recon_Loss + β · KL

    Args:
        x_recon   : (B, 3, 128, 128) decoder output
        x         : (B, 3, 128, 128) original input
        mu        : (B, LATENT_DIM)
        log_var   : (B, LATENT_DIM)
        loss_type : "mse" or "bce"
        beta      : KL weight

    Returns:
        total_loss, recon_loss, kl_loss  (all scalar tensors)
    """
    batch_size = x.size(0)

    # ── Reconstruction loss ────────────────────────────────────────────────────
    if loss_type == "mse":
        # Mean Squared Error summed over pixels, averaged over batch
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size
    elif loss_type == "bce":
        # Binary Cross-Entropy (requires inputs and targets in [0,1])
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / batch_size
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'mse' or 'bce'.")

    # ── KL Divergence: -½ Σ (1 + log σ² - μ² - σ²) ───────────────────────────
    # Summed over latent dimensions, averaged over batch
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def model_summary(model: nn.Module) -> None:
    """Prints a brief parameter count summary."""
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'─'*50}")
    print(f"  Model  : {model.__class__.__name__}")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable params    : {train:,}")
    print(f"  Latent dimension    : {model.latent_dim}")
    print(f"{'─'*50}\n")
