"""
models/autoencoder.py
=====================
Standard deterministic Autoencoder (AE) for comparison with the VAE.

Key differences from the VAE
------------------------------
| Property              | VAE                         | AE (this file)          |
|-----------------------|-----------------------------|-------------------------|
| Latent space          | Probabilistic N(μ, σ²)      | Deterministic vector    |
| Reparameterisation    | Yes (sampling + gradient)   | No                      |
| Loss                  | Recon + β·KL divergence     | Reconstruction only     |
| Generation quality    | Smooth interpolation        | Fragmented / noisy      |
| Regularisation        | KL forces N(0,I) structure  | None (may collapse)     |

The AE uses identical convolutional architecture so results are comparable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import config


class AutoEncoder(nn.Module):
    """
    Deterministic Convolutional Autoencoder.
    Same encoder / decoder architecture as the VAE but no reparameterisation
    and no KL term — only reconstruction loss is minimised.
    """

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_encode = nn.Linear(config.FLAT_DIM, latent_dim)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.fc_decode = nn.Linear(latent_dim, config.FLAT_DIM)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_encode(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0),
                   config.ENCODER_LAST_CHANNELS,
                   config.ENCODER_FEATURE_MAP_SIZE,
                   config.ENCODER_FEATURE_MAP_SIZE)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def ae_loss(x_recon: torch.Tensor,
            x: torch.Tensor,
            loss_type: str = "mse") -> torch.Tensor:
    """Reconstruction-only loss for the standard AE."""
    if loss_type == "mse":
        return F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
    return F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
