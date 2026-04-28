"""
visualization/plots.py
=======================
All plotting functions for the VAE PlantVillage project.

Saved figures
-------------
1. loss_curves_vae.png         — VAE total / recon / KL loss vs epoch
2. loss_curves_ae.png          — AE reconstruction loss vs epoch
3. loss_comparison.png         — VAE vs AE val reconstruction loss overlay
4. reconstructions.png         — Grid of (original | reconstruction) pairs
5. generated_samples.png       — Grid of VAE-sampled synthetic images
6. latent_pca.png              — 2D PCA scatter of latent space
7. latent_tsne.png             — 2D t-SNE scatter of latent space
8. interpolation.png           — Latent-space interpolation strips
9. utility_comparison.png      — Bar chart of TRTR vs TSTR classifier accuracy
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

import config


# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
})

_PALETTE = plt.cm.tab10.colors   # consistent class colours


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_vae_loss_curves(history: Dict[str, List[float]]) -> None:
    """Three-panel plot: total, reconstruction, and KL divergence losses."""
    epochs = range(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("VAE Training Loss Curves", fontsize=13, fontweight="bold")

    panels = [
        ("train_total",  "val_total",  "Total Loss (Recon + β·KL)"),
        ("train_recon",  "val_recon",  "Reconstruction Loss (MSE)"),
        ("train_kl",     "val_kl",     "KL Divergence"),
    ]
    for ax, (tr_key, va_key, title) in zip(axes, panels):
        ax.plot(epochs, history[tr_key], label="Train", color="#1f77b4", linewidth=1.5)
        ax.plot(epochs, history[va_key], label="Val",   color="#ff7f0e",
                linewidth=1.5, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    _save(fig, os.path.join(config.RECON_DIR, "loss_curves_vae.png"))


def plot_ae_loss_curves(history: Dict[str, List[float]]) -> None:
    """Single-panel AE reconstruction loss."""
    epochs = range(1, len(history["train_recon"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_recon"], label="Train", color="#1f77b4", linewidth=1.5)
    ax.plot(epochs, history["val_recon"],   label="Val",   color="#ff7f0e",
            linewidth=1.5, linestyle="--")
    ax.set_title("AutoEncoder Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(config.RECON_DIR, "loss_curves_ae.png"))


def plot_loss_comparison(
    vae_history: Dict[str, List[float]],
    ae_history : Dict[str, List[float]],
) -> None:
    """Overlay of VAE vs AE validation reconstruction loss."""
    epochs = range(1, len(vae_history["val_recon"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, vae_history["val_recon"], label="VAE (val recon)",
            color="#2ca02c", linewidth=2)
    ax.plot(epochs, ae_history["val_recon"],  label="AE  (val recon)",
            color="#d62728", linewidth=2, linestyle="--")
    ax.set_title("VAE vs AutoEncoder — Validation Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(config.RECON_DIR, "loss_comparison.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reconstruction grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstructions(
    originals     : torch.Tensor,   # (N, 3, H, W) in [0,1]
    reconstructions: torch.Tensor,  # (N, 3, H, W) in [0,1]
    title         : str = "Original (top) vs Reconstruction (bottom)",
) -> None:
    """
    Saves a two-row grid: top row originals, bottom row reconstructions.
    """
    n = min(originals.size(0), 8)   # show at most 8 pairs
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for i in range(n):
        # Original
        img = originals[i].permute(1, 2, 0).numpy().clip(0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=8)

        # Reconstruction
        rec = reconstructions[i].permute(1, 2, 0).numpy().clip(0, 1)
        axes[1, i].imshow(rec)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Recon", fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(config.RECON_DIR, "reconstructions.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generated samples
# ─────────────────────────────────────────────────────────────────────────────

def plot_generated_samples(
    samples: torch.Tensor,   # (N, 3, H, W) in [0,1]
    nrow   : int = 8,
) -> None:
    """
    Saves a grid of randomly generated synthetic plant images.
    """
    n   = min(samples.size(0), nrow * nrow)
    rows = (n + nrow - 1) // nrow

    fig, axes = plt.subplots(rows, nrow, figsize=(nrow * 1.5, rows * 1.5))
    fig.suptitle("VAE — Synthetic Generated Samples (z ~ N(0,I))",
                 fontsize=11, fontweight="bold")

    axes = np.array(axes).flatten()
    for i in range(len(axes)):
        if i < n:
            img = samples[i].permute(1, 2, 0).numpy().clip(0, 1)
            axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()
    _save(fig, os.path.join(config.GEN_DIR, "generated_samples.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Latent space scatter plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_latent_scatter(
    embedding   : np.ndarray,   # (N, 2)
    labels      : np.ndarray,   # (N,) integer class IDs
    class_names : list,
    method      : str = "t-SNE",
    filename    : str = "latent_tsne.png",
) -> None:
    """
    Scatter plot of 2D latent embedding coloured by class label.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    unique_labels = np.unique(labels)

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if label < len(class_names) else f"Class {label}"
        # Shorten long class names (e.g. "Apple___Apple_scab" → "Apple scab")
        short_name = name.replace("___", " ").replace("_", " ")
        if len(short_name) > 20:
            short_name = short_name[:18] + "…"
        color = _PALETTE[idx % len(_PALETTE)]
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=8, alpha=0.6, color=color, label=short_name)

    ax.set_title(f"Latent Space — {method} Projection", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{method} Component 1")
    ax.set_ylabel(f"{method} Component 2")
    ax.legend(markerscale=2, loc="best", framealpha=0.7,
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, os.path.join(config.LATENT_DIR, filename))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interpolation strip
# ─────────────────────────────────────────────────────────────────────────────

def plot_interpolation(
    frames  : torch.Tensor,   # (n_pairs * steps, 3, H, W)
    steps   : int = config.INTERP_STEPS,
    n_pairs : int = 4,
) -> None:
    """
    Displays interpolation strips where each row is one pair's sequence.
    """
    fig, axes = plt.subplots(n_pairs, steps,
                              figsize=(steps * 1.5, n_pairs * 1.5))
    fig.suptitle("Latent Space Interpolation  (left → right: z_A → z_B)",
                 fontsize=11, fontweight="bold")

    for row in range(n_pairs):
        for col in range(steps):
            idx  = row * steps + col
            img  = frames[idx].permute(1, 2, 0).numpy().clip(0, 1)
            ax   = axes[row, col] if n_pairs > 1 else axes[col]
            ax.imshow(img)
            ax.axis("off")
        if n_pairs > 1:
            axes[row, 0].set_ylabel(f"Pair {row+1}", fontsize=8, rotation=90)

    plt.tight_layout()
    _save(fig, os.path.join(config.GEN_DIR, "interpolation.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Downstream utility bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_utility_comparison(
    results: Dict[str, Dict[str, float]],
) -> None:
    """
    Grouped bar chart comparing TRTR vs TSTR accuracy for each classifier.
    Mirrors the utility table visualisation in the DBN report (Section 9).
    """
    clf_names = list(results.keys())
    trtr_accs = [results[c]["trtr_acc"] for c in clf_names]
    tstr_accs = [results[c]["tstr_acc"] for c in clf_names]
    ratios    = [results[c]["utility_ratio"] for c in clf_names]

    x     = np.arange(len(clf_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Downstream Classification Utility — TRTR vs TSTR",
                 fontsize=12, fontweight="bold")

    # Left: grouped bars
    ax = axes[0]
    bars1 = ax.bar(x - width/2, trtr_accs, width, label="TRTR (real)",
                   color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + width/2, tstr_accs, width, label="TSTR (synthetic)",
                   color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=12, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Accuracy per Classifier")
    ax.grid(axis="y", alpha=0.3)
    for bar in (*bars1, *bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    # Right: utility ratio
    ax2 = axes[1]
    ax2.bar(clf_names, ratios, color="#2ca02c", alpha=0.85)
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2,
                label="Perfect utility (ratio=1.0)")
    ax2.axhline(y=0.85, color="orange", linestyle=":", linewidth=1.2,
                label="High-quality threshold (0.85)")
    ax2.set_ylabel("Utility Ratio (TSTR / TRTR)")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Utility Ratio by Classifier")
    ax2.set_xticklabels(clf_names, rotation=12, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(config.LATENT_DIR, "utility_comparison.png"))
