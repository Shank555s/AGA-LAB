"""
training/train.py
=================
Full training loop for both the VAE and the comparison AutoEncoder.

Features
--------
* Per-epoch tracking of total loss, reconstruction loss, and KL loss
* tqdm progress bar with live loss display
* Checkpoint saving every N epochs and at the final epoch
* Returns history dict with all recorded losses for downstream plotting
"""

import os
import time
from typing import Tuple, Dict, List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.vae        import VAE, vae_loss
from models.autoencoder import AutoEncoder, ae_loss


# ─────────────────────────────────────────────────────────────────────────────
# VAE training
# ─────────────────────────────────────────────────────────────────────────────

def train_vae(
    model       : VAE,
    train_loader: DataLoader,
    val_loader  : DataLoader,
) -> Dict[str, List[float]]:
    """
    Trains the VAE for config.EPOCHS epochs.

    Args:
        model        : VAE instance (will be moved to config.DEVICE)
        train_loader : DataLoader for the training split
        val_loader   : DataLoader for the validation split

    Returns:
        history : dict with keys
                  "train_total", "train_recon", "train_kl",
                  "val_total",   "val_recon",   "val_kl"
                  Each value is a list of per-epoch scalar losses.
    """
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)

    # Cosine annealing LR scheduler for smooth convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-5
    )

    history: Dict[str, List[float]] = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total":   [], "val_recon":   [], "val_kl":   [],
    }

    print(f"\n{'═'*60}")
    print(f"  Training VAE  │  Device: {config.DEVICE}  │  "
          f"Epochs: {config.EPOCHS}  │  LR: {config.LEARNING_RATE}")
    print(f"{'═'*60}\n")

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        # ── Training phase ─────────────────────────────────────────────────────
        model.train()
        epoch_total = epoch_recon = epoch_kl = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{config.EPOCHS} [Train]",
                    leave=False, ncols=90)
        for images, _ in pbar:
            images = images.to(config.DEVICE)

            optimizer.zero_grad()
            x_recon, mu, log_var = model(images)
            loss, recon, kl = vae_loss(x_recon, images, mu, log_var,
                                       loss_type=config.LOSS_TYPE,
                                       beta=config.BETA)
            loss.backward()
            # Gradient clipping prevents occasional large updates in early epochs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             recon=f"{recon.item():.4f}",
                             kl=f"{kl.item():.4f}")

        n_batches = len(train_loader)
        history["train_total"].append(epoch_total / n_batches)
        history["train_recon"].append(epoch_recon / n_batches)
        history["train_kl"].append(epoch_kl    / n_batches)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_total = val_recon = val_kl = 0.0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(config.DEVICE)
                x_recon, mu, log_var = model(images)
                loss, recon, kl = vae_loss(x_recon, images, mu, log_var,
                                           loss_type=config.LOSS_TYPE,
                                           beta=config.BETA)
                val_total += loss.item()
                val_recon += recon.item()
                val_kl    += kl.item()

        n_val = len(val_loader)
        history["val_total"].append(val_total / n_val)
        history["val_recon"].append(val_recon / n_val)
        history["val_kl"].append(val_kl    / n_val)

        scheduler.step()

        # ── Logging ────────────────────────────────────────────────────────────
        elapsed = (time.time() - t_start) / 60
        print(f"Epoch {epoch:3d}/{config.EPOCHS}  │  "
              f"Train: total={history['train_total'][-1]:.4f}  "
              f"recon={history['train_recon'][-1]:.4f}  "
              f"kl={history['train_kl'][-1]:.4f}  │  "
              f"Val: total={history['val_total'][-1]:.4f}  "
              f"recon={history['val_recon'][-1]:.4f}  │  "
              f"{elapsed:.1f} min elapsed")

        # ── Checkpoint saving ──────────────────────────────────────────────────
        is_best  = history["val_total"][-1] < best_val_loss
        save_now = (epoch % config.SAVE_EVERY == 0) or (epoch == config.EPOCHS) or is_best

        if is_best:
            best_val_loss = history["val_total"][-1]

        if save_now:
            _save_checkpoint(model, optimizer, epoch, history, "vae",
                             tag="best" if is_best else f"epoch{epoch:03d}")

    total_time = (time.time() - t_start) / 60
    print(f"\nVAE training complete in {total_time:.1f} min. "
          f"Best val loss: {best_val_loss:.4f}\n")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# AutoEncoder training  (comparison baseline)
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    model       : AutoEncoder,
    train_loader: DataLoader,
    val_loader  : DataLoader,
) -> Dict[str, List[float]]:
    """
    Trains the comparison AutoEncoder.
    Interface identical to train_vae for easy swapping.
    """
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-5
    )

    history: Dict[str, List[float]] = {
        "train_recon": [], "val_recon": []
    }

    print(f"\n{'═'*60}")
    print(f"  Training AutoEncoder (AE baseline)  │  Device: {config.DEVICE}")
    print(f"{'═'*60}\n")

    best_val = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        ep_recon = 0.0
        for images, _ in tqdm(train_loader,
                               desc=f"AE Epoch {epoch:3d}/{config.EPOCHS}",
                               leave=False, ncols=80):
            images = images.to(config.DEVICE)
            optimizer.zero_grad()
            x_recon = model(images)
            loss    = ae_loss(x_recon, images, config.LOSS_TYPE)
            loss.backward()
            optimizer.step()
            ep_recon += loss.item()
        history["train_recon"].append(ep_recon / len(train_loader))

        # ── Validate ────────────────────────────────────────────────────────────
        model.eval()
        val_recon = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(config.DEVICE)
                x_recon = model(images)
                val_recon += ae_loss(x_recon, images, config.LOSS_TYPE).item()
        history["val_recon"].append(val_recon / len(val_loader))

        scheduler.step()

        is_best = history["val_recon"][-1] < best_val
        if is_best:
            best_val = history["val_recon"][-1]
            _save_checkpoint(model, optimizer, epoch, history, "ae", tag="best")

        if epoch % config.SAVE_EVERY == 0:
            print(f"AE Epoch {epoch:3d}  │  Train recon: "
                  f"{history['train_recon'][-1]:.4f}  │  "
                  f"Val recon: {history['val_recon'][-1]:.4f}")

    print(f"AE training complete. Best val recon: {best_val:.4f}\n")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utility
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, epoch, history, model_name, tag=""):
    """Saves model weights, optimizer state, and loss history to disk."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    fname = os.path.join(config.CHECKPOINT_DIR,
                         f"{model_name}_checkpoint_{tag}.pt")
    torch.save({
        "epoch":     epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history":         history,
    }, fname)
    # Only print for best / final saves to avoid noise
    if "best" in tag or "epoch" not in tag:
        print(f"  [OK] Checkpoint saved: {fname}")


def load_checkpoint(model, checkpoint_path: str):
    """
    Loads model weights from a saved checkpoint.

    Args:
        model           : VAE or AutoEncoder instance
        checkpoint_path : full path to the .pt file

    Returns:
        epoch   : epoch at which the checkpoint was saved
        history : loss history dict
    """
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"[OK] Loaded checkpoint from '{checkpoint_path}' (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt.get("history", {})
