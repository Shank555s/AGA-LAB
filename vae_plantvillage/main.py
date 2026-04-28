"""
main.py
=======
Entry point for the VAE PlantVillage project.

Executes the full pipeline in order:
    1.  Seed everything for reproducibility
    2.  Create output directories
    3.  Load and split the PlantVillage dataset
    4.  Build and summarise the VAE model
    5.  Train the VAE
    6.  Train the comparison AutoEncoder (AE)
    7.  Plot all loss curves
    8.  Evaluate reconstruction quality (VAE and AE)
    9.  Generate and save synthetic images
    10. Generate and save latent interpolations
    11. Extract latent representations
    12. Visualise latent space (PCA + t-SNE)
    13. Compute cluster quality metrics
    14. Run downstream classification utility (TRTR vs TSTR)
    15. Print the final evaluation summary

Run:
    python main.py

All outputs are saved to the outputs/ directory.
"""

import sys
import os
import random
import numpy as np
import torch

# Force UTF-8 output on Windows so Unicode box-drawing characters don't crash
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Project imports ─────────────────────────────────────────────────────────
import config
from data.loader              import get_dataloaders
from models.vae               import VAE, model_summary
from models.autoencoder       import AutoEncoder
from training.train           import train_vae, train_autoencoder, load_checkpoint
from evaluation.reconstruction import evaluate_reconstruction, compare_vae_ae
from evaluation.latent_space   import extract_latents, pca_projection, \
                                       tsne_projection, cluster_quality_metrics
from evaluation.classifier     import evaluate_downstream_utility
from generation.generate       import generate_random_samples, \
                                       interpolate_random_pairs, \
                                       build_reconstruction_grid
from visualization.plots       import (
    plot_vae_loss_curves, plot_ae_loss_curves, plot_loss_comparison,
    plot_reconstructions, plot_generated_samples,
    plot_latent_scatter, plot_interpolation, plot_utility_comparison,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────────────────────

def create_output_dirs() -> None:
    for d in [config.CHECKPOINT_DIR, config.RECON_DIR,
              config.GEN_DIR, config.LATENT_DIR, config.REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 65)
    print("  VAE for PlantVillage Crop Disease Compression & Generation")
    print("═" * 65)

    # ── 1. Setup ──────────────────────────────────────────────────────────────
    set_seed()
    create_output_dirs()
    print(f"\n[Config] Device      : {config.DEVICE}")
    print(f"[Config] Image size  : {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
    print(f"[Config] Latent dim  : {config.LATENT_DIM}")
    print(f"[Config] Epochs      : {config.EPOCHS}")
    print(f"[Config] Batch size  : {config.BATCH_SIZE}")

    # ── 2. Data ────────────────────────────────────────────────────────────────
    print("\n[Step 1/8] Loading PlantVillage dataset …")
    train_loader, val_loader, class_names = get_dataloaders()
    n_classes = len(class_names)
    print(f"  Classes found: {n_classes}")

    # ── 3. Build models ────────────────────────────────────────────────────────
    print("\n[Step 2/8] Building models …")
    vae = VAE(latent_dim=config.LATENT_DIM)
    ae  = AutoEncoder(latent_dim=config.LATENT_DIM)
    model_summary(vae)

    # ── 4. Train VAE ───────────────────────────────────────────────────────────
    print("\n[Step 3/8] Training VAE …")
    vae_history = train_vae(vae, train_loader, val_loader)

    # ── 5. Train AE ────────────────────────────────────────────────────────────
    print("\n[Step 4/8] Training AutoEncoder (comparison baseline) …")
    ae_history = train_autoencoder(ae, train_loader, val_loader)

    # ── 6. Loss plots ──────────────────────────────────────────────────────────
    print("\n[Step 5/8] Saving loss curves …")
    plot_vae_loss_curves(vae_history)
    plot_ae_loss_curves(ae_history)
    plot_loss_comparison(vae_history, ae_history)

    # ── 7. Reconstruction quality ──────────────────────────────────────────────
    print("\n[Step 6/8] Evaluating reconstruction quality …")

    # Load best VAE checkpoint for evaluation
    best_ckpt_vae = os.path.join(config.CHECKPOINT_DIR, "vae_checkpoint_best.pt")
    if os.path.exists(best_ckpt_vae):
        load_checkpoint(vae, best_ckpt_vae)

    best_ckpt_ae = os.path.join(config.CHECKPOINT_DIR, "ae_checkpoint_best.pt")
    if os.path.exists(best_ckpt_ae):
        load_checkpoint(ae, best_ckpt_ae)

    vae_metrics = evaluate_reconstruction(vae, val_loader, model_type="vae")
    ae_metrics  = evaluate_reconstruction(ae,  val_loader, model_type="ae")
    compare_vae_ae(vae_metrics, ae_metrics)

    # ── 8. Generation & visualisation ─────────────────────────────────────────
    print("\n[Step 7/8] Generating synthetic images and visualisations …")

    # Reconstruction grid
    originals, recons = build_reconstruction_grid(vae, val_loader,
                                                   n=config.NUM_RECON_DISPLAY)
    plot_reconstructions(originals, recons)

    # Random generation
    synth_images = generate_random_samples(vae, n_samples=config.NUM_GEN_SAMPLES,
                                            seed=config.SEED)
    plot_generated_samples(synth_images, nrow=8)

    # Latent interpolations
    interp_frames = interpolate_random_pairs(vae, n_pairs=4,
                                              steps=config.INTERP_STEPS)
    plot_interpolation(interp_frames, steps=config.INTERP_STEPS, n_pairs=4)

    # ── 9. Latent space analysis ───────────────────────────────────────────────
    print("\n[Step 8/8] Analysing latent space …")

    latents, labels = extract_latents(vae, val_loader)

    # PCA
    pca_emb = pca_projection(latents, n_components=2)
    plot_latent_scatter(pca_emb, labels, class_names,
                         method="PCA", filename="latent_pca.png")

    # t-SNE
    tsne_emb = tsne_projection(latents)
    plot_latent_scatter(tsne_emb, labels, class_names,
                         method="t-SNE", filename="latent_tsne.png")

    # Cluster quality
    cluster_metrics = cluster_quality_metrics(latents, labels)

    # Downstream utility
    utility_results = evaluate_downstream_utility(
        vae, train_loader, val_loader, class_names
    )
    plot_utility_comparison(utility_results)

    # ── 10. Final summary ──────────────────────────────────────────────────────
    avg_ratio = np.mean([v["utility_ratio"] for v in utility_results.values()])
    print("\n" + "═" * 65)
    print("  FINAL EVALUATION SUMMARY")
    print("═" * 65)
    print(f"  Dataset            : PlantVillage ({n_classes} classes)")
    print(f"  Image size         : {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
    print(f"  Latent dimension   : {config.LATENT_DIM}")
    print(f"  Training epochs    : {config.EPOCHS}")
    print()
    print("  ── Reconstruction Quality (VAE vs AE) ──────────────────────")
    print(f"  VAE MSE   : {vae_metrics['mse']:.6f}  │  AE MSE : {ae_metrics['mse']:.6f}")
    print(f"  VAE PSNR  : {vae_metrics['psnr']:.2f} dB  │  AE PSNR: {ae_metrics['psnr']:.2f} dB")
    print(f"  VAE SSIM  : {vae_metrics['ssim']:.4f}    │  AE SSIM: {ae_metrics['ssim']:.4f}")
    print()
    print("  ── Latent Space Quality ────────────────────────────────────")
    print(f"  Silhouette Score    : {cluster_metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin Idx  : {cluster_metrics['davies_bouldin']:.4f}")
    print()
    print("  ── Downstream Utility (TSTR / TRTR) ────────────────────────")
    for clf, v in utility_results.items():
        print(f"  {clf:<22} TRTR={v['trtr_acc']:.3f}  TSTR={v['tstr_acc']:.3f}  "
              f"Ratio={v['utility_ratio']:.3f}")
    print(f"  Average utility ratio : {avg_ratio:.3f}")
    print()
    print("  ── Output files ─────────────────────────────────────────────")
    print(f"  Reconstructions      → {config.RECON_DIR}/")
    print(f"  Generated images     → {config.GEN_DIR}/")
    print(f"  Latent visualisations→ {config.LATENT_DIR}/")
    print(f"  Checkpoints          → {config.CHECKPOINT_DIR}/")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
