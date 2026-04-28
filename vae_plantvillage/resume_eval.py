"""
resume_eval.py
==============
Runs only the evaluation / visualisation steps using saved checkpoints.
Use this after training has already completed to skip re-training.
"""
import sys
import os
# Force UTF-8 output on Windows so Unicode characters don't crash the terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import config
from data.loader               import get_dataloaders
from models.vae                import VAE
from models.autoencoder        import AutoEncoder
from training.train            import load_checkpoint
from evaluation.reconstruction import evaluate_reconstruction, compare_vae_ae
from evaluation.latent_space   import extract_latents, pca_projection, \
                                       tsne_projection, cluster_quality_metrics
from evaluation.classifier     import evaluate_downstream_utility
from generation.generate       import generate_random_samples, \
                                       interpolate_random_pairs, \
                                       build_reconstruction_grid
from visualization.plots       import (
    plot_reconstructions, plot_generated_samples,
    plot_latent_scatter, plot_interpolation, plot_utility_comparison,
)

def main():
    print("\nLoading data ...")
    train_loader, val_loader, class_names = get_dataloaders()

    # ── Load VAE ──────────────────────────────────────────────────────────────
    vae = VAE().to(config.DEVICE)
    load_checkpoint(vae, os.path.join(config.CHECKPOINT_DIR,
                                      "vae_checkpoint_best.pt"))

    # ── Load AE ───────────────────────────────────────────────────────────────
    ae = AutoEncoder().to(config.DEVICE)
    load_checkpoint(ae, os.path.join(config.CHECKPOINT_DIR,
                                     "ae_checkpoint_best.pt"))

    # ── Reconstruction quality ─────────────────────────────────────────────────
    vae_metrics = evaluate_reconstruction(vae, val_loader, model_type="vae")
    ae_metrics  = evaluate_reconstruction(ae,  val_loader, model_type="ae")
    compare_vae_ae(vae_metrics, ae_metrics)

    # ── Reconstruction grid ────────────────────────────────────────────────────
    originals, recons = build_reconstruction_grid(vae, val_loader,
                                                   n=config.NUM_RECON_DISPLAY)
    plot_reconstructions(originals, recons)

    # ── Generated samples ──────────────────────────────────────────────────────
    synth = generate_random_samples(vae, n_samples=config.NUM_GEN_SAMPLES,
                                     seed=config.SEED)
    plot_generated_samples(synth, nrow=8)

    # ── Interpolation ──────────────────────────────────────────────────────────
    frames = interpolate_random_pairs(vae, n_pairs=4, steps=config.INTERP_STEPS)
    plot_interpolation(frames, steps=config.INTERP_STEPS, n_pairs=4)

    # ── Latent space ───────────────────────────────────────────────────────────
    latents, labels = extract_latents(vae, val_loader)

    pca_emb = pca_projection(latents, n_components=2)
    plot_latent_scatter(pca_emb, labels, class_names,
                         method="PCA", filename="latent_pca.png")

    tsne_emb = tsne_projection(latents)
    plot_latent_scatter(tsne_emb, labels, class_names,
                         method="t-SNE", filename="latent_tsne.png")

    cluster_metrics = cluster_quality_metrics(latents, labels)

    # ── Downstream utility ─────────────────────────────────────────────────────
    utility_results = evaluate_downstream_utility(
        vae, train_loader, val_loader, class_names
    )
    plot_utility_comparison(utility_results)

    # ── Summary ────────────────────────────────────────────────────────────────
    avg_ratio = np.mean([v["utility_ratio"] for v in utility_results.values()])
    print("\n" + "="*60)
    print("  FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"  VAE MSE  : {vae_metrics['mse']:.6f}  |  AE MSE : {ae_metrics['mse']:.6f}")
    print(f"  VAE PSNR : {vae_metrics['psnr']:.2f} dB  |  AE PSNR: {ae_metrics['psnr']:.2f} dB")
    print(f"  VAE SSIM : {vae_metrics['ssim']:.4f}    |  AE SSIM: {ae_metrics['ssim']:.4f}")
    print(f"  Silhouette  : {cluster_metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {cluster_metrics['davies_bouldin']:.4f}")
    for clf, v in utility_results.items():
        print(f"  {clf:<22} TRTR={v['trtr_acc']:.3f}  TSTR={v['tstr_acc']:.3f}  "
              f"Ratio={v['utility_ratio']:.3f}")
    print(f"  Average utility ratio: {avg_ratio:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
