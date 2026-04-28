"""
evaluation/latent_space.py
===========================
Analyses the structure and quality of the learned latent space.

Methods
-------
1. Latent extraction — encode the full validation set to collect
   (z_vectors, labels) pairs.

2. PCA projection  — reduce to 2D via principal component analysis
   for a quick linear view of the latent structure.

3. t-SNE projection — non-linear 2D embedding; PCA pre-reduction
   (to config.PCA_N_COMPONENTS) is applied first for speed/quality.

4. Cluster quality metrics
   * Silhouette Score   (higher → better separation, range [−1, +1])
   * Davies-Bouldin Index (lower → tighter, better-separated clusters)
   These mirror the quantitative evaluation approach from the reference
   DBN project's Section 8 (Statistical Fidelity Evaluation).
"""

from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm

import config
from models.vae import VAE


# ─────────────────────────────────────────────────────────────────────────────
# Latent extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_latents(
    model     : VAE,
    loader    : DataLoader,
    max_samples: int = config.MAX_LATENT_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes images from `loader` and returns:
        latents : (N, LATENT_DIM) numpy array of posterior means μ
        labels  : (N,)             numpy array of integer class labels

    Uses the deterministic posterior mean μ (no sampling) so that
    repeated calls return the same representation for the same inputs.
    """
    model = model.to(config.DEVICE)
    model.eval()

    all_latents = []
    all_labels  = []
    collected   = 0

    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Extracting latent representations",
                                  leave=True, ncols=80):
            if collected >= max_samples:
                break
            images = images.to(config.DEVICE)
            mu, _  = model.encoder(images)          # use μ, not sampled z
            all_latents.append(mu.cpu().numpy())
            all_labels.append(lbls.numpy())
            collected += images.size(0)

    latents = np.concatenate(all_latents, axis=0)[:max_samples]
    labels  = np.concatenate(all_labels,  axis=0)[:max_samples]
    return latents, labels


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────

def pca_projection(latents: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Returns 2D PCA projection of the latent vectors.
    Fast and linear — good for a first look at global structure.
    """
    pca = PCA(n_components=n_components, random_state=config.SEED)
    return pca.fit_transform(latents)


def tsne_projection(
    latents    : np.ndarray,
    perplexity : int = config.TSNE_PERPLEXITY,
    pca_first  : int = config.PCA_N_COMPONENTS,
) -> np.ndarray:
    """
    Returns 2D t-SNE embedding.

    PCA pre-reduction (to min(pca_first, latent_dim) components) is applied
    before t-SNE to denoise and speed up the embedding.
    """
    # PCA pre-reduction
    n_pca = min(pca_first, latents.shape[1], latents.shape[0] - 1)
    if n_pca > 2:
        latents = PCA(n_components=n_pca,
                      random_state=config.SEED).fit_transform(latents)

    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                learning_rate="auto",
                init="pca",
                random_state=config.SEED,
                max_iter=1000)
    print(f"  Running t-SNE on {latents.shape[0]} samples "
          f"(perplexity={perplexity}) …")
    return tsne.fit_transform(latents)


# ─────────────────────────────────────────────────────────────────────────────
# Cluster quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def cluster_quality_metrics(
    latents : np.ndarray,
    labels  : np.ndarray,
) -> Dict[str, float]:
    """
    Computes cluster quality metrics in the latent space.

    Returns:
        silhouette       : float in [−1, 1],  higher is better
        davies_bouldin   : float > 0,          lower is better
    """
    # Subsample if needed for silhouette (O(n²) complexity)
    if len(latents) > 3000:
        rng  = np.random.RandomState(config.SEED)
        idx  = rng.choice(len(latents), 3000, replace=False)
        sub_latents = latents[idx]
        sub_labels  = labels[idx]
    else:
        sub_latents, sub_labels = latents, labels

    # Silhouette and DB require at least 2 distinct classes
    n_unique = len(np.unique(sub_labels))
    if n_unique < 2:
        print("  [WARN] Only one class found; skipping cluster metrics.")
        return {"silhouette": float("nan"), "davies_bouldin": float("nan")}

    sil = silhouette_score(sub_latents, sub_labels, metric="euclidean",
                           sample_size=min(2000, len(sub_latents)),
                           random_state=config.SEED)
    dbi = davies_bouldin_score(sub_latents, sub_labels)

    results = {"silhouette": sil, "davies_bouldin": dbi}
    print(f"\n{'─'*50}")
    print(f"  Latent Space Cluster Quality")
    print(f"  Silhouette Score    : {sil:.4f}  (range [−1,1];  higher=better)")
    print(f"  Davies-Bouldin Index: {dbi:.4f}  (>0;            lower=better)")
    print(f"{'─'*50}\n")
    return results
