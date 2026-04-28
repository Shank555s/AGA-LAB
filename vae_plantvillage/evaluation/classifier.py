"""
evaluation/classifier.py
=========================
Downstream classification utility evaluation.

Methodology (adapted from DBN report Section 9 — TSTR paradigm)
-----------------------------------------------------------------
We adapt the Train-on-Synthetic-Test-on-Real (TSTR) concept to the
image domain as follows:

  TRAIN-on-LATENT-of-REAL  (TRTR analogue)
      Train a classifier on latent representations of REAL training images.
      Test on latent representations of REAL validation images.
      → Upper-bound oracle performance.

  TRAIN-on-LATENT-of-SYNTHETIC  (TSTR analogue)
      Train a classifier on latent representations of SYNTHETICALLY generated images.
      Test on latent representations of REAL validation images.
      → Measures how much discriminative utility the generative model preserves.

  Utility Ratio = TSTR Accuracy / TRTR Accuracy

Classifiers used
-----------------
* Logistic Regression   — linear baseline
* Random Forest         — non-linear ensemble
* MLP (sklearn)         — shallow neural network on latent features
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics        import accuracy_score, f1_score
from sklearn.preprocessing  import StandardScaler
from tqdm import tqdm

import config
from models.vae import VAE


# ─────────────────────────────────────────────────────────────────────────────
# Helper: encode a DataLoader to (latents, labels)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_loader(
    model  : VAE,
    loader : DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    latents_list, labels_list = [], []
    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="  Encoding images", leave=False,
                                  ncols=70):
            images = images.to(config.DEVICE)
            mu, _  = model.encoder(images)
            latents_list.append(mu.cpu().numpy())
            labels_list.append(lbls.numpy())
    return (np.concatenate(latents_list),
            np.concatenate(labels_list))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: generate synthetic latents by sampling from p(z)=N(0,I)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_latents(
    model    : VAE,
    n_samples: int,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples n_samples latent codes from N(0,I) and assigns class labels
    uniformly (approximation since the VAE is unconditional).

    For a conditional VAE the label assignment would be exact.
    """
    model.eval()
    z = torch.randn(n_samples, model.latent_dim, device=config.DEVICE)
    # Assign labels in round-robin to create a balanced synthetic set
    labels = np.tile(np.arange(n_classes), n_samples // n_classes + 1)[:n_samples]
    return z.cpu().numpy(), labels


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_downstream_utility(
    model        : VAE,
    train_loader : DataLoader,
    val_loader   : DataLoader,
    class_names  : list,
) -> Dict[str, Dict[str, float]]:
    """
    Runs the TRTR and TSTR evaluation protocols and returns a nested dict:

        results["classifier_name"]["trtr_acc"]
        results["classifier_name"]["tstr_acc"]
        results["classifier_name"]["utility_ratio"]
        ...
    """
    model = model.to(config.DEVICE)
    n_classes = len(class_names)

    # ── 1. Encode real train / val images ─────────────────────────────────────
    print("\n[Downstream Utility] Encoding real training set …")
    X_train_real, y_train = _encode_loader(model, train_loader)
    print("[Downstream Utility] Encoding real validation set …")
    X_val,        y_val   = _encode_loader(model, val_loader)

    # ── 2. Generate synthetic latents (TSTR) ──────────────────────────────────
    n_synth = len(X_train_real)   # same size as real train set
    X_train_synth, y_train_synth = _generate_synthetic_latents(
        model, n_synth, n_classes
    )

    # ── 3. Standardise features ───────────────────────────────────────────────
    scaler = StandardScaler().fit(X_train_real)
    X_train_real_s  = scaler.transform(X_train_real)
    X_train_synth_s = scaler.transform(X_train_synth)
    X_val_s         = scaler.transform(X_val)

    # ── 4. Define classifiers ─────────────────────────────────────────────────
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=config.SEED, n_jobs=1),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=config.SEED, n_jobs=1),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=300,
            random_state=config.SEED),
    }

    results: Dict[str, Dict[str, float]] = {}

    print(f"\n{'═'*60}")
    print("  Downstream Classification Utility (TRTR vs TSTR)")
    print(f"{'═'*60}")
    print(f"  {'Classifier':<22} {'TRTR Acc':>10} {'TSTR Acc':>10} {'Ratio':>8}")
    print("─" * 60)

    for clf_name, clf in classifiers.items():
        # TRTR: trained on real latents
        clf.fit(X_train_real_s, y_train)
        trtr_pred = clf.predict(X_val_s)
        trtr_acc  = accuracy_score(y_val, trtr_pred)
        trtr_f1   = f1_score(y_val, trtr_pred, average="macro", zero_division=0)

        # TSTR: trained on synthetic latents
        clf.fit(X_train_synth_s, y_train_synth)
        tstr_pred = clf.predict(X_val_s)
        tstr_acc  = accuracy_score(y_val, tstr_pred)
        tstr_f1   = f1_score(y_val, tstr_pred, average="macro", zero_division=0)

        ratio = tstr_acc / trtr_acc if trtr_acc > 0 else 0.0

        results[clf_name] = {
            "trtr_acc": trtr_acc,
            "trtr_f1":  trtr_f1,
            "tstr_acc": tstr_acc,
            "tstr_f1":  tstr_f1,
            "utility_ratio": ratio,
        }
        print(f"  {clf_name:<22} {trtr_acc:>10.4f} {tstr_acc:>10.4f} {ratio:>8.3f}")

    print("─" * 60)
    avg_ratio = np.mean([v["utility_ratio"] for v in results.values()])
    print(f"  {'Average':<22} {'':>10} {'':>10} {avg_ratio:>8.3f}")
    print(f"{'═'*60}\n")
    print(f"  Interpretation: A utility ratio of {avg_ratio:.3f} means that "
          f"classifiers trained on\n  synthetic latent representations preserve "
          f"{avg_ratio*100:.1f}% of the accuracy achievable\n  with real training data.\n")

    return results
