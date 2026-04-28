"""
config.py
=========
Central configuration file for the VAE PlantVillage project.
Updated to use the real PlantVillage1 dataset with pre-existing train/val split.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))

# Real PlantVillage dataset — already split into train/ and val/ sub-folders
DATA_DIR       = os.path.join(BASE_DIR, "data", "PlantVillage1")
TRAIN_DIR      = os.path.join(DATA_DIR, "train")
VAL_DIR        = os.path.join(DATA_DIR, "val")

# Set to True when the dataset folder already contains train/ and val/ sub-dirs.
# Set to False to fall back to the old random-split logic (for synthetic data).
DATASET_PRESPLIT = True

OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RECON_DIR      = os.path.join(OUTPUT_DIR, "reconstructions")
GEN_DIR        = os.path.join(OUTPUT_DIR, "generated")
LATENT_DIR     = os.path.join(OUTPUT_DIR, "latent")
REPORT_DIR     = os.path.join(BASE_DIR, "report")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE       = 128     # Resize to 128×128
BATCH_SIZE       = 32      # 32 keeps steps/epoch manageable on CPU
NUM_WORKERS      = 0       # 0 avoids Windows multiprocessing issues
# 10 classes → ~11,400 train images → ~357 batches/epoch → ~30 min/epoch on CPU
# Change to None for all 38 classes (needs GPU / overnight run)
MAX_CLASSES      = 10
USE_AUGMENTATION = True

# ─────────────────────────────────────────────────────────────────────────────
# VAE ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
LATENT_DIM               = 256   # Increased from 128 — 38 real classes need
                                  # more representational capacity
ENCODER_LAST_CHANNELS    = 256
ENCODER_FEATURE_MAP_SIZE = 8
FLAT_DIM = ENCODER_LAST_CHANNELS * ENCODER_FEATURE_MAP_SIZE * ENCODER_FEATURE_MAP_SIZE
# FLAT_DIM = 256 × 8 × 8 = 16,384

BETA = 0.5   # Reduced KL weight: with real complex images a weaker KL
              # constraint preserves more visual detail while still
              # regularising the latent space

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS        = 20           # More epochs — real images need longer training
LEARNING_RATE = 5e-4         # Slightly lower LR for stable convergence on
                              # the complex real-image distribution
WEIGHT_DECAY  = 1e-4         # Stronger L2 regularisation to prevent overfitting
                              # on the relatively small dataset (~1,200 images)
LOSS_TYPE     = "mse"
SAVE_EVERY    = 10

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────
NUM_GEN_SAMPLES = 64
INTERP_STEPS    = 10

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
NUM_RECON_DISPLAY  = 16
MAX_LATENT_SAMPLES = 2000
TSNE_PERPLEXITY    = 15      # Lower perplexity for small val set (~295 samples)
PCA_N_COMPONENTS   = 50
