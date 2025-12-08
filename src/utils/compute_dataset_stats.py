"""
Script to compute dataset mean and std for Audio Mamba normalization.
Run this script to get the correct normalization values for your dataset.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add paths for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../external/audio_mamba")
)
from src import dataloader

# Configuration - match your training config
TRAIN_JSON = "data/processed/audio_mamba/train_data_noisy.json"
LABEL_CSV = "data/processed/audio_mamba/class_labels_indices.csv"

# Audio config - MUST match your training config exactly
# Set skip_norm=True to compute stats without normalization
audio_conf = {
    "num_mel_bins": 128,  # Match your spectrogram_size[0]
    "target_length": 1024,  # Match your spectrogram_size[1]
    "freqm": 0,  # Disable augmentation for accurate stats
    "timem": 0,  # Disable augmentation for accurate stats
    "mixup": 0,  # Disable mixup for accurate stats
    "skip_norm": True,  # CRITICAL: Must be True to compute stats
    "mode": "train",
    "dataset": "custom",
    "fshift": 10,
}

print("=" * 60)
print("Computing dataset normalization statistics...")
print("=" * 60)
print(f"Training JSON: {TRAIN_JSON}")
print(f"Label CSV: {LABEL_CSV}")
print(f"Config: {audio_conf}")
print("=" * 60)

# Create dataset and dataloader
train_dataset = dataloader.AudiosetDataset(
    TRAIN_JSON, label_csv=LABEL_CSV, audio_conf=audio_conf
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1000,  # Large batch size for efficiency
    shuffle=False,  # Don't shuffle for reproducibility
    num_workers=4,
    pin_memory=True,
)

# Compute statistics
means = []
stds = []
total_samples = 0

print("\nProcessing batches...")
for i, (audio_input, labels, wav_paths) in enumerate(train_loader):
    # Compute mean and std for this batch
    cur_mean = torch.mean(audio_input).item()
    cur_std = torch.std(audio_input).item()

    means.append(cur_mean)
    stds.append(cur_std)
    total_samples += audio_input.shape[0]

    print(f"Processed {i + 1} batches ({total_samples} samples)...")

# Compute final statistics
dataset_mean = np.mean(means)
dataset_std = np.mean(stds)

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Dataset Mean: {dataset_mean:.7f}")
print(f"Dataset Std:  {dataset_std:.7f}")
print("=" * 60)
print("\nUpdate your config with these values:")
print(f'    "dataset_mean": {dataset_mean:.7f},')
print(f'    "dataset_std": {dataset_std:.7f},')
print("=" * 60)
