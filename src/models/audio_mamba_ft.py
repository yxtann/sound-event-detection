"""
Example script for fine-tuning a pre-trained Audio-Mamba model on fewer classes.

This demonstrates how to:
1. Load a pre-trained model (e.g., aum-base_audioset-vggsound)
2. Fine-tune it on a dataset with fewer classes
3. The classification head is automatically re-initialized when num_classes differs
"""

# import sys
import os

# # Add the parent directory to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

import external.audio_mamba.src.models as models

from external.audio_mamba.src import dataloader
from src.config import CLASSES
from src.utils.audio_mamba_metadata_generator import generate_audio_mamba_metadata

# Configuration
CONFIG = {
    # Model settings
    "model_type": "base",  # base, small, or tiny
    "pretrained_path": "external/audio_mamba/examples/inference/models/aum-base_audioset-vggsound.pth",
    # "pretrained_path": "checkpoints/audio_mamba_ft.pth",
    "pretrained_fstride": 16,
    "pretrained_tstride": 16,
    # Your dataset settings
    "num_classes": len(CLASSES),
    "spectrogram_size": (128, 1024),  # (melbins, audio_length)
    "patch_size": (16, 16),
    "strides": (16, 16),
    # Data settings
    "train_json": "data/processed/audio_mamba/train_data.json",
    "val_json": "data/processed/audio_mamba/val_data.json",
    # "val_json": "data/processed/yamnet/extracted_audio/audio_mamba_metadata.json",
    "label_csv": "data/processed/audio_mamba/class_labels_indices.csv",
    "dataset_mean": -5.0767093,  # TODO: Modify per dataset
    "dataset_std": 4.4533687,
    # Training settings
    "batch_size": 64,
    "learning_rate": 1e-5,  # Lower LR for fine-tuning
    "num_epochs": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def create_model(config):
    """
    Create the AudioMamba model with pre-trained weights.

    When num_classes differs from the pre-trained model, the classification
    head is automatically re-initialized while the backbone weights are loaded.
    """
    # Determine embed_dim based on model type
    if "base" in config["model_type"]:
        embed_dim = 768
        depth = 24
    elif "small" in config["model_type"]:
        embed_dim = 384
        depth = 24
    elif "tiny" in config["model_type"]:
        embed_dim = 192
        depth = 24
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    # Create the model
    model = models.AudioMamba(
        spectrogram_size=config["spectrogram_size"],
        patch_size=config["patch_size"],
        strides=config["strides"],
        depth=depth,
        embed_dim=embed_dim,
        num_classes=config["num_classes"],  # This will trigger head re-initialization
        imagenet_pretrain=False,
        aum_pretrain=True,  # Enable loading pre-trained AuM weights
        aum_pretrain_path=config["pretrained_path"],
        aum_pretrain_fstride=config["pretrained_fstride"],
        aum_pretrain_tstride=config["pretrained_tstride"],
        bimamba_type="v1",  # Fo-Bi variant (matches aum-base_audioset-vggsound)
    )

    return model


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    # Training audio config
    train_audio_conf = {
        "num_mel_bins": config["spectrogram_size"][0],
        "target_length": config["spectrogram_size"][1],
        "freqm": 48,  # Frequency masking
        "timem": 192,  # Time masking
        "mixup": 0,
        "dataset": config.get("dataset_name", "custom"),
        "mode": "train",
        "mean": config["dataset_mean"],
        "std": config["dataset_std"],
        "noise": False,
        "fshift": 10,
    }

    # Validation audio config
    val_audio_conf = {
        "num_mel_bins": config["spectrogram_size"][0],
        "target_length": config["spectrogram_size"][1],
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "dataset": config.get("dataset_name", "custom"),
        "mode": "evaluation",
        "mean": config["dataset_mean"],
        "std": config["dataset_std"],
        "noise": False,
        "fshift": 10,
    }

    # Create datasets
    train_dataset = dataloader.AudiosetDataset(
        config["train_json"], label_csv=config["label_csv"], audio_conf=train_audio_conf
    )

    val_dataset = dataloader.AudiosetDataset(
        config["val_json"], label_csv=config["label_csv"], audio_conf=val_audio_conf
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, config):
    """Simple training loop (for demonstration)."""
    model = model.to(config["device"])
    model.train()

    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=5e-7
    )

    logger.info("Starting training...")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Classification head shape: {model.head.weight.shape}")
    logger.info(f"Number of classes: {config['num_classes']}")

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, (audio_input, labels, paths) in pbar:
            audio_input = audio_input.to(config["device"])
            labels = labels.to(config["device"])

            # Forward pass
            optimizer.zero_grad()
            output = model(audio_input)
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_description(f"Epoch {epoch}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

        # Validation (simplified)
        validate_model(model, val_loader, config)

    return model


def validate_model(model, val_loader, config):
    """Simple validation loop."""
    model.eval()
    total_correct = 0
    total_samples = 0
    preds_gt_list = []

    with torch.no_grad():
        for audio_input, labels, paths in val_loader:
            audio_input = audio_input.to(config["device"])
            labels = labels.to(config["device"])

            output = model(audio_input)
            predictions = torch.sigmoid(output) > 0.5  # For multi-label

            # Simple accuracy calculation (adjust for your metric)
            total_correct += (predictions == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

            for pred, gt in zip(predictions, labels):
                # Convert to the actual class label
                pred_label = str(CLASSES[pred.int().argmax()])
                gt_label = str(CLASSES[gt.argmax()])
                if pred_label == gt_label:
                    preds_gt_list.append((pred_label, gt_label, True))
                else:
                    preds_gt_list.append((pred_label, gt_label, False))

    accuracy = total_correct / total_samples
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    # model.train()


def audio_mamba_inference(checkpoint_path, val_json_path):

    # Replace the CONFIG vals
    CONFIG["pretrained_path"] = checkpoint_path
    CONFIG["val_json"] = val_json_path

    logger.info(f"Checkpoint found at {checkpoint_path}, skipping training")
    model = create_model(CONFIG)
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    model.load_state_dict(checkpoint)
    model = model.to(CONFIG["device"])
    _, val_loader = create_dataloaders(CONFIG)
    validate_model(model, val_loader, CONFIG)


if __name__ == "__main__":
    # Only train if the checkpoint doesn't exist
    checkpoint_path = "checkpoints/audio_mamba_ft.pth"
    retrain_model = False

    if not os.path.exists(checkpoint_path) or retrain_model:
        # Regenerate the Audio Mamba metadata
        generate_audio_mamba_metadata()

        # Start the actual fine-tuning
        logger.info("=" * 60)
        logger.info(f"Fine-tuning Audio-Mamba on {CONFIG['num_classes']} Classes")
        logger.info("=" * 60)

        # Create model
        logger.info("1. Creating model with pre-trained weights...")
        model = create_model(CONFIG)

        # Create dataloaders
        logger.info("2. Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(CONFIG)
        logger.info(f"   Training samples: {len(train_loader.dataset)}")
        logger.info(f"   Validation samples: {len(val_loader.dataset)}")

        # Train
        logger.info("3. Starting training...")
        trained_model = train_model(model, train_loader, val_loader, CONFIG)

        # Save model
        logger.info("4. Saving model...")
        save_path = "checkpoints/audio_mamba_ft.pth"
        torch.save(trained_model.state_dict(), save_path)
        logger.info(f"   Model saved to {save_path}")

        logger.info("" + "=" * 60)
        logger.info("Fine-tuning completed!")
        logger.info("=" * 60)

    else:
        # Just running validation
        logger.info(f"Checkpoint found at {checkpoint_path}, skipping training")
        model = create_model(CONFIG)
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
        model.load_state_dict(checkpoint)
        model = model.to(CONFIG["device"])
        _, val_loader = create_dataloaders(CONFIG)
        validate_model(model, val_loader, CONFIG)
