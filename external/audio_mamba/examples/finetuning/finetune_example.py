"""
Example script for fine-tuning a pre-trained Audio-Mamba model on fewer classes.

This demonstrates how to:
1. Load a pre-trained model (e.g., aum-base_audioset-vggsound)
2. Fine-tune it on a dataset with fewer classes
3. The classification head is automatically re-initialized when num_classes differs
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import src.models as models
from src import dataloader

# Configuration
CONFIG = {
    # Model settings
    'model_type': 'base',  # base, small, or tiny
    'pretrained_path': 'external/audio_mamba/examples/inference/models/aum-base_audioset-vggsound.pth',  # Path to your downloaded checkpoint
    'pretrained_fstride': 16,
    'pretrained_tstride': 16,
    
    # Your dataset settings
    'num_classes': 4,  # Your number of classes (fewer than original)
    'spectrogram_size': (128, 1024),  # (melbins, audio_length)
    'patch_size': (16, 16),
    'strides': (16, 16),
    
    # Data settings
    'train_json': 'external/audio_mamba/examples/finetuning/datafiles/custom_train.json',
    'val_json': 'external/audio_mamba/examples/finetuning/datafiles/custom_eval.json',
    'label_csv': 'external/audio_mamba/examples/finetuning/datafiles/class_labels_indices.csv',
    'dataset_mean': -5.0767093,  # Compute for your dataset
    'dataset_std': 4.4533687,
    
    # Training settings
    'batch_size': 12,
    'learning_rate': 1e-5,  # Lower LR for fine-tuning
    'num_epochs': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def create_model(config):
    """
    Create the AudioMamba model with pre-trained weights.
    
    When num_classes differs from the pre-trained model, the classification
    head is automatically re-initialized while the backbone weights are loaded.
    """
    # Determine embed_dim based on model type
    if 'base' in config['model_type']:
        embed_dim = 768
        depth = 24
    elif 'small' in config['model_type']:
        embed_dim = 384
        depth = 24
    elif 'tiny' in config['model_type']:
        embed_dim = 192
        depth = 24
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Create the model
    model = models.AudioMamba(
        spectrogram_size=config['spectrogram_size'],
        patch_size=config['patch_size'],
        strides=config['strides'],
        depth=depth,
        embed_dim=embed_dim,
        num_classes=config['num_classes'],  # This will trigger head re-initialization
        imagenet_pretrain=False,
        aum_pretrain=True,  # Enable loading pre-trained AuM weights
        aum_pretrain_path=config['pretrained_path'],
        aum_pretrain_fstride=config['pretrained_fstride'],
        aum_pretrain_tstride=config['pretrained_tstride'],
        bimamba_type='v1',  # Fo-Bi variant (matches aum-base_audioset-vggsound)
    )
    
    return model


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    # Training audio config
    train_audio_conf = {
        'num_mel_bins': config['spectrogram_size'][0],
        'target_length': config['spectrogram_size'][1],
        'freqm': 48,  # Frequency masking
        'timem': 192,  # Time masking
        'mixup': 0,
        'dataset': config.get('dataset_name', 'custom'),
        'mode': 'train',
        'mean': config['dataset_mean'],
        'std': config['dataset_std'],
        'noise': False,
        'fshift': 10,
    }
    
    # Validation audio config
    val_audio_conf = {
        'num_mel_bins': config['spectrogram_size'][0],
        'target_length': config['spectrogram_size'][1],
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': config.get('dataset_name', 'custom'),
        'mode': 'evaluation',
        'mean': config['dataset_mean'],
        'std': config['dataset_std'],
        'noise': False,
        'fshift': 10,
    }
    
    # Create datasets
    train_dataset = dataloader.AudiosetDataset(
        config['train_json'],
        label_csv=config['label_csv'],
        audio_conf=train_audio_conf
    )
    
    val_dataset = dataloader.AudiosetDataset(
        config['val_json'],
        label_csv=config['label_csv'],
        audio_conf=val_audio_conf
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, config):
    """Simple training loop (for demonstration)."""
    model = model.to(config['device'])
    model.train()
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    # Use nn.CrossEntropyLoss() for single-label classification
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=5e-7
    )
    
    print("Starting training...")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Classification head shape: {model.head.weight.shape}")
    print(f"Number of classes: {config['num_classes']}")
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (audio_input, labels, paths) in enumerate(train_loader):
            audio_input = audio_input.to(config['device'])
            labels = labels.to(config['device'])
            
            # Forward pass
            optimizer.zero_grad()
            output = model(audio_input)
            loss = criterion(output, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Validation (simplified)
        if (epoch + 1) % 5 == 0:
            validate_model(model, val_loader, config)
    
    return model


def validate_model(model, val_loader, config):
    """Simple validation loop."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for audio_input, labels, paths in val_loader:
            audio_input = audio_input.to(config['device'])
            labels = labels.to(config['device'])
            
            output = model(audio_input)
            predictions = torch.sigmoid(output) > 0.5  # For multi-label
            
            # Simple accuracy calculation (adjust for your metric)
            total_correct += (predictions == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f'Validation Accuracy: {accuracy:.4f}')
    model.train()


if __name__ == '__main__':
    print("=" * 60)
    print("Fine-tuning Audio-Mamba on Fewer Classes")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating model with pre-trained weights...")
    model = create_model(CONFIG)
    print("   Model created successfully!")
    print(f"   Note: If you see 'Num classes differ! Can only load the backbone weights.'")
    print(f"   This is expected - the backbone loads and a new head is initialized.")
    
    # Create dataloaders
    print("\n2. Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(CONFIG)
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    
    # Train
    print("\n3. Starting training...")
    trained_model = train_model(model, train_loader, val_loader, CONFIG)
    
    # Save model
    print("\n4. Saving model...")
    save_path = 'finetuned_model.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"   Model saved to {save_path}")
    
    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print("=" * 60)

