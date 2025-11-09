# Fine-tuning Audio-Mamba on Fewer Classes

This guide explains how to use a pre-trained Audio-Mamba model (e.g., `aum-base_audioset-vggsound`) and fine-tune it on a dataset with fewer classes.

## Overview

When you load a pre-trained model with a different number of classes, the code automatically:
1. Loads all backbone weights (patch embedding, positional embedding, Mamba layers, etc.)
2. **Skips** the classification head weights (since the number of classes differs)
3. Initializes a **new** classification head with the correct number of classes for your task

This is handled automatically in `src/models/mamba_models.py` (lines 445-451).

## Steps to Fine-tune

### 1. Prepare Your Data

You need:
- **Training data JSON**: A JSON file listing your training samples with their labels
- **Validation data JSON**: A JSON file listing your validation samples
- **Label CSV**: A CSV file mapping class indices to class names (similar to `class_labels_indices.csv`)

Example JSON format (from VGGSound):
```json
{
    "data": [
        {
            "wav": "/path/to/audio1.wav",
            "labels": "/m/class1"
        },
        {
            "wav": "/path/to/audio2.wav",
            "labels": "/m/class2"
        }
    ]
}
```

Example CSV format:
```csv
index,mid,display_name
0,/m/class1,Class 1 Name
1,/m/class2,Class 2 Name
...
```

### 2. Download the Pre-trained Model

Download the pre-trained model checkpoint. For `aum-base_audioset-vggsound`:
- Link: https://drive.google.com/file/d/1spsJXncpEXHKmIvDcB7ddkcgrzARpEeK/view?usp=drive_link
- Save it to a location accessible by your training script

### 3. Create a Training Script

Create a bash script similar to `aum-base_audioset-vggsound.sh` but with your custom settings:

**Key parameters to modify:**
- `aum_pretrain=True` - Enable loading pre-trained weights
- `aum_pretrain_path` - Path to your downloaded checkpoint
- `n_class` - **Set this to your number of classes** (must be less than the original)
- `data-train` - Path to your training JSON
- `data-val` - Path to your validation JSON
- `label-csv` - Path to your label CSV file
- `dataset_mean` and `dataset_std` - Normalization stats for your dataset (you may need to compute these)

### 4. Example Training Script

See `exps/custom/finetune_example.sh` for a complete example.

### 5. Run Training

```bash
cd exps/custom
bash finetune_example.sh
```

## Important Notes

1. **Classification Head**: The classification head is automatically re-initialized when `num_classes` differs. You don't need to manually modify it.

2. **Learning Rate**: For fine-tuning, use a lower learning rate (e.g., `1e-5` to `1e-4`) compared to training from scratch.

3. **Dataset Statistics**: Make sure to use the correct `dataset_mean` and `dataset_std` for your dataset. You can compute these using `src/get_norm_stats.py`.

4. **Model Architecture**: The backbone architecture (patch size, strides, embed_dim, etc.) should match the pre-trained model. The example script uses the same architecture as the base model.

5. **Loss Function**: 
   - Use `BCE` (Binary Cross Entropy) for multi-label classification
   - Use `CE` (Cross Entropy) for single-label classification

## Troubleshooting

- **"Num classes differ! Can only load the backbone weights."**: This is expected and normal. The backbone loads successfully, and a new head is initialized.

- **Mismatched patch sizes/strides**: Make sure `fstride`, `tstride`, `fpatch_size`, `tpatch_size` match the pre-trained model (typically 16x16 patches with 16x16 strides for base models).

- **Positional embedding issues**: The code handles interpolation automatically, but ensure your `audio_length` and `melbins` are compatible.

