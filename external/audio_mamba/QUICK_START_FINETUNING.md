# Quick Start: Fine-tuning on Fewer Classes

## TL;DR

To fine-tune `aum-base_audioset-vggsound` on fewer classes:

1. **Set these key parameters in your training script:**
   ```bash
   aum_pretrain=True
   aum_pretrain_path=/path/to/aum-base_audioset-vggsound.pth
   n_class=YOUR_NUMBER_OF_CLASSES  # e.g., 10, 20, etc.
   ```

2. **The code automatically:**
   - ✅ Loads all backbone weights (patches, embeddings, Mamba layers)
   - ✅ Skips the classification head (since num_classes differs)
   - ✅ Initializes a new head with your number of classes

3. **That's it!** The model is ready to fine-tune.

## Example Command

```bash
python ../../src/run.py \
  --model aum \
  --model_type base \
  --dataset your_dataset \
  --data-train /path/to/train.json \
  --data-val /path/to/val.json \
  --label-csv /path/to/labels.csv \
  --n_class 10 \
  --aum_pretrain True \
  --aum_pretrain_path /path/to/aum-base_audioset-vggsound.pth \
  --aum_pretrain_fstride 16 \
  --aum_pretrain_tstride 16 \
  --lr 1e-5 \
  --n-epochs 20 \
  --batch-size 12 \
  --run_type train
```

## What Happens Under the Hood

When you load a pre-trained model with different `num_classes`, the code in `src/models/mamba_models.py` (lines 445-451) detects this and:

```python
# check if num_classes is the same
if weights['head.weight'].shape[0] != num_classes:
    print('Num classes differ! Can only load the backbone weights.')
    del weights['head.weight']  # Skip loading head weights
    del weights['head.bias']     # Skip loading head bias
```

The classification head (`self.head`) is already initialized with your `num_classes` at line 286, so it's ready to train!

## Important Notes

- **Learning Rate**: Use a lower LR for fine-tuning (e.g., `1e-5` to `1e-4`)
- **Architecture**: Keep `fstride=16`, `tstride=16`, `patch_size=(16,16)` to match pre-trained model
- **Expected Message**: You'll see "Num classes differ! Can only load the backbone weights." - this is **normal and expected**

## Full Examples

- **Bash script**: `exps/custom/finetune_example.sh`
- **Python script**: `examples/finetuning/finetune_example.py`
- **Detailed guide**: `FINETUNING_GUIDE.md`

