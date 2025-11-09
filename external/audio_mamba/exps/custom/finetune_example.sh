#!/bin/bash

# Example script for fine-tuning aum-base_audioset-vggsound on fewer classes
# Modify the paths and parameters according to your setup

model=aum
model_type=base
dataset=your_dataset_name  # Change this to your dataset name

# Pre-trained model settings
aum_pretrain=True
aum_pretrain_path=/path/to/aum-base_audioset-vggsound.pth  # Path to downloaded checkpoint
aum_pretrain_fstride=16
aum_pretrain_tstride=16

# Training settings
bal=full  # or 'bal' for balanced sampling
lr=1e-5  # Lower learning rate for fine-tuning
epoch=20
tr_data=./data/datafiles/your_train.json  # Path to your training data JSON
lrscheduler_start=5
lrscheduler_step=2
lrscheduler_decay=0.75

te_data=./data/datafiles/your_val.json  # Path to your validation data JSON
freqm=48
timem=192
mixup=0

# Model architecture (should match pre-trained model)
fstride=16
tstride=16
batch_size=12

# Data settings
label_csv=./data/class_labels_indices.csv  # Your label CSV file
dataset_mean=-5.0767093  # Compute these for your dataset if different
dataset_std=4.4533687
audio_length=1024
noise=False

# Classification settings
metrics=acc  # or 'mAP' for multi-label
loss=BCE  # or 'CE' for single-label
warmup=True

# IMPORTANT: Set this to your number of classes (fewer than original)
n_class=10  # Change this to your actual number of classes

exp_root=/path/to/your/experiments  # Modify according to yours
exp_name=aum-base_audioset-finetune-custom

exp_dir=$exp_root/$exp_name

if [ -d $exp_dir ]; then
  echo "The experiment directory exists. Should I remove it? [y/k/n]"
  read answer
  if [ $answer == "y" ]; then
    rm -r $exp_dir
  elif [ $answer == "k" ]; then
    echo "Keeping the directory"
  else
    echo "Please remove the directory or change the name in the script and run again"
    exit 1
  fi
fi

mkdir -p $exp_dir

# Run training with accelerate
CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 accelerate launch --mixed_precision=fp16 ../../src/run.py \
  --model ${model} \
  --dataset ${dataset} \
  --data-train ${tr_data} \
  --data-val ${te_data} \
  --exp-dir $exp_dir \
  --label-csv ${label_csv} \
  --n_class ${n_class} \
  --lr $lr \
  --n-epochs ${epoch} \
  --batch-size $batch_size \
  --save_model True \
  --freqm $freqm \
  --timem $timem \
  --mixup ${mixup} \
  --bal ${bal} \
  --tstride $tstride \
  --fstride $fstride \
  --aum_pretrain $aum_pretrain \
  --aum_pretrain_path $aum_pretrain_path \
  --aum_pretrain_fstride $aum_pretrain_fstride \
  --aum_pretrain_tstride $aum_pretrain_tstride \
  --dataset_mean ${dataset_mean} \
  --dataset_std ${dataset_std} \
  --audio_length ${audio_length} \
  --noise ${noise} \
  --metrics ${metrics} \
  --loss ${loss} \
  --warmup ${warmup} \
  --lrscheduler_start ${lrscheduler_start} \
  --lrscheduler_step ${lrscheduler_step} \
  --lrscheduler_decay ${lrscheduler_decay} \
  --exp-name ${exp_name} \
  --model_type ${model_type} \
  --run_type train

