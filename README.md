# Sound Event Detection

This repository investigates the possibility of using multi-stage detectors and classifiers for audio classification much like how R-CNN works for images.  

Our motivation is to use small detectors which can potentially be deployed on edge devices with large classifiers that may sit on large servers and can handle multiple input streams (e.g., from many detectors) at once. To this end, we want to quickly detect if there's an event, but ensure that the classification of the event is accurate as well, thus combining the speed of small models with the accuracy of large models.

## Architecture

Our proposed architecture is shown below.

![Two Stage](diagrams/two_stage.png)

## Models Used

We compared the following models
- Convolutional Recurrent Neural Networks (R-CNN) - single stage only
- Yet Another Mobile Network (YAMNet) - single stage and as detector
- Hierarchical Token Semantic Audio Transformer (HTS-AT) - single stage and as classifier
- Audio Mamba (AuM) - as classifier only

## Using the Repository

### Running Single Stage Pipeline

### Running Two-Stage Pipeline

### Generating Data



Detect audio events using deep learning techniques.
This repository contains code for training and evaluating models for sound event detection.
It includes data preprocessing, model architectures, and evaluation metrics.
To get started, clone the repository and install the required dependencies listed in requirements.txt.

Run in root folder with `python -m src.module`

Create a `.env` file in the repo root and insert your HF_TOKEN

i.e.
```python
# .env
HF_TOKEN = xxxx
```

### Generating soundscapes
To generate, scaper and soxbindings (from submodule, and install from source) is required. Otherwise, this can be ignored.