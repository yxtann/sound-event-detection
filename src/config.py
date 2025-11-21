from pathlib import Path

# General settings
CLASSES = ["cough", "dog_bark", "gun_shot", "siren", "car_horn"]

# Data
DOWNLOAD_PATH = Path("data") / "raw"
AUDIO_DETECTION_ANNOT_PATH = Path("")
DETECTION_TRAIN_PATH = Path("data") / "processed" / "detection" / "train"
DETECTION_TEST_PATH = Path("data") / "processed" / "detection" / "test"

YAMNET_DATA_PATH = Path("data") / "processed" / "yamnet"

# Scaper (soundscape generator) settings
FG_PATH = Path("data") / "processed" / "classification"
BG_PATH = Path("data") / "raw" / "background"
OUTPUT_PATH = Path("data") / "processed" / "detection"
N_SCENES = 500
SCENE_DURATION = 10.0
START_SEED = 42

# Audio Mamba
AUDIO_MAMBA_INDEX_LIST = [
    {"index": 0, "mid": f"/m/{CLASSES[0]}", "display_name": CLASSES[0]},
    {"index": 1, "mid": f"/m/{CLASSES[1]}", "display_name": CLASSES[1]},
    {"index": 2, "mid": f"/m/{CLASSES[2]}", "display_name": CLASSES[2]},
    {"index": 3, "mid": f"/m/{CLASSES[3]}", "display_name": CLASSES[3]},
    {"index": 4, "mid": f"/m/{CLASSES[4]}", "display_name": CLASSES[4]},
]
