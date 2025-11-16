# General settings
CLASSES = ["cough", "dog_bark", "gun_shot", "siren"]

# Data
DOWNLOAD_PATH = "data/raw"
REGENERATE_AUDIO_DETECTION_SET = False
AUDIO_DETECTION_ANNOT_PATH = ""

# Scaper (soundscape generator) settings
FG_PATH = "data/classification/"
BG_PATH = "data/background/"
OUTPUT_PATH = "data/detection"
N_SCENES = 200
SCENE_DURATION = 10.0
START_SEED = 42

# Audio Mamba
AUDIO_MAMBA_INDEX_LIST = [
    {"index": 0, "mid": f"/m/{CLASSES[0]}", "display_name": CLASSES[0]},
    {"index": 1, "mid": f"/m/{CLASSES[1]}", "display_name": CLASSES[1]},
    {"index": 2, "mid": f"/m/{CLASSES[2]}", "display_name": CLASSES[2]},
    {"index": 3, "mid": f"/m/{CLASSES[3]}", "display_name": CLASSES[3]},
]
