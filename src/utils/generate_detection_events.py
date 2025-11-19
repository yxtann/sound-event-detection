import scaper
import os
import numpy as np
import soundfile as sf

from loguru import logger
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login

from src.utils.download_classification_dataset import download_classification_dataset

from src.constants import (
    FG_PATH,
    BG_PATH,
    OUTPUT_PATH,
    N_SCENES,
    SCENE_DURATION,
    START_SEED,
    CLASSES,
)


def generate_detection_events(split: str, num_scenes: int):

    AUDIO_OUTPUT_PATH = os.path.join(OUTPUT_PATH, split)
    os.makedirs(AUDIO_OUTPUT_PATH, exist_ok=True)

    for n in range(num_scenes):
        logger.info(f"Generating scene {n+1}/{num_scenes}...")

        sc = scaper.Scaper(
            SCENE_DURATION,
            fg_path=f"{FG_PATH}/{split}",
            bg_path=BG_PATH,
            random_state=START_SEED + n,
        )
        sc.ref_db = -20

        # Add background
        sc.add_background(
            label=("choose", []),  # Randomly choose any background
            source_file=("choose", []),  # Randomly choose any file
            source_time=("const", 0),
        )

        # Add foreground events
        n_events = np.random.randint(1, 2)
        for _ in range(n_events):
            sc.add_event(
                label=("choose", []),
                source_file=("choose", []),  # Randomly choose a file from that label
                source_time=("const", 0),
                event_time=(
                    "uniform",
                    0,
                    SCENE_DURATION - 1.0,
                ),  # Ensure event starts before the end
                event_duration=("uniform", 1.0, 4.0),  # Make event 1-4 sec long
                snr=(
                    "uniform",
                    10,
                    30,
                ),  # Random volume. This is CRITICAL for a robust model!
                pitch_shift=("uniform", -1.0, 1.0),  # Random pitch shift
                time_stretch=("uniform", 0.8, 1.2),  # Random time stretch
            )

        # Generate the audio and annotation; annotation in jams format
        audio_file = os.path.join(AUDIO_OUTPUT_PATH, f"{split}_scene_{n:04d}.wav")
        jams_file = os.path.join(AUDIO_OUTPUT_PATH, f"{split}_scene_{n:04d}.jams")

        sc.generate(
            audio_file,
            jams_file,
            allow_repeated_label=False,
            allow_repeated_source=True,
        )

    logger.info(f"Done generating scenes for {split} split.")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    login(token=os.getenv("HF_TOKEN"))
    
    for split in ["train", "test"]:

        # Skip download if already exists
        if os.path.exists(f"data/processed/classification/{split}"):
            pass
        else:
            download_classification_dataset(
                "kuross/dl-proj-classification", "data/processed/classification", split
            )

        generate_detection_events(split, N_SCENES)
