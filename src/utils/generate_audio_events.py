import scaper
import os
import glob
import numpy as np
from loguru import logger

from src.constants import (
    FG_PATH,
    BG_PATH,
    OUTPUT_PATH,
    N_SCENES,
    SCENE_DURATION,
    START_SEED,
)


def generate_audio_events():

    AUDIO_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "audio")
    os.makedirs(AUDIO_OUTPUT_PATH, exist_ok=True)

    for n in range(N_SCENES):
        logger.info(f"Generating scene {n+1}/{N_SCENES}...")

        sc = scaper.Scaper(
            SCENE_DURATION,
            fg_path=FG_PATH,
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
                label=("choose", []),  # Randomly choose a label
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
        audio_file = os.path.join(AUDIO_OUTPUT_PATH, f"scene_{n:04d}.wav")
        jams_file = os.path.join(AUDIO_OUTPUT_PATH, f"scene_{n:04d}.jams")

        sc.generate(
            audio_file,
            jams_file,
            allow_repeated_label=False,
            allow_repeated_source=True,
        )

    logger.info("Done generating scenes.")


if __name__ == "__main__":
    generate_audio_events()
