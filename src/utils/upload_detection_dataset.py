import pandas as pd
import glob
import jams
import os

from loguru import logger
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from huggingface_hub import login, upload_folder

from src.constants import REGENERATE_AUDIO_DETECTION_SET
from src.utils import generate_audio_events


def upload_detection_dataset():

    JAMS_PATH = "data/detection/audio"
    OUTPUT_FILE = "data/detection/audio/annotations.csv"

    all_annotations = []

    # Find all .jams files
    jams_files = glob.glob(os.path.join(JAMS_PATH, "*.jams"))

    for jams_file in jams_files:
        # Load the jams annotation
        jam = jams.load(jams_file)

        # Get the audio filename (e.g., "audio/scene_0000.wav")
        # We add the 'audio/' prefix to match the HF repo structure
        audio_filename = "audio/" + os.path.basename(jams_file).replace(".jams", ".wav")

        # Find the 'scaper' annotations
        scaper_anns = jam.annotations.search(namespace="scaper")
        if not scaper_anns:
            continue

        # Iterate over every event scaper logged
        for obs in scaper_anns[0].data:
            event = obs.value

            all_annotations.append(
                {
                    "filename": audio_filename,
                    "onset": obs.time,
                    "offset": obs.time + obs.duration,
                    "event_label": event["label"],
                }
            )

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_annotations)
    df.to_csv(OUTPUT_FILE, index=False)

    logger.info(f"Successfully created {OUTPUT_FILE} with {len(df)} annotations.")

    upload_folder(
        folder_path="data/detection/audio",
        repo_id="kuross/dl-proj-detection",
        repo_type="dataset",
    )


if __name__ == "__main__":
    if REGENERATE_AUDIO_DETECTION_SET:
        logger.info("Regenerating audio detection set...")
        generate_audio_events()

    load_dotenv(find_dotenv())
    login(token=os.getenv("HF_TOKEN"))
    upload_detection_dataset()
