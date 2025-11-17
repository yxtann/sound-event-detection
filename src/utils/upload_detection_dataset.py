import pandas as pd
import glob
import jams
import os

from loguru import logger
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from huggingface_hub import login, upload_folder
from datasets import DatasetDict, Dataset


def upload_detection_dataset():

    ds_dict = DatasetDict()

    for split in ["train", "test"]:

        JAMS_PATH = f"data/processed/detection/{split}"
        OUTPUT_FILE = f"data/processed/detection/{split}/annotations.csv"

        all_annotations = []

        # Find all .jams files
        jams_files = glob.glob(os.path.join(JAMS_PATH, "*.jams"))

        for jams_file in jams_files:
            # Load the jams annotation
            jam = jams.load(jams_file)

            # Get the audio filename (e.g., "audio/scene_0000.wav")
            # We add the 'audio/' prefix to match the HF repo structure
            audio_filename = "audio/" + os.path.basename(jams_file).replace(
                ".jams", ".wav"
            )

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

            logger.info(
                f"Successfully created {OUTPUT_FILE} with {len(df)} annotations for split {split}."
            )

        ds_dict[split] = Dataset.from_pandas(df)

    ds_dict.push_to_hub("kuross/dl-proj-detection")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    login(token=os.getenv("HF_TOKEN"))
    upload_detection_dataset()
