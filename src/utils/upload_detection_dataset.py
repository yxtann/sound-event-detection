import pandas as pd
import glob
import jams
import os

from loguru import logger
from dotenv import load_dotenv, find_dotenv
from datasets import DatasetDict, Dataset
from huggingface_hub import login, upload_folder

os.environ["HF_HUB_HTTP_TIMEOUT"] = "10000"


def upload_detection_dataset():

    REPO_ID = "kuross/dl-proj-detection"
    AUDIO_DIR = "data/processed/detection/"

    ds_dict = DatasetDict()

    for split in ["train", "test"]:

        BASE_PATH = os.path.join("data", "processed", "detection", split)
        JAMS_PATH = BASE_PATH
        OUTPUT_FILE = os.path.join(BASE_PATH, "annotations.csv")
        all_annotations = []

        # Find all .jams files
        jams_files = glob.glob(os.path.join(JAMS_PATH, "*.jams"))
        if not jams_files:
            logger.warning(f"No .jams files found in {JAMS_PATH}. Skipping split.")
            continue

        logger.info(f"Processing {len(jams_files)} files for split {split}...")

        for jams_file in jams_files:
            try:
                # Load the jams annotation
                jam = jams.load(jams_file)

                # Get the audio filename relative to the repo root
                audio_filename = os.path.basename(jams_file).replace(".jams", ".wav")

                # Find the 'scaper' annotations
                scaper_anns = jam.annotations.search(namespace="scaper")

                if scaper_anns:
                    for obs in scaper_anns[0].data:
                        event = obs.value
                        all_annotations.append(
                            {
                                "filename": audio_filename,
                                "onset": obs.time,
                                "offset": obs.time + obs.duration,
                                "event_label": event["label"],
                            }
                        ),
            except Exception as e:
                logger.error(f"Failed to process {jams_file}: {e}")

        if all_annotations:
            df = pd.DataFrame(all_annotations)
            df.to_csv(OUTPUT_FILE, index=False)
            ds_dict[split] = Dataset.from_pandas(df)
            logger.info(f"Finished split {split}: {len(df)} annotations.")
        else:
            logger.warning(f"No annotations found for split {split}.")

    if ds_dict:
        logger.info("Pushing dataset metadata to Hub...")
        ds_dict.push_to_hub(REPO_ID)

    if os.path.exists(AUDIO_DIR):
        logger.info("Uploading audio files...")
        upload_folder(
            folder_path=AUDIO_DIR,
            repo_id=REPO_ID,
            path_in_repo=".",
            repo_type="dataset",
        )
    else:
        logger.warning(
            f"Audio directory {AUDIO_DIR} not found. Audio files were not uploaded."
        )


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        upload_detection_dataset()
    else:
        logger.error("HF_TOKEN not found in .env file.")
