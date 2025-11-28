import os
import csv
import json
import random

from pathlib import Path

import pandas as pd

from loguru import logger
from tqdm import tqdm

from src.config import AUDIO_MAMBA_INDEX_LIST, CLASSES


class AudioMambaMetadataGenerator:
    def __init__(
        self, data_train_path: str, data_val_path: str, metadata_output_path: str
    ):
        self.data_train_path = Path(data_train_path)
        self.data_val_path = Path(data_val_path)
        self.metadata_output_path = Path(metadata_output_path)
        self.metadata_output_path.mkdir(parents=True, exist_ok=True)

    def generate_class_labels_indices(self):
        logger.info("Generating class labels indices...")
        with open(
            Path(self.metadata_output_path) / "class_labels_indices.csv", "w"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["index", "mid", "display_name"])
            for index in tqdm(AUDIO_MAMBA_INDEX_LIST):
                writer.writerow(
                    [index["index"], index["mid"], f'"{index["display_name"]}"']
                )
        logger.info(
            f"Class labels indices generated and saved to {self.metadata_output_path}"
        )

    def generate_metadata_from_folders(self, dataset_type: str):

        logger.info(f"Generating metadata for {dataset_type}...")
        if dataset_type == "train":
            data_path = self.data_train_path
        elif dataset_type == "val":
            data_path = self.data_val_path
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        dataset_dict = {"data": []}

        for sound_class in CLASSES:
            sound_class_path = data_path / sound_class
            logger.info(f"Loading {sound_class}...")
            audio_files = list(sound_class_path.glob("*.wav"))

            for audio_file in tqdm(audio_files):
                dataset_dict["data"].append(
                    {"wav": str(audio_file), "labels": f"/m/{sound_class}"}
                )
            logger.info(
                f"Loaded {len(dataset_dict['data'])} audio files for {sound_class}"
            )

        logger.info(f"Generated metadata for {dataset_type} successfully!")

        with open(f"{self.metadata_output_path}/{dataset_type}_data.json", "w") as f:
            json.dump(dataset_dict, f)


def generate_audio_mamba_metadata():
    logger.info("Generating Audio Mamba metadata...")
    amm_gen = AudioMambaMetadataGenerator(
        data_train_path="data/processed/classification/train",
        data_val_path="data/processed/classification/test",
        metadata_output_path="data/processed/audio_mamba/",
    )
    amm_gen.generate_class_labels_indices()
    amm_gen.generate_metadata_from_folders(dataset_type="train")
    amm_gen.generate_metadata_from_folders(dataset_type="val")


def generate_metadata_from_detector(groundtruth_csv, detector_audio_output_path):
    # TODO: Currently this is specific for audio mamba only!!
    classes_of_interest = CLASSES
    predicted_audio_event_files = os.listdir(detector_audio_output_path)
    ground_truth_events = pd.read_csv(groundtruth_csv)
    audio_metadata_dict = {"data": []}
    for _, ground_truth in ground_truth_events.iterrows():

        # Skip the background sound labels
        if ground_truth["event_label"] not in classes_of_interest:
            continue

        base_filename = ground_truth["filename"].split(".")[0]  # Remove the extension
        for predicted_audio_event_file in predicted_audio_event_files:
            if base_filename in predicted_audio_event_file:
                audio_metadata_dict["data"].append(
                    {
                        "wav": f"{detector_audio_output_path}/{predicted_audio_event_file}",
                        "labels": f"/m/{ground_truth['event_label']}",
                    }
                )
                break

    with open(f"{detector_audio_output_path}/audio_mamba_metadata.json", "w") as f:
        json.dump(audio_metadata_dict, f)

    logger.info(
        f"Created detector metadata at {detector_audio_output_path}/audio_mamba_metadata.json"
    )


if __name__ == "__main__":
    # generate_audio_mamba_metadata()
    generate_metadata_from_detector(
        groundtruth_csv="data/processed/detection/test/annotations.csv",
        detector_audio_output_path="data/processed/yamnet/extracted_audio/",
    )
