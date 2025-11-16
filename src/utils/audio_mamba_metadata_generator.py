import os
import csv
import json
import random

from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.constants import AUDIO_MAMBA_INDEX_LIST, CLASSES


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
        with open(f"{self.metadata_output_path}/class_labels_indices.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "mid", "display_name"])
            for index in tqdm(AUDIO_MAMBA_INDEX_LIST):
                writer.writerow([index["index"], index["mid"], f'"{index["display_name"]}"'])
        logger.info(
            f"Class labels indices generated and saved to {self.metadata_output_path}"
        )

    def generate_metadata(self, dataset_type: str):

        # TODO Naive split, do proper train-test-val split later on..

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

            # TODO: Do proper train val split
            if dataset_type == "train":
                audio_files = random.sample(audio_files, int(len(audio_files) * 0.8))
            if dataset_type == "val":
                audio_files = random.sample(audio_files, int(len(audio_files) * 0.2))

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


if __name__ == "__main__":
    amm_gen = AudioMambaMetadataGenerator(
        data_train_path="data/raw/classification",  # Same path for now..
        data_val_path="data/raw/classification",
        metadata_output_path="data/processed/audio_mamba/",
    )
    amm_gen.generate_class_labels_indices()
    amm_gen.generate_metadata(dataset_type="train")
    amm_gen.generate_metadata(dataset_type="val")
