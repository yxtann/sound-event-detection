import os
import librosa

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, Audio, ClassLabel, DatasetDict
from loguru import logger
from huggingface_hub import login


def add_duration(example):
    example["duration"] = librosa.get_duration(path=example["file_path"])
    return example


def upload_classification_dataset(data_dir: Path):

    filepaths = []
    labels = []
    class_names = []

    # Find all subdirectories (which are our classes)
    for class_folder in data_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            class_names.append(class_name)

            # Find all audio files in this class folder
            # Add any other extensions you have (e.g., ".ogg", ".m4a")
            for audio_file in class_folder.glob("*"):
                if audio_file.suffix in [".wav", ".mp3", ".flac", ".webm"]:
                    filepaths.append(str(audio_file))
                    labels.append(class_name)

    logger.info(f"Found {len(filepaths)} files in {len(class_names)} classes.")

    train_filepaths, test_filepaths, train_labels, test_labels = train_test_split(
        filepaths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    ds_dict = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"file_path": train_filepaths, "label": train_labels}
            ),
            "test": Dataset.from_dict(
                {"file_path": test_filepaths, "label": test_labels}
            ),
        }
    )

    logger.info("Calculating durations...")
    ds_dict = ds_dict.map(add_duration, num_proc=1)

    logger.info("Casting features...")
    class_label_feature = ClassLabel(names=sorted(class_names))
    ds_dict = ds_dict.cast_column("label", class_label_feature)

    ds_dict = ds_dict.cast_column("file_path", Audio())
    ds_dict = ds_dict.rename_column("file_path", "file")

    logger.info("Dataset created successfully!")
    logger.info(ds_dict)

    logger.info("\nExample row (from train split):")
    logger.info(ds_dict["train"][0])

    ds_dict.push_to_hub("kuross/dl-proj-classification")
    logger.info("Successfully pushed to Hub!")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    login(token=os.getenv("HF_TOKEN"))
    upload_classification_dataset(
        data_dir=Path("data/raw/classification")
    )
