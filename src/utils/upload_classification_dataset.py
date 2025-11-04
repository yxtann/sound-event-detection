import os

import librosa

from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from datasets import Dataset, Audio, ClassLabel
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

    ds = Dataset.from_dict({"file_path": filepaths, "label": labels})

    logger.info("Calculating durations...")
    ds = ds.map(add_duration, num_proc=1)

    logger.info("Casting features...")
    ds = ds.cast_column("label", ClassLabel(names=sorted(class_names)))

    ds = ds.cast_column("file_path", Audio())
    ds = ds.rename_column("file_path", "file")

    logger.info("Dataset created successfully!")
    logger.info(ds)

    logger.info("\nExample row:")
    logger.info(ds[0])

    ds.push_to_hub("kuross/dl-proj-classification")
    logger.info("Successfully pushed to Hub!")

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    login(token=os.getenv("HF_TOKEN"))
    upload_classification_dataset(data_dir = Path("data/raw/final_dataset/classification"))