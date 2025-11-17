import os
import shutil
from pathlib import Path
import soundfile as sf

from loguru import logger
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from tqdm import tqdm


def download_classification_dataset(repo_id: str, output_dir: str, split="train"):
    """
    Downloads a HF dataset and saves it in a folder
    structure that Scaper understands:
    output_dir/label_A/file1.wav
    output_dir/label_B/file2.wav
    """

    ds = load_dataset(repo_id)
    output_path = Path(output_dir)

    logger.info(f"Loading dataset '{repo_id}'...")
    label_feature = ds[split].features["label"]
    logger.info(f"Exporting '{split}' split to {output_path.resolve()}...")

    for example in tqdm(ds[split]):
        # Get audio data
        audio_data = example["file"]["array"]
        sr = example["file"]["sampling_rate"]

        # Get label name
        label_name = label_feature.int2str(example["label"])

        # Create label sub-directory
        label_dir = output_path / split / label_name
        label_dir.mkdir(parents=True, exist_ok=True)

        # Get a unique filename (e.g., from the 'path' or just generate one)
        # Using the original path's stem is safest for uniqueness
        original_path = Path(example["file"]["path"])
        filename = f"{original_path.stem}.wav"

        sf.write(label_dir / filename, audio_data, sr)

    logger.info("Export complete!")


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    login(token=os.getenv("HF_TOKEN"))
    for split in ["train", "test"]:
        download_classification_dataset(
            "kuross/dl-proj-classification", "data/processed/classification", split
        )
