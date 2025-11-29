import os

from pathlib import Path
from huggingface_hub import snapshot_download

import pandas as pd
import pickle
import librosa
import matplotlib.pyplot as plt
import numpy as np

from loguru import logger
from tqdm import tqdm

from src.config import DETECTION_TRAIN_PATH, DETECTION_TEST_PATH


def download_detection_dataset():
    if not DETECTION_TRAIN_PATH.exists() or not DETECTION_TEST_PATH.exists():
        repo_id = "kuross/dl-proj-detection"
        logger.info(f"Detection dataset not found! Downloading from {repo_id}...")
        snapshot_download(
            repo_id, repo_type="dataset", local_dir=DETECTION_TRAIN_PATH.parent
        )
        logger.info("Detection dataset downloaded successfully")
    else:
        logger.info("Detection dataset already downloaded")


def get_mel_spec(file, sr, n_fft=2048, hop_length=512, n_mels=64):
    y, sr = librosa.load(file, sr=sr)
    fmax = sr // 2
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=fmax,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)  # log scale (dB)
    return S_db


def get_stft_spec(filename, path, sr):
    y, sr = librosa.load(path / filename, sr=sr)
    D = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(D, ref=np.max)
    return S_db


def display_spectrogram(S_db=None, sr=44100, onset=None, offset=None, file=None):
    """
    To plot from either S_db or the filename and path
    """
    if S_db is None:
        if file is not None:
            S_db = get_mel_spec(file, sr)
        else:
            raise Exception('Provide either S_db or file and path')
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    if (onset is not None) & (offset is not None):
        plt.axvline(onset)
        plt.axvline(offset)
    plt.tight_layout()
    plt.show()


def get_wav_files(input_path: Path, export_path: Path, split: str):
    # Gather all the wav files in the path
    wav_files = os.listdir(input_path)
    wav_files = [i for i in wav_files if i.endswith(".wav")]
    wav_files.sort()

    annotations = (
        pd.read_csv(input_path / "annotations.csv")
        .sort_values(by="filename")
        .reset_index(drop=True)
    )
    # annotations['filename'] = annotations['filename'].str[6:]
    annot_filter = annotations.query(f"onset != 0")
    assert (
        annot_filter["filename"] == wav_files
    ).all(), "Available files and annotation.csv filenames do not add up"

    # Get the spectrograms
    S_db_all = []
    sr = 44100
    logger.info(f"Getting {split} spectrograms for {len(wav_files)} files in {input_path}...")
    for file in tqdm(wav_files, desc="Getting spectrograms"):
        S_db = get_mel_spec(os.path.join(input_path, file), sr)
        S_db_all.append(S_db)

    # Export the spectrograms as .pkl files
    logger.info(f"Exporting {split} spectrograms to {export_path}...")
    export = {
        "sr": sr,
        "S_db": S_db_all,
        "files": wav_files,
        "onset": np.array(annot_filter["onset"]),
        "offset": np.array(annot_filter["offset"]),
        "event_label": np.array(annot_filter["event_label"]),
        "background_label": np.array(annotations.query(f"onset == 0")["event_label"]),
    }
    with open(export_path / f"spectrograms_{split}.pkl", "wb") as f:
        pickle.dump(export, f)


def create_spectrogram_pkl():
    download_detection_dataset()
    yamnet_data_path = Path("data") / "processed" / "yamnet"
    if  os.path.exists(yamnet_data_path / "spectrograms_train.pkl") and os.path.exists(yamnet_data_path / "spectrograms_test.pkl"):
        logger.info("YAMNet dataset files already exist")
        return
    logger.info("Creating YAMNet dataset files...")
    get_wav_files(DETECTION_TRAIN_PATH, yamnet_data_path, "train")
    get_wav_files(DETECTION_TEST_PATH, yamnet_data_path, "test")


if __name__ == "__main__":
    create_spectrogram_pkl()