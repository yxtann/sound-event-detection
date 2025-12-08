import pandas as pd
import soundfile as sf
import os
from tqdm import tqdm
from pathlib import Path

def extract_audio_events_from_csv(csv_path, output_path, split):
    df = pd.read_csv(csv_path)
    condition = df["onset"] > 0.0
    df = df[condition]
    detection_base_path = "data/processed/detection"

    for _, row in tqdm(df.iterrows(), desc=f"Extracting audio events from {csv_path}"):
        filename = row["filename"].replace(".wav", "")
        onset = row["onset"]
        offset = row["offset"]
        event_label = row["event_label"]

        if not os.path.exists(Path(output_path) / event_label):
            os.makedirs(Path(output_path) / event_label)

        audio_path = Path(detection_base_path) / split / f"{filename}.wav"
        audio_data, sr = sf.read(audio_path)
        audio_data = audio_data[int(onset * sr):int(offset * sr)]
        sf.write(Path(output_path) / event_label / f"{filename}_{onset:.2f}_{offset:.2f}.wav", audio_data, sr)


if __name__ == "__main__":
    extract_audio_events_from_csv("data/processed/detection/train/annotations.csv", "data/processed/classification/train_noisy", "train")
    extract_audio_events_from_csv("data/processed/detection/test/annotations.csv", "data/processed/classification/test_noisy", "test")