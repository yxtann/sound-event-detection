import os
import soundfile as sf
import librosa
import numpy as np

from loguru import logger
from tqdm import tqdm

from pathlib import Path


class AmplitudeFilterSnip:
    def __init__(self, audio_in_path: Path, audio_out_path: Path):
        """Snips audio according to noise threshold

        Args:
            audio_in_path (Path): Input folder containing all classes
            audio_out_path (Path): Output folder containing all snipped audio
        """
        self.audio_in_path = audio_in_path
        self.audio_out_path = audio_out_path
        self.all_data = []

    def collect_all_files(self):

        # Walk through the classification audio folder
        for root, dirs, files in os.walk(self.audio_in_path):
            curr_label = os.path.basename(root)
            for file_name in files:

                # Skip non-wav files
                if not file_name.endswith(".wav"):
                    continue

                file_path = os.path.join(root, file_name)
                record = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "label": curr_label,
                    "onset": 0.0,
                    "offset": 0.0,
                }
                self.all_data.append(record)

    def find_threshold(self):
        HOP_LENGTH = 512
        FRAME_LENGTH = 2048
        THRESHOLD_FACTOR = 1.5
        ABSOLUTE_NOISE_THRESHOLD = 0.05
        BOUNDARY_BUFFER_TIME = 0.10
        MAX_RMS_STD_FOR_CONSTANT_CLIP = 0.01
        MAX_ENERGY_RANGE_FOR_CONSTANT_CLIP = 0.01

        for i, record in tqdm(
            enumerate(self.all_data),
            total=len(self.all_data),
            desc="Determining thresholds...",
        ):
            audio, sr = librosa.load(record["file_path"], sr=None)
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            duration = librosa.get_duration(y=audio, sr=sr)

            # Calculate RMS energy
            rms_energy = librosa.feature.rms(
                y=audio, frame_length=2048, hop_length=HOP_LENGTH
            )[0]

            # Determine Threshold
            noise_floor = np.percentile(rms_energy, 10)
            rms_10th = np.percentile(rms_energy, 10)
            rms_90th = np.percentile(rms_energy, 90)
            energy_range = rms_90th - rms_10th

            rms_std_dev = np.std(rms_energy)

            # --- 1. Global Noise Floor Check ---
            if noise_floor >= ABSOLUTE_NOISE_THRESHOLD:
                # ACCEPT ENTIRE CLIP LOGIC
                time_onset = 0.0
                time_offset = duration
                rms_onset_value = np.max(rms_energy)  # Using max for the 'onset' RMS
                rms_offset_value = np.min(rms_energy)  # Using min for the 'offset' RMS

                # print(f"🚨 Noise Floor ({noise_floor:.4f}) is above the absolute threshold ({ABSOLUTE_NOISE_THRESHOLD:.4f}).")
                # print(f"➡️ Accepting the entire clip: 0.000 s to {duration:.3f} s")
            elif rms_std_dev <= MAX_RMS_STD_FOR_CONSTANT_CLIP:
                # ACCEPT ENTIRE CLIP LOGIC
                time_onset = 0.0
                time_offset = duration
                rms_onset_value = np.max(rms_energy)
                rms_offset_value = np.min(rms_energy)

                # print(f"🚨 RMS Energy distribution is too narrow (Std Dev: {rms_std_dev:.4f}).")
                # print(f"➡️ Accepting the entire clip due to constant loudness.")
            elif energy_range <= MAX_ENERGY_RANGE_FOR_CONSTANT_CLIP:
                # ACCEPT ENTIRE CLIP LOGIC
                time_onset = 0.0
                time_offset = duration
                rms_onset_value = np.max(rms_energy)
                rms_offset_value = np.min(rms_energy)

                # print(f"🚨 RMS Energy range ({energy_range:.4f}) is too narrow.")
                # print(f"➡️ Accepting the entire clip due to constant loudness.")
            else:
                # --- 2. Standard Onset/Offset Detection ---
                threshold = noise_floor * THRESHOLD_FACTOR
                onset_frame_indices = np.where(rms_energy > threshold)[0]

                if len(onset_frame_indices) > 0:
                    # Initial Detection
                    onset_frame = onset_frame_indices[0]
                    time_onset = librosa.frames_to_time(
                        onset_frame, sr=sr, hop_length=HOP_LENGTH
                    )

                    offset_frame = onset_frame_indices[-1]
                    time_offset = librosa.frames_to_time(
                        offset_frame, sr=sr, hop_length=HOP_LENGTH
                    )

                    # # Check if Onset is near the start
                    # if time_onset_detected <= BOUNDARY_BUFFER_TIME:
                    #     print(f"Onset {time_onset_detected} is near start, setting to start")
                    #     time_onset_detected = 0

                    # # Check if Offset is near the end
                    # if time_offset_detected >= (duration - BOUNDARY_BUFFER_TIME):
                    #     print(f"Offset {time_offset_detected} is near end, setting to {duration:.2f}")
                    #     time_offset_detected = duration

                else:
                    # Clip is too quiet even for standard detection
                    time_onset = 0
                    time_offset = duration
                    print(
                        "Sound event not detected above threshold. Onset/Offset set to full duration."
                    )

            record["onset"] = time_onset
            record["offset"] = time_offset

            # Replace with the new onset and offset
            self.all_data[i] = record

    def cut_audio(self):
        for record in tqdm(
            self.all_data, total=len(self.all_data), desc="Cutting audio..."
        ):
            audio, sr = sf.read(record["file_path"])
            audio = audio[int(record["onset"] * sr) : int(record["offset"] * sr)]
            os.makedirs(self.audio_out_path / record["label"], exist_ok=True)
            sf.write(
                self.audio_out_path / record["label"] / record["file_name"],
                audio,
                sr,
            )

    def run_threshold_snip(self):
        self.collect_all_files()
        self.find_threshold()
        self.cut_audio()


if __name__ == "__main__":

    train_audio_snipper = AmplitudeFilterSnip(
        audio_in_path=Path("data/processed/classification/train"),
        audio_out_path=Path("data/processed/classification/train_snipped"),
    )
    train_audio_snipper.run_threshold_snip()

    test_audio_snipper = AmplitudeFilterSnip(
        audio_in_path=Path("data/processed/classification/test"),
        audio_out_path=Path("data/processed/classification/test_snipped"),
    )
    test_audio_snipper.run_threshold_snip()
