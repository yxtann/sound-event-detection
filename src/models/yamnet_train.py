import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import os

from loguru import logger

from tqdm import tqdm
from pathlib import Path

import sys
from src.config import DETECTION_TRAIN_PATH, DETECTION_TEST_PATH
from src.utils.audio_to_spectrograms import create_spectrogram_pkl

YAMNET_DETECTOR_CHECKPOINT = Path("checkpoints") / "yamnet_detector.pth"

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def preprocess_and_cache_features(file_list, cache_dir="data/cache/yamnet_embeddings"):
    """
    Runs YAMNet on all files and saves the embeddings as .pt files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Pre-computing features for {len(file_list)} files...")
    
    for file_path in tqdm(file_list):
        # Generate a unique filename for the cache
        filename = os.path.basename(file_path).replace('.wav', '.pt')
        save_path = os.path.join(cache_dir, filename)
        
        # Skip if already exists
        if os.path.exists(save_path):
            continue
            
        audio, _ = librosa.load(file_path, sr=16000, mono=True)
        _, embeddings, _ = yamnet_model(audio)
        embeddings_tensor = torch.tensor(embeddings.numpy(), dtype=torch.float32) # embeddings shape: (N_frames, 1024)
        torch.save(embeddings_tensor, save_path)

    return cache_dir

def detect_onsets_offsets(preds, threshold=0.5, hop_time=0.48):
    """Convert framewise probabilities into onset/offset pairs"""
    events = preds > threshold
    changes = np.diff(events.astype(int))
    onsets = (
        np.where(changes == 1)[0] + 1
    ) * hop_time  # first frame with pred > threshold
    offsets = (
        np.where(changes == -1)[0] + 1
    ) * hop_time  # until mid of frame (since frame width = 2*hop_time)
    if len(offsets) < len(onsets):
        offsets = np.append(offsets, (len(preds) + 1) * hop_time)
    elif len(offsets) > len(onsets):
        onsets = np.append(0, onsets)
    return onsets, offsets


class YAMNetSEDDataset(Dataset):
    def __init__(self, file_list, onset_list, offset_list, cache_dir, hop_time=0.48):
        self.file_list = file_list
        self.onset_list = onset_list
        self.offset_list = offset_list
        self.hop_time = hop_time
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        original_filename = os.path.basename(self.file_list[idx])
        cached_filename = original_filename.replace('.wav', '.pt')
        cache_path = os.path.join(self.cache_dir, cached_filename)
        embeddings = torch.load(cache_path) # Shape: [Time, 1024]

        num_frames = embeddings.shape[0]
        frame_times = np.arange(num_frames) * self.hop_time

        labels = np.zeros(num_frames, dtype=np.float32)
        onset, offset = self.onset_list[idx], self.offset_list[idx]
        labels[(frame_times >= onset) & (frame_times <= offset)] = 1

        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        return embeddings, y


class YAMNetGRU(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x)
        return self.sigmoid(x)


class Solver(object):
    def __init__(self, train_idx, **kwargs):
        self.epochs = kwargs.pop("epochs", 5)
        self.lr = kwargs.pop("lr", 1e-3)
        self.hop_time = kwargs.pop("hop_time", 0.48)
        self.sr = kwargs.pop("sr", 16000)
        self.device = kwargs.pop("device", "cpu")
        self.mode = kwargs.pop("mode", "train")

        self.train_path = kwargs.pop("train_path", "data/processed/yamnet/spectrograms_train.pkl")
        self.test_path = kwargs.pop("test_path", "data/processed/yamnet/spectrograms_test.pkl")
        self.checkpoint_path = kwargs.pop("checkpoint_path", YAMNET_DETECTOR_CHECKPOINT)

        data = pickle.load(open(self.train_path, "rb"))
        self.original_sr = data["sr"]
        self.data = data

        self.model = YAMNetGRU().to(self.device)

        if self.mode == "train":
            # The below is only used for training the model
            all_indices = list(range(len(data["event_label"])))
            all_files = [f"{DETECTION_TRAIN_PATH}/train_snipped_scene_{str(i).zfill(4)}.wav" for i in all_indices]
            self.cache_dir = preprocess_and_cache_features(all_files)
            assert max(train_idx) < len(data["event_label"])
            assert min(train_idx) >= 0

            # Training
            train_files = [all_files[i] for i in train_idx]
            train_dataset = YAMNetSEDDataset(
                train_files,
                data["onset"][train_idx],
                data["offset"][train_idx],
                self.cache_dir
            )
            self.trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

            # Validation
            test_idx = [i for i in range(len(data["event_label"])) if i not in train_idx]
            test_files = [all_files[i] for i in test_idx]
            test_dataset = YAMNetSEDDataset(
                test_files,
                data["onset"][test_idx],
                data["offset"][test_idx],
                self.cache_dir
            )
            self.testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
            self.criterion = nn.BCELoss().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def load_model(self, checkpoint_path=YAMNET_DETECTOR_CHECKPOINT):
        if checkpoint_path == None:
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from {checkpoint_path}")
        except Exception as e:
            raise ValueError(f"Invalid checkpoint format: {checkpoint_path}") from e

    def _train_step(self):
        self.model.train()
        total_loss = 0
        for X, y in self.trainloader:
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)  # shape [1, time, 1]
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.trainloader)

    def _val_step(self):
        total_loss = 0
        for X, y in self.testloader:
            with torch.no_grad():
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
        return total_loss / len(self.testloader)

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self._train_step()
            print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}")
            val_loss = self._val_step()
            print(f"Epoch {epoch+1}: valildation loss = {val_loss:.4f}")

        # Save the model at the end of training
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)
        logger.info(f"Saved model to {self.checkpoint_path}")

    def inference(self, test_files, threshold=0.5):#, plot_detection_viz=False, output_folder=None):
        events_dict = {}
        self.cache_dir = preprocess_and_cache_features(test_files)

        for file in tqdm(test_files, desc=f"Testing..."):
            original_filename = os.path.basename(file)
            cached_filename = original_filename.replace('.wav', '.pt')
            cache_path = os.path.join(self.cache_dir, cached_filename)
            embeddings = torch.load(cache_path) # Shape: [Time, 1024]

            #num_frames = embeddings.shape[0]
            #frame_times = np.arange(num_frames) * self.hop_time

            #labels = np.zeros(num_frames, dtype=np.float32)
            #onset, offset = self.data["onset"][idx], self.data["offset"][idx]
            #labels[(frame_times >= onset) & (frame_times <= offset)] = 1

            #X = torch.tensor(embeddings, dtype=torch.float32)
            #y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
            #event_label = self.data["event_label"][idx]

            with torch.no_grad():
                X = embeddings.to(self.device)
                #y = y.to(self.device)
                outputs = self.model(X)
                preds = outputs[:, 0].cpu().numpy()

            onsets, offsets = detect_onsets_offsets(
                preds, threshold=threshold, hop_time=self.hop_time
            )

            """if plot_detection_viz:
                filename = f"{DETECTION_TRAIN_PATH}/{original_filename}"
                audio, sr = librosa.load(filename, sr=None)
                plot_detection(
                    audio,
                    sr,
                    preds,
                    self.hop_time,
                    self.original_sr,
                    list(zip(onsets, offsets)),
                    labels=(onset, offset),
                    title=f"{idx} | {event_label}",
                    output_folder=output_folder,
                )"""

            # Event detection output
            events_dict[original_filename] = [{'file':original_filename, 'event_onset': float(o), 'event_offset': float(f)} for o, f in zip(onsets, offsets)]
        return events_dict

def plot_detection(
    audio,
    sr,
    preds,
    hop_time,
    original_sr=44100,
    onsets_offsets=None,
    labels=None,
    title=None,
    output_folder=None,
):
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    S = librosa.feature.melspectrogram(
        y=audio, sr=original_sr, n_fft=1024, hop_length=256, n_mels=128
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_db, sr=original_sr, hop_length=256, x_axis="time", y_axis="mel", ax=axs[0]
    )
    axs[0].set_title(title or "Mel Spectrogram")

    for onset, offset in onsets_offsets:
        # axs[0].axvspan(onset, offset, color='orange', alpha=0.3, label='Detected Event')
        axs[0].axvline(onset, label="pred", color="green")
        axs[0].axvline(offset, color="green")
    t_pred = np.arange(len(preds)) * hop_time
    axs[1].plot(t_pred, preds, label="Predicted prob", color="blue", marker="+")
    axs[1].axhline(0.5, color="gray", linestyle="--", label="Threshold")

    if labels is not None:
        onset, offset = labels
        axs[1].axvline(onset, label="actual", color="red")
        axs[1].axvline(offset, color="red")
        axs[0].axvline(onset, label="actual", color="red")
        axs[0].axvline(offset, color="red")

    for onset, offset in onsets_offsets:
        axs[1].axvline(onset, label="pred", color="green")
        axs[1].axvline(offset, color="green")

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].legend()
    plt.tight_layout()
    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, f"{title}.png"))
        plt.close()
    else:
        # Likely running on Jupyter notebook
        plt.show()


def train_yamnet(checkpoint_path=None):
    np.random.seed(0)
    train_data = pickle.load(
        open(Path("data") / "processed" / "yamnet" / "spectrograms_train.pkl", "rb")
    )
    train_size = 0.8
    train_idx = []
    for label in np.unique(train_data["event_label"]):
        choices = np.where(train_data["event_label"] == label)[0]
        train_idx.append(
            np.sort(
                np.random.choice(
                    choices,
                    size=int(np.round(len(choices) * train_size)),
                    replace=False,
                )
            )
        )
    train_idx = np.sort(np.concatenate(train_idx))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Start training on {device}")
    solver = Solver(
        epochs=5,
        train_idx=train_idx,
        device=torch.device(device),
        checkpoint_path=checkpoint_path,
        mode="train",
    )
    solver.train()
    return solver

def run_yamnet(test_files,
    checkpoint_path = YAMNET_DETECTOR_CHECKPOINT,
):
    if not os.path.exists(checkpoint_path):
        logger.info("No model checkpoint, training model...")
        trained_solver = train_yamnet(checkpoint_path=checkpoint_path)
        logger.info(f"Saved solver to {checkpoint_path}")
    else:
        logger.info(f"Loading from checkpoint {checkpoint_path}...")
        trained_solver = Solver(
            train_idx=[], checkpoint_path=checkpoint_path, mode="infer"
        )
        trained_solver.load_model(checkpoint_path)

    events_list = trained_solver.inference(test_files)
    return events_list


if __name__ == "__main__":
    test_path = Path("data") / "processed" / "yamnet" / "spectrograms_test.pkl"
    test_data = pickle.load(open(test_path, "rb"))
    filepaths = [os.path.join(DETECTION_TEST_PATH, file) for file in test_data['files']]
    events_list = run_yamnet(filepaths)
    print(events_list)
