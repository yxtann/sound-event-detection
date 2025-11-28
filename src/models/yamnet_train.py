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
from src.config import DETECTION_TRAIN_PATH, DETECTION_TEST_PATH
from src.utils.audio_to_spectrograms import create_spectrogram_pkl

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")


def extract_yamnet_embeddings(audio, sr=16000):
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    _, embeddings, _ = yamnet_model(audio)
    return embeddings.numpy()


def detect_onsets_offsets(preds, threshold=0.5, hop_time=0.48):
    """Convert framewise probabilities into onset/offset pairs"""
    events = preds > threshold
    changes = np.diff(events.astype(int))
    onsets = (
        np.where(changes == 1)[0] + 1
    ) * hop_time  # first frame with pred > threshold
    offsets = (
        np.where(changes == -1)[0] + 2
    ) * hop_time  # until end of frame (frame width = 2*hop_time)
    if len(offsets) < len(onsets):
        offsets = np.append(offsets, (len(preds) + 1) * hop_time)
    elif len(offsets) > len(onsets):
        onsets = np.append(0, onsets)
    return list(zip(onsets, offsets))


class YAMNetSEDDataset(Dataset):
    def __init__(self, file_list, onset_list, offset_list, sr=16000, hop_time=0.48):
        self.file_list = file_list
        self.onset_list = onset_list
        self.offset_list = offset_list
        self.hop_time = hop_time

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.file_list[idx], sr=None)
        embeddings = extract_yamnet_embeddings(audio, sr)
        num_frames = embeddings.shape[0]
        frame_times = np.arange(num_frames) * self.hop_time

        labels = np.zeros(num_frames, dtype=np.float32)
        onset, offset = self.onset_list[idx], self.offset_list[idx]
        labels[(frame_times >= onset) & (frame_times <= offset)] = 1

        X = torch.tensor(embeddings, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        return X, y


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
        self.epochs = kwargs.pop("epochs", 10)
        self.lr = kwargs.pop("lr", 1e-3)
        self.path_prefix = kwargs.pop("path_prefix", "..")
        self.hop_time = kwargs.pop("hop_time", 0.48)
        self.sr = kwargs.pop("sr", 16000)
        self.device = kwargs.pop("device", "cpu")
        self.train_path = kwargs.pop(
            "train_path", "data/processed/yamnet/spectrograms_train.pkl"
        )
        self.test_path = kwargs.pop(
            "test_path", "data/processed/yamnet/spectrograms_test.pkl"
        )
        self.checkpoint_path = kwargs.pop("checkpoint_path", None)

        data = pickle.load(open(self.train_path, "rb"))
        self.original_sr = data["sr"]
        self.data = data

        self.mode = kwargs.pop("mode", "train")

        if self.mode == "train":
            # The below is only used for training the model
            assert max(train_idx) < len(data["event_label"])
            assert min(train_idx) >= 0

            file_path = DETECTION_TRAIN_PATH

            # Training
            train_files = [
                f"{file_path}/train_scene_0{str(i).zfill(3)}.wav" for i in train_idx
            ]
            train_dataset = YAMNetSEDDataset(
                train_files,
                data["onset"][train_idx],
                data["offset"][train_idx],
                data["event_label"][train_idx],
            )
            self.trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

            # Validation
            test_idx = [
                i for i in range(len(data["event_label"])) if i not in train_idx
            ]
            test_files = [
                f"{file_path}/train_scene_0{str(i).zfill(3)}.wav" for i in test_idx
            ]
            test_dataset = YAMNetSEDDataset(
                test_files,
                data["onset"][test_idx],
                data["offset"][test_idx],
                data["event_label"][test_idx],
            )
            self.testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            self.criterion = nn.BCELoss().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model = YAMNetGRU().to(self.device)

    def load_model(self, checkpoint_path):
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
        checkpoint_path = os.path.join("checkpoints", "yamnet_detector.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Saved model to {checkpoint_path}")

    """def evaluate(self, threshold=0.5):
        total_loss = 0
        for X, y, audio, labels, event_label, file in self.testloader:
            with torch.no_grad():
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                preds = outputs[0, :, 0].numpy()

            events = detect_onsets_offsets(preds, threshold=threshold, hop_time=self.hop_time)
            plot_detection(audio.numpy()[0], self.sr, preds, self.hop_time, self.original_sr, events, labels = [i[0] for i in labels], title=f"{event_label[0]} | {file[0]}")"""

    def evaluate_full(
        self, idx_ls, threshold=0.5, plot_detection_viz=False, output_folder=None
    ):
        file_path = DETECTION_TEST_PATH
        events_list = []

        for idx in tqdm(idx_ls, desc=f"Testing on {file_path}..."):
            filename = f"{file_path}/test_scene_0{str(idx).zfill(3)}.wav"
            audio, sr = librosa.load(filename, sr=None)
            embeddings = extract_yamnet_embeddings(audio, sr)
            num_frames = embeddings.shape[0]
            frame_times = np.arange(num_frames) * self.hop_time

            labels = np.zeros(num_frames, dtype=np.float32)
            onset, offset = self.data["onset"][idx], self.data["offset"][idx]
            labels[(frame_times >= onset) & (frame_times <= offset)] = 1

            X = torch.tensor(embeddings, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
            event_label = self.data["event_label"][idx]

            with torch.no_grad():
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                preds = outputs[:, 0].cpu().numpy()

            events = detect_onsets_offsets(
                preds, threshold=threshold, hop_time=self.hop_time
            )

            if plot_detection_viz:
                plot_detection(
                    audio,
                    self.sr,
                    preds,
                    self.hop_time,
                    self.original_sr,
                    events,
                    labels=(onset, offset),
                    title=f"{idx} | {event_label}",
                    output_folder=output_folder,
                )

            # Event detection output
            events_list.append({"filename": filename, "events": events})
        return events_list


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
        epochs=2,
        train_idx=train_idx,
        device=torch.device(device),
        checkpoint_path=checkpoint_path,
        mode="train",
    )
    solver.train()
    return solver


def run_yamnet(
    checkpoint_path: Path = Path("checkpoints") / "yamnet_detector.pth",
    test_path: Path = Path("data") / "processed" / "yamnet" / "spectrograms_test.pkl",
):
    if not os.path.exists(checkpoint_path):
        logger.info("No model checkpoint, training model...")
        trained_solver = train_yamnet()
        logger.info(f"Saved solver to {checkpoint_path}")
    else:
        logger.info(f"Loading from checkpoint {checkpoint_path}...")
        trained_solver = Solver(
            train_idx=[], checkpoint_path=checkpoint_path, mode="infer"
        )
        trained_solver.load_model(checkpoint_path)

    logger.info(f"Loading test data from {test_path}...")
    test_data = pickle.load(open(test_path, "rb"))

    events_list = trained_solver.evaluate_full(
        [i for i in range(len(test_data["event_label"]))],
        plot_detection_viz=False,
        output_folder=None,
    )
    return events_list


if __name__ == "__main__":
    events_list = run_yamnet()
    print(events_list)
