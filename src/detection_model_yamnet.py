import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_yamnet_embeddings(audio, sr=16000):
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    _, embeddings, _ = yamnet_model(audio)
    return embeddings.numpy()

def detect_onsets_offsets(preds, threshold=0.5, hop_time=0.48):
    """Convert framewise probabilities into onset/offset pairs"""
    events = preds > threshold
    changes = np.diff(events.astype(int))
    onsets = (np.where(changes == 1)[0] + 1) * hop_time # first frame with pred > threshold
    offsets = (np.where(changes == -1)[0] + 2) * hop_time # until end of frame (frame width = 2*hop_time)
    if len(offsets) < len(onsets):
        offsets = np.append(offsets, (len(preds)+1) * hop_time)
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
        self.fc = nn.Linear(hidden_dim*2, 1)
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
        self.device = kwargs.pop("device", 'cpu')

        data = pickle.load(open(os.path.join(self.path_prefix, 'data/processed/detection.p') , 'rb'))
        self.original_sr = data['sr']
        self.data = data

        assert max(train_idx) < len(data['event_label'])
        assert min(train_idx) >= 0

        file_path = os.path.join(self.path_prefix, 'data/detection')
        train_files = [f"{file_path}/scene_0{str(i).zfill(3)}.wav" for i in train_idx]
        train_dataset = YAMNetSEDDataset(train_files, data['onset'][train_idx], data['offset'][train_idx], data['event_label'][train_idx])
        self.trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        test_idx = [i for i in range(len(data['event_label'])) if i not in train_idx]
        test_files = [f"{file_path}/scene_0{str(i).zfill(3)}.wav" for i in test_idx]
        test_dataset = YAMNetSEDDataset(test_files, data['onset'][test_idx], data['offset'][test_idx], data['event_label'][test_idx])
        self.testloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        self.model = YAMNetGRU().to(self.device)
        self.criterion = nn.BCELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

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
        return total_loss/len(self.trainloader)

    def _val_step(self):
        total_loss = 0
        for X, y in self.testloader:
            with torch.no_grad():
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
        return total_loss/len(self.testloader)
    
    def train(self):
        for epoch in range(self.epochs):
            train_loss = self._train_step()
            print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}")
            val_loss = self._val_step()
            print(f"Epoch {epoch+1}: valildation loss = {val_loss:.4f}")

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

    def evaluate_full(self, idx_ls, threshold=0.5):
        file_path = os.path.join(self.path_prefix, 'data/detection')

        for idx in tqdm(idx_ls):
            file = f"{file_path}/scene_0{str(idx).zfill(3)}.wav"
            audio, sr = librosa.load(file, sr=None)
            embeddings = extract_yamnet_embeddings(audio, sr)
            num_frames = embeddings.shape[0]
            frame_times = np.arange(num_frames) * self.hop_time

            labels = np.zeros(num_frames, dtype=np.float32)
            onset, offset = self.data['onset'][idx], self.data['offset'][idx]
            labels[(frame_times >= onset) & (frame_times <= offset)] = 1

            X = torch.tensor(embeddings, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
            event_label = self.data['event_label'][idx]
        
            with torch.no_grad():
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y).item()
                preds = outputs[:, 0].numpy()

            events = detect_onsets_offsets(preds, threshold=threshold, hop_time=self.hop_time)
            plot_detection(audio, self.sr, preds, self.hop_time, self.original_sr, events, labels = (onset, offset), title=f"{idx} | {event_label}")
            print(f'Loss: {loss}')


def plot_detection(audio, sr, preds, hop_time, original_sr = 44100, onsets_offsets=None, labels=None, title=None):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    S = librosa.feature.melspectrogram(y=audio, sr=original_sr, n_fft=1024, hop_length=256, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=original_sr, hop_length=256, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title(title or "Mel Spectrogram")

    for onset, offset in onsets_offsets:
        #axs[0].axvspan(onset, offset, color='orange', alpha=0.3, label='Detected Event')
        axs[0].axvline(onset, label = 'pred', color = 'green'); axs[0].axvline(offset, color = 'green')
    t_pred = np.arange(len(preds)) * hop_time
    axs[1].plot(t_pred, preds, label='Predicted prob', color='blue', marker = '+')
    axs[1].axhline(0.5, color='gray', linestyle='--', label='Threshold')

    if labels is not None:
        onset, offset = labels
        axs[1].axvline(onset, label = 'actual', color = 'red'); axs[1].axvline(offset, color = 'red')
        axs[0].axvline(onset, label = 'actual', color = 'red'); axs[0].axvline(offset, color = 'red')

    for onset, offset in onsets_offsets:
        axs[1].axvline(onset, label = 'pred', color = 'green'); axs[1].axvline(offset, color = 'green')

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].legend()
    plt.tight_layout()
    plt.show()