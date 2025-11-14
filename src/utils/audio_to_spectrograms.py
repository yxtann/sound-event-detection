import librosa
import matplotlib.pyplot as plt
import numpy as np

def get_mel_spec(filename, path, sr, n_fft = 2048, hop_length = 512, n_mels = 64):
    y, sr = librosa.load(path +'/'+ filename, sr = sr)
    fmax = sr // 2
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=fmax, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)  # log scale (dB)
    return S_db

def get_stft_spec(filename, path, sr):
    y, sr = librosa.load(path +'/'+ filename, sr = sr)
    D = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(D, ref=np.max)
    return S_db

def display_spectrogram(S_db, sr, onset = None, offset = None):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    if (onset is not None) & (offset is not None):
        plt.axvline(onset)
        plt.axvline(offset)
    plt.tight_layout()
    plt.show()