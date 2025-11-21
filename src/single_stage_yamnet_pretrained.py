import librosa
import csv
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import matplotlib.pyplot as plt

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

labels_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)

with open(labels_path) as f:
    reader = csv.DictReader(f)
    class_names = [row['display_name'] for row in reader]

HOP_TIME = 0.48 # YAMNet requirement

#class_dict_inv = {} # maps audioset class id to our labels
class_dict = {} # maps our labels to audioset classid
for cls, search in zip(['dog_bark','siren','cough','gun_shot','car_horn'], ['bark','siren','cough','gun','horn']):
    #for i, c in zip(range(len(class_names)), class_names):
        #if search in c.lower():
            #assert i not in class_dict_inv, "Overlap of classes"
            #class_dict_inv[i] = cls
    class_dict[cls] = [i for i, c in zip(range(len(class_names)), class_names) if search in c.lower()]
    print({cls: [i for i in class_names if search in i.lower()]}) # sanity check mapping
#print(class_dict_inv)
#print(class_dict)


def detect_events(y, sr, threshold=0.3):

    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000) # YAMNet requirement
        sr = 16000

    scores, _, _ = yamnet_model(y) #embeddings, spectrogram
    scores = scores.numpy() # n_frames x audioset_classes (20 x 521)

    events = []
    all_probs = []
    
    for custom_label, class_ids in class_dict.items():

        # Collect all frames where ANY mapped AudioSet class_id is above threshold
        probs = np.round(scores[:, class_ids].max(axis = 1), 2)
        all_probs.append(probs)
        combined_mask = probs > threshold

        if not np.any(combined_mask):
            continue

        # Split into continuous segments
        idx = np.where(combined_mask)[0]
        splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

        for seg in splits:
            onset_frame = seg[0]
            offset_frame = seg[-1]

            onset_t = onset_frame * HOP_TIME
            offset_t = (offset_frame + 1) * HOP_TIME

            # Extract audio
            start_sample = int(onset_t * sr)
            end_sample = int(offset_t * sr)
            #audio_clip = y[start_sample:end_sample]

            events.append({"event_type": custom_label,"onset": onset_t,"offset": offset_t})

    return events, np.array(all_probs) # n_classes x n_frames

"""
sample output:
[{'event_type': 'siren',
  'onset': np.float64(6.72),
  'offset': np.float64(7.68)},
 {'event_type': 'car_horn',
  'onset': np.float64(5.76),
  'offset': np.float64(6.72)},
  ...]
"""

def plot_detection(audio, sr, preds, hop_time, original_sr = 44100, threshold = 0.3, onsets_offsets=None, labels=None, title=None):
    _, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    S = librosa.feature.melspectrogram(y=audio, sr=original_sr, n_fft=1024, hop_length=256, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=original_sr, hop_length=256, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title(title or "Mel Spectrogram")

    t_pred = np.arange(len(preds)) * hop_time
    axs[1].plot(t_pred, preds, label=class_dict.keys(), marker = '+')
    axs[1].axhline(threshold, color='gray', linestyle='--', label='Threshold')

    if labels is not None:
        onset, offset = labels
        axs[1].axvline(onset, label = 'actual', color = 'black'); axs[1].axvline(offset, color = 'black')
        axs[0].axvline(onset, label = 'actual', color = 'black'); axs[0].axvline(offset, color = 'black')

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].legend()
    plt.tight_layout()
    plt.show()