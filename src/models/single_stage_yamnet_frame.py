import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import librosa
import os
from tqdm import tqdm
import pickle
import json
from scipy.ndimage import median_filter

YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
YAMNET_HOP_SEC = 0.48
TARGET_SR = 16000  # YAMNet requires 16kHz
AUDIO_LENGTH = 20 # assume fixed
N_FRAMES = int(np.ceil((AUDIO_LENGTH - YAMNET_HOP_SEC*2)/YAMNET_HOP_SEC + 1)) # window size is 2*hop
PRED_YAMNET_PATH = r"outputs/single_stage_yamnet.json"
YAMNET_CACHE_DIR = r"data/cache/single_stage_yamnet_embeddings"

def load_audio_mono(path, sr=TARGET_SR):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)

def labels_to_frames(onsets_offsets, n_frames=N_FRAMES, frame_hop = YAMNET_HOP_SEC):
    """
    Convert onset-offset labels to frame-level integer labels
    onsets_offsets: list of (onset_sec, offset_sec, class_idx) OR np.array with shape (n_labels,3)
    """
    frame_labels = np.zeros(n_frames, dtype=np.int32)  # default background=0
    onsets_offsets = np.atleast_2d(onsets_offsets) # Ensure it's 2D array, in case of multi events per file

    for row in onsets_offsets:
        onset, offset, class_idx = row
        start_frame = int(np.floor(onset / frame_hop))
        end_frame = int(np.ceil(offset / frame_hop))
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        frame_labels[start_frame:end_frame] = int(class_idx)
    
    return frame_labels

def get_embedding_path(audio_path):
    """Generates a cache filename based on the original audio filename."""
    filename = os.path.basename(audio_path)
    name_only = os.path.splitext(filename)[0] # Change extension to .npy
    return os.path.join(YAMNET_CACHE_DIR, name_only + ".npy")

def precompute_embeddings(filepaths):
    """
    Runs YAMNet on the provided files and saves embeddings to disk.
    This runs ONCE before training starts.
    """
    os.makedirs(YAMNET_CACHE_DIR, exist_ok=True)
    print(f"Pre-computing embeddings for {len(filepaths)} files into {YAMNET_CACHE_DIR}...")
    
    for path in tqdm(filepaths):
        npy_path = get_embedding_path(path)
        if not os.path.exists(npy_path):
            wav = load_audio_mono(path)
            _, embeddings, _ = YAMNET(wav)
            np.save(npy_path, embeddings.numpy())

def generator(filepaths, annotations, shuffle=True):
    idxs = list(range(len(filepaths)))
    if shuffle:
        random.shuffle(idxs)
    for i in idxs:
        npy_path = get_embedding_path(filepaths[i])
        embeddings = np.load(npy_path)
        annots = annotations[i]  # list of (onset, offset, class_idx)
        frame_labels = labels_to_frames(annots, N_FRAMES)
        yield embeddings, frame_labels

class Trainer:
    def __init__(self, train_files, train_annotations, val_files, val_annotations, **kwargs):
        self.lr = kwargs.pop("lr", 1e-4)
        self.epochs = kwargs.pop("epochs", 30)
        self.batch_size = kwargs.pop("batch_size", 16)
        self.model_save_dir = kwargs.pop("model_save_dir", "checkpoints/yamnet.keras")
        self.classes = kwargs.pop("classes")
        self.n_classes = len(self.classes) + 1 # class for no events

        self.train_steps = len(train_files) // self.batch_size
        self.val_steps = len(val_files) // self.batch_size

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.callbacks = [
            keras.callbacks.ModelCheckpoint("checkpoints/yamnet_checkpoint_frame.keras", save_best_only=True, monitor="val_loss"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        ]
        
        all_files = train_files + val_files
        precompute_embeddings(all_files)

        output_signature = (
            tf.TensorSpec(shape=(None,1024), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_files, train_annotations, shuffle=True),
            output_signature=output_signature
        ).batch(self.batch_size, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_files, val_annotations, shuffle=False),
            output_signature=output_signature
        ).batch(self.batch_size, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE)

        self.train_ds = train_ds
        self.val_ds = val_ds

    def build_model(self):
        embeddings = keras.Input(shape=(None, 1024), dtype=tf.float32)
        x = keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu'))(embeddings)
        x = keras.layers.Dropout(0.3)(x)
        output = keras.layers.TimeDistributed(keras.layers.Dense(self.n_classes, activation='softmax'))(x)
        self.model = keras.Model(inputs=embeddings, outputs=output)

    def train(self):
        self.build_model()
        self.model.summary()
        self.model.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.fit(
            self.train_ds, 
            validation_data=self.val_ds, 
            epochs=self.epochs, 
            callbacks=self.callbacks, 
            steps_per_epoch=self.train_steps, 
            validation_steps=self.val_steps
        )
        self.model.save(self.model_save_dir, include_optimizer=False)
        print("Saved model to", self.model_save_dir)

    def load_model(self, model_path=None):
        """
        Load a previously saved model into self.model
        """
        if model_path is None:
            model_path=self.model_save_dir
        self.model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded model from {model_path}")

    def predict_frames(self, file, med_filter=False):
        """
        Returns probability of each class at each frame (frames, num_classes)
        """
        npy_path = get_embedding_path(file)
        if not os.path.exists(npy_path):
            waveform_np = load_audio_mono(file, sr=TARGET_SR)
            _, embeddings, _ = YAMNET(waveform_np)
        else:
            embeddings = np.load(npy_path)
        embeddings_batch = tf.expand_dims(embeddings, axis=0)
        frame_probs = self.model(embeddings_batch, training=False).numpy()[0]
        if med_filter:
            frame_probs = median_filter(frame_probs, size=(3, 1)) # smoothing between frames
        return frame_probs 
    
    def predict_class(self, file, med_filter=False):
        """
        Returns predicted class at each frame (frames, )
        """
        frame_probs = self.predict_frames(file, med_filter=med_filter)
        frame_classes = np.argmax(frame_probs, axis=-1)
        return frame_classes
    
    def predict_events(self, file, frame_hop_sec = YAMNET_HOP_SEC, med_filter=False):
        """
        Returns events in correct output format
        """
        frame_classes = self.predict_class(file, med_filter=med_filter)
        n_frames = frame_classes.shape[0]
        filename = os.path.basename(file)
        events = []
        current_class = frame_classes[0]
        start_frame = 0
        for i in range(1, n_frames):
            if frame_classes[i] != current_class:
                if current_class != 0:  # event end detected
                    onset = start_frame * frame_hop_sec # start of frame i
                    offset = i * frame_hop_sec # start of frame i (half of frame i-1 since window=2*hop_sec)
                    events.append({'file': filename, 'event_onset': onset, 'event_offset': offset, 'event_label':self.classes[int(current_class)-1]})
                # event start or end detected
                current_class = frame_classes[i] 
                start_frame = i

        if current_class != 0: # last frame, in case event lasts till the end
            onset = start_frame * frame_hop_sec
            offset = n_frames * frame_hop_sec
            events.append({'file': filename, 'event_onset': onset, 'event_offset': offset, 'event_label':self.classes[int(current_class)-1]})
        return events
    
    def inference(self, test_files, med_filter=False):
        """
        Saves test files inference events in correct format to pickle
        test_files: list of file path for test wav files
        """
        estimated_event_outputs = {}
        for i in tqdm(range(len(test_files))):
            test_path = test_files[i]
            estimated_event = self.predict_events(test_path, med_filter=med_filter)
            estimated_event_outputs[os.path.basename(test_path)] = estimated_event
        os.makedirs("outputs", exist_ok=True)
        with open(PRED_YAMNET_PATH, "w") as f:
            json.dump(estimated_event_outputs, f, indent=4)
        print(f"Saved YAMNet predicted events pickle file to {PRED_YAMNET_PATH}")