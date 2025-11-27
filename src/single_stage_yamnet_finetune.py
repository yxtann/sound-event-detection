import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import librosa
from pathlib import Path
import sys
import pickle

UNFREEZE_YAMNET = True      # set True to fine-tune YamNet weights (careful: small LR)
TARGET_SR = 16000            # YAMNet requires 16kHz       

# For sliding-window inference (higher temporal resolution)
SLIDING_WIN_SEC = 1.0   # window length for inference
SLIDING_HOP_SEC = 0.1   # hop between windows -> effective temporal resolution

# -----------------------------
# Build tf.data Dataset using a python generator
# (we yield raw waveforms as float32 1D numpy arrays and labels)
# -----------------------------

def load_audio_mono(path, sr=TARGET_SR):
    """
    Helper: load audio (librosa ensures 16k resampling)
    """
    wav, orig_sr = librosa.load(path, sr=sr, mono=True)
    # normalize to -1..1 if not already
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    return wav

def generator(filepaths, labels, shuffle=True):
    idxs = list(range(len(filepaths)))
    if shuffle:
        random.shuffle(idxs)
    for i in idxs:
        path = filepaths[i]
        label = labels[i]
        wav = load_audio_mono(path, sr=TARGET_SR)
        # convert to float32 numpy array
        wav = wav.astype(np.float32)
        yield wav, np.int32(label)

class YamNetLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    def call(self, inputs):
        # inputs: (batch, samples)

        def run_yamnet(waveform_1d):
            # waveform_1d: (samples,)
            _, embeddings, _ = self.yamnet_model(waveform_1d)
            return embeddings   # (frames, 1024)

        # Apply to each batch element
        embeddings = tf.map_fn(
            fn=run_yamnet,
            elems=inputs,
            fn_output_signature=tf.TensorSpec(shape=(None, 1024), dtype=tf.float32)
        )
        return embeddings    # (batch, frames, 1024)
    
# -----------------------------
# Build classifier model:
# Input: raw waveform
# YAMNet returns embeddings per frame -> we pool across frames (mean) to obtain clip embedding
# Then Dense head for NUM_CLASSES
# -----------------------------

class Trainer(object):
    def __init__(self, train_files, train_labels, val_files, val_labels, **kwargs):
        self.lr = kwargs.pop("lr", 1e-4)
        self.epochs = kwargs.pop("epochs", 30)
        self.device = kwargs.pop("device", "cpu")
        self.batch_size = kwargs.pop("batch_size", 16)
        self.model_save_dir = kwargs.pop("model_save_dir", "../src/models/yamnet_finetune_model.keras")

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint("../src/models/yamnet_finetune_checkpoint.keras", monitor="val_accuracy", save_best_only=True, save_weights_only=False),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        ]
        self.n_classes = kwargs.pop("n_classes", 5)

        # Define tf.data shapes: waveform is variable length -> use RaggedTensor through tf.data by using from_generator
        output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),  # waveform
            tf.TensorSpec(shape=(), dtype=tf.int32)         # label
        )

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_files, train_labels, shuffle=True),
            output_signature=output_signature
        )

        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_files, val_labels, shuffle=False),
            output_signature=output_signature
        )

        # Batch: since waveforms have variable length, we create batches of size 1 and then use map to preprocess in-model.
        # For performance you can pad to fixed length, or use bucketing - omitted here for simplicity.
        train_ds = train_ds.batch(self.batch_size)
        val_ds = val_ds.batch(self.batch_size)

        self.train_ds = train_ds
        self.val_ds = val_ds

    def build_model(self, num_classes):#, yamnet_layer, unfreeze_yamnet=False):
        waveform_input = keras.Input(shape=(None,), dtype=tf.float32)
        embedding = YamNetLayer()(waveform_input) # Wrap in custom layer
        pooled = keras.layers.GlobalAveragePooling1D()(embedding) # Optional: temporal pooling (mean over frames)

        # Classification head
        x = keras.layers.Dense(256, activation="relu")(pooled)
        x = keras.layers.Dropout(0.3)(x)
        output = keras.layers.Dense(num_classes, activation="softmax")(x)
        self.model = keras.Model(inputs=waveform_input, outputs=output)
    
    def train(self):
        self.build_model(self.n_classes)
        self.model.summary()
        self.model.compile(
            optimizer=self.optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
            )

        print("Starting training...")
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=self.callbacks
        )

        self.model.save(self.model_save_dir, include_optimizer=False)
        print("Saved model to", self.model_save_dir)

    def predict_clip(self, waveform_np):
        """
        Clip level predictions
        waveform_np: 1D numpy array at 16kHz
        returns: (pred_class_idx, pred_prob, prob_vector)
        """
        waveform_tf = tf.convert_to_tensor(waveform_np.astype(np.float32))
        waveform_tf = tf.expand_dims(waveform_tf, axis=0)  # batch
        probs = self.model(waveform_tf, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), probs

    def sliding_window_inference(self, waveform_np, win_sec=SLIDING_WIN_SEC, hop_sec=SLIDING_HOP_SEC):
        """
        Sliding-window inference for better time resolution
        """
        sr = TARGET_SR
        win_samples = int(win_sec * sr)
        hop_samples = int(hop_sec * sr)
        n = len(waveform_np)
        times = []
        probs = []
        for start in range(0, max(1, n - win_samples + 1), hop_samples):
            seg = waveform_np[start:start + win_samples]
            idx, prob, vec = self.predict_clip(seg)
            t_center = (start + win_samples/2) / sr
            times.append(t_center)
            probs.append(vec)
        if len(probs) == 0:
            # handle short audio by padding window
            idx, prob, vec = self.predict_clip(waveform_np)
            times = [len(waveform_np)/(2*sr)]
            probs = [vec]
        probs = np.vstack(probs)   # shape (n_windows, num_classes)
        times = np.array(times)
        return times, probs