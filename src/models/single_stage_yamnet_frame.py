import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import librosa
import os

YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
TARGET_SR = 16000  # YAMNet requires 16kHz


def load_audio_mono(path, sr=TARGET_SR):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)

def labels_to_frames(onsets_offsets, n_frames, audio_length = 10):
    """
    Convert onset-offset labels to frame-level integer labels
    onsets_offsets: list of (onset_sec, offset_sec, class_idx) OR np.array with shape (n_labels,3)
    """
    frame_labels = np.zeros(n_frames, dtype=np.int32)  # default background=0
    frame_hop = audio_length / n_frames
    onsets_offsets = np.atleast_2d(onsets_offsets) # Ensure it's 2D array

    for row in onsets_offsets:
        onset, offset, class_idx = row
        start_frame = int(np.floor(onset / frame_hop))
        end_frame = int(np.ceil(offset / frame_hop))
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        frame_labels[start_frame:end_frame] = int(class_idx)
    
    return frame_labels

def generator(filepaths, annotations, shuffle=True):
    idxs = list(range(len(filepaths)))
    if shuffle:
        random.shuffle(idxs)
    yamnet_model = YAMNET
    for i in idxs:
        path = filepaths[i]
        annots = annotations[i]  # list of (onset, offset, class_idx)
        wav = load_audio_mono(path)
        # get embeddings to determine n_frames
        _, embeddings, _ = yamnet_model(wav)
        n_frames = embeddings.shape[0]
        frame_labels = labels_to_frames(annots, n_frames, audio_length=len(wav)/TARGET_SR)
        yield wav, frame_labels

class YamNetLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.yamnet_model = YAMNET

    def call(self, inputs):
        def run_yamnet(waveform_1d):
            _, embeddings, _ = self.yamnet_model(waveform_1d)
            return embeddings
        embeddings = tf.map_fn(
            fn=run_yamnet,
            elems=inputs,
            fn_output_signature=tf.TensorSpec(shape=(None, 1024), dtype=tf.float32)
        )
        return embeddings

class Trainer:
    def __init__(self, train_files, train_annotations, val_files, val_annotations, **kwargs):
        self.lr = kwargs.pop("lr", 1e-4)
        self.epochs = kwargs.pop("epochs", 30)
        self.batch_size = kwargs.pop("batch_size", 16)
        self.model_save_dir = kwargs.pop("model_save_dir", "../src/models/yamnet.keras")
        self.classes = kwargs.pop("classes")
        self.n_classes = len(self.classes) + 1 # background noise

        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.callbacks = [
            keras.callbacks.ModelCheckpoint("../src/models/yamnet_checkpoint_frame.keras", save_best_only=True, monitor="val_loss"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        ]

        output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_files, train_annotations, shuffle=True),
            output_signature=output_signature
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_files, val_annotations, shuffle=False),
            output_signature=output_signature
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        self.train_ds = train_ds
        self.val_ds = val_ds

    def build_model(self, num_classes):
        waveform_input = keras.Input(shape=(None,), dtype=tf.float32)
        embeddings = YamNetLayer()(waveform_input)  # (batch, frames, 1024)
        x = keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu'))(embeddings)
        x = keras.layers.Dropout(0.3)(x)
        output = keras.layers.TimeDistributed(keras.layers.Dense(num_classes, activation='softmax'))(x)
        self.model = keras.Model(inputs=waveform_input, outputs=output)

    def train(self):
        self.build_model(self.n_classes)
        self.model.summary()
        self.model.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy', #loss_fn
            metrics=['accuracy']
        )
        self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=self.epochs, callbacks=self.callbacks)
        self.model.save(self.model_save_dir, include_optimizer=False)
        print("Saved model to", self.model_save_dir)

    def load_model(self, model_path=None):
        """
        Load a previously saved model into self.model
        """
        if model_path is None:
            model_path=self.model_save_dir
        self.model = keras.models.load_model(model_path, compile=False, custom_objects={"YamNetLayer": YamNetLayer})
        print(f"Loaded model from {model_path}")

    def predict_frames(self, file):
        """
        Returns probability of each class at each frame (frames, num_classes)
        """
        waveform_np = load_audio_mono(file, sr=TARGET_SR)
        waveform_tf = tf.convert_to_tensor(waveform_np.astype(np.float32))
        waveform_tf = tf.expand_dims(waveform_tf, axis=0)
        frame_probs = self.model(waveform_tf, training=False).numpy()[0]
        return frame_probs 
    
    def predict_class(self, file):
        """
        Returns predicted class at each frame (frames, )
        """
        frame_probs = self.predict_frames(file)
        frame_classes = np.argmax(frame_probs, axis=-1)
        return frame_classes
    
    def predict_events(self, file, audio_length=10, min_duration=0.1):
        frame_classes = self.predict_class(file)
        n_frames = frame_classes.shape[0]
        frame_hop_sec = audio_length / n_frames
        filename = os.path.basename(file)
        events = []
        current_class = frame_classes[0]
        start_frame = 0
        for i in range(1, n_frames):
            if frame_classes[i] != current_class:
                if current_class != 0:
                    onset = start_frame * frame_hop_sec
                    offset = i * frame_hop_sec
                    if offset - onset >= min_duration:
                        events.append({'file': filename, 'event_onset': onset, 'event_offset': offset, 'event_label':self.classes[int(current_class)-1]})
                current_class = frame_classes[i]
                start_frame = i

        if current_class != 0:
            onset = start_frame * frame_hop_sec
            offset = n_frames * frame_hop_sec
            if offset - onset >= min_duration:
                events.append({'file': filename, 'event_onset': onset, 'event_offset': offset, 'event_label':self.classes[int(current_class)-1]})

        return events