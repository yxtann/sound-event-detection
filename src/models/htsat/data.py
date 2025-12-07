import librosa
from datasets import Dataset, ClassLabel
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
import logging
import random
import pickle
from pathlib import Path


def process_data_for_classification(events_list):

    from src.config import YAMNET_EXTRACTED_AUDIO_PATH

    test_path = Path("data") / "processed" / "yamnet" / "spectrograms_test.pkl"
    test_data = pickle.load(open(test_path, "rb"))

    extracted_audio_info_df = []

    for test_data_i in range(len(test_data['files'])):
        test_filename = test_data['files'][test_data_i]

        detected_events = events_list.get(test_filename, [])

        for detected_event in detected_events:
            extracted_audio_info_df.append(
                {
                    "file": test_filename,
                    "file_path": f'{YAMNET_EXTRACTED_AUDIO_PATH}/{detected_event["extracted_audio_filename"]}',
                    "S_db": test_data["S_db"][test_data_i],
                    "onset": test_data["onset"][test_data_i],
                    "offset": test_data["offset"][test_data_i],
                    "event_label": test_data['event_label'][test_data_i],
                    "background_label": test_data['background_label'][test_data_i],
                    "sr": test_data['sr']
                }
            )
    extracted_dataset = Dataset.from_list(extracted_audio_info_df)

    return extracted_dataset


def process_data_for_combined():

    from src.config import DETECTION_TEST_PATH

    test_path = Path("data") / "processed" / "yamnet" / "spectrograms_test.pkl"
    test_data = pickle.load(open(test_path, "rb"))

    num_records = len(test_data["files"])
    test_info_df = []
    for i in range(num_records):
        test_info_df.append(
            {
                "file": test_data["files"][i],
                "file_path": f'{DETECTION_TEST_PATH}/{test_data["files"][i]}',
                "S_db": test_data["S_db"][i],
                "onset": test_data["onset"][i],
                "offset": test_data["offset"][i],
                "event_label": test_data["event_label"][i],
                "background_label": test_data["background_label"][i],
                "sr": test_data["sr"]

            }
        )

    test_dataset = Dataset.from_list(test_info_df)

    return test_dataset

def process_data_for_train():

    from src.config import DETECTION_TRAIN_PATH

    train_path = Path("data") / "processed" / "yamnet" / "spectrograms_train.pkl"
    train_data = pickle.load(open(train_path, "rb"))

    num_records = len(train_data["files"])
    train_info_df = []
    for i in range(num_records):
        train_info_df.append(
            {
                "file": train_data["files"][i],
                "file_path": f'{DETECTION_TRAIN_PATH}/{train_data["files"][i]}',
                "S_db": train_data["S_db"][i],
                "onset": train_data["onset"][i],
                "offset": train_data["offset"][i],
                "event_label": train_data["event_label"][i],
                "background_label": train_data["background_label"][i],
                "sr": train_data["sr"]

            }
        )
    test_dataset = Dataset.from_list(train_info_df)

    return test_dataset


def process_train_data_for_classification(events_list):

    from src.config import GROUND_TRUTH_EXTRACTED_AUDIO_PATH

    train_path = Path("data") / "processed" / "yamnet" / "spectrograms_train.pkl"
    train_data = pickle.load(open(train_path, "rb"))

    extracted_audio_info_df = []

    for train_data_i in range(len(train_data['files'])):
        train_filename = train_data['files'][train_data_i]

        detected_events = events_list.get(train_filename, [])

        for detected_event in detected_events:
            extracted_audio_info_df.append(
                {
                    "file": train_filename,
                    "file_path": f'{GROUND_TRUTH_EXTRACTED_AUDIO_PATH}/{detected_event["extracted_audio_filename"]}',
                    "S_db": train_data["S_db"][train_data_i],
                    "onset": train_data["onset"][train_data_i],
                    "offset": train_data["offset"][train_data_i],
                    "event_label": train_data['event_label'][train_data_i],
                    "background_label": train_data['background_label'][train_data_i],
                    "sr": train_data['sr']
                }
            )
    extracted_dataset = Dataset.from_list(extracted_audio_info_df)

    return extracted_dataset


def format_dataset(dataset: Dataset, CLASS_LABELS: ClassLabel):

    def format_function(record, idx):
        audio_filename= record["file"]

        audio_path = record['file_path']
        audio, sr = librosa.load(audio_path, sr=record['sr'])
        target = CLASS_LABELS.str2int(record['event_label'])

        return record | {
            "audio_name": audio_filename,
            "target": target,
            "waveform": audio,
            "real_len": len(audio)
        }

    formatted_dataset = dataset.map(
        format_function,
        with_indices=True,
    ).with_format("torch")

    return formatted_dataset


class HTSATDataset(TorchDataset):
    def __init__(self, dataset, config, eval_mode = False):
        self.dataset = dataset
        self.config = config       
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        logging.info("total dataset size: %d" %(self.total_size))
        if not eval_mode:
            self.generate_queue()

    def generate_queue(self):
        random.shuffle(self.queue)
        logging.info("queue regenerated:%s" %(self.queue[-5:]))


    def __getitem__(self, index):
        # Get actual index of the record
        p = self.queue[index]

        # Get actual item
        record = self.dataset[p]

        # one-hot target if required
        if self.config.loss_type == 'clip_bce':
            label_tensor = torch.as_tensor(record["target"], dtype=torch.long)
            target = F.one_hot(label_tensor, num_classes=self.config.classes_num).float()
        else:
            target = torch.as_tensor(record["target"])

        data_dict = record | {
            "target": target
        }
        logging.info(f"getitem: {p}")
        return data_dict

    def __len__(self):
        return self.total_size