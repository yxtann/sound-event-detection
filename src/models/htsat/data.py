import librosa
from datasets import Dataset, ClassLabel
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
import logging
import random

def process_data_for_classification(events_list):
    import pickle
    from pathlib import Path

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