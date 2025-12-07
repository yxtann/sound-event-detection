import os
from loguru import logger
from pathlib import Path
from datasets import ClassLabel
import torch
from torch.utils.data import DataLoader
import lightning as L
from scipy.ndimage import gaussian_filter1d

from src.config import CLASSES
from src.models.htsat.data import HTSATDataset, format_dataset, process_data_for_combined
from src.models.htsat.model import init_model, get_config, run_htsat_train

from src.models.htsat.constants import HTSAT_CHECKPOINT, NON_EVENT_LABEL, CLASS_LABELS


def get_events_from_batch(batch, config):
    NON_EVENT_LABEL_INT = CLASS_LABELS.str2int(NON_EVENT_LABEL)

    file = batch['file']
    pred_map = batch['pred_map']
    batch_size, num_frame, num_classes = pred_map.shape


    # pred_map_median = median_filter(pred_map, size=(1, 7, 1)) # smoothing between frames
    pred_map_gaussian_np = gaussian_filter1d(
        input=pred_map, 
        sigma=3,
        axis=1,             # Apply the filter along the time axis (num_frame)
        mode='nearest'      # How to handle the boundaries of the frames
    )
    pred_map = torch.as_tensor(pred_map_gaussian_np)
    pred_map_labels = torch.argmax(pred_map, dim=2)

    events = {}

    for batch_idx in range(batch_size):
        pred_map_record = pred_map_labels[batch_idx]
        file_record = file[batch_idx]
        onset_frame = 0
        file_events = []
        current_label = pred_map_record[0]
        for i, pred_label in enumerate(pred_map_record, start = 1):
            if pred_label != current_label:
                if current_label != NON_EVENT_LABEL_INT:
                    onset_time = onset_frame / num_frame * config.clip_duration
                    offset_time = i / num_frame * config.clip_duration
                    file_events.append(
                        {
                            'file': file_record,
                            'event_onset': onset_time,
                            'event_offset': offset_time,
                            'event_label': CLASS_LABELS.int2str(current_label.item())
                        }
                    )
                    
                onset_frame = i
                current_label = pred_label

        if current_label != NON_EVENT_LABEL_INT:
            onset_time = onset_frame / num_frame * config.clip_duration
            offset_time = i / num_frame * config.clip_duration
            file_events.append(
                {
                    'file': file_record,
                    'event_onset': onset_time,
                    'event_offset': offset_time,
                    'event_label': CLASS_LABELS.int2str(current_label.item())
                }
            )
        events[file_record] = file_events

        return events
    
def get_events(frame_predictions, config):
    pred_event_dict = {}
    for batch in frame_predictions:
        batch_events = get_events_from_batch(batch, config)
        pred_event_dict = pred_event_dict | batch_events
    return pred_event_dict

def run_htsat_combined(
    checkpoint_path = HTSAT_CHECKPOINT,
):
    # If Model Checkpoints not available, train model
    if not os.path.exists(checkpoint_path):
        # TODO
        logger.info(f"HTS-AT model checkpoints not detected at {checkpoint_path}, Training Model")
        run_htsat_train(checkpoint_path = checkpoint_path)

    # Model Checkpoints are available now, run test
    config = get_config()
    model = init_model(config, True)


    test_dataset = process_data_for_combined()
    formatted_test_dataset = format_dataset(test_dataset, CLASS_LABELS)
    htsat_test_dataset = HTSATDataset(formatted_test_dataset, config, eval_mode=True)

    test_loader = DataLoader(
        dataset = htsat_test_dataset,
        num_workers = config.num_workers,
        batch_size = 1,
        shuffle = False,
    )

    minimal_trainer = L.Trainer(
        max_epochs=config.max_epoch,
        default_root_dir="./lightning_checkpoints",
    )

    minimal_trainer.test(
        model, 
        dataloaders=test_loader
    )

    frame_predictions = model.test_step_outputs
    events_list = get_events(frame_predictions, config)

    return events_list
