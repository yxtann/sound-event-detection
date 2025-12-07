import os
from loguru import logger
from pathlib import Path
from datasets import ClassLabel
import torch
from torch.utils.data import DataLoader
import lightning as L

from src.config import CLASSES
from src.models.htsat.data import HTSATDataset, process_data_for_classification, format_dataset
from src.models.htsat.model import init_model, get_config, run_htsat_train

from src.models.htsat.constants import HTSAT_CHECKPOINT, CLASS_LABELS

def get_classification_events(frame_predictions, events_list, CLASS_LABELS):
    for batch in frame_predictions:

        for i in range(len(batch['file'])):
            filename = batch['file'][i]
            extracted_audio_filename = os.path.basename(batch['file_path'][i])
            pred_map = batch['pred_map'][i][:, :-1]
            pred_map_labels = torch.argmax(pred_map, dim=1)
            
            x_flat = pred_map_labels.flatten().flatten()
            unique_values, counts = torch.unique(x_flat, return_counts=True)
            pred_class_int = unique_values[torch.argmax(counts)]
            pred_class = CLASS_LABELS.int2str(pred_class_int.item())

            events = events_list[filename]
            for j in range(len(events)):
                if events[j]["extracted_audio_filename"] == extracted_audio_filename:
                    events[j]["event_label"] = pred_class
                    break
            events_list[filename] = events

    return events_list

def run_htsat_classification(event_list,
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

    extracted_dataset = process_data_for_classification(event_list)
    formatted_extracted_dataset = format_dataset(extracted_dataset, CLASS_LABELS)
    htsat_extracted_dataset = HTSATDataset(formatted_extracted_dataset, config, eval_mode=True)

    extracted_test_loader = DataLoader(
        dataset = htsat_extracted_dataset,
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
        dataloaders=extracted_test_loader
    )

    frame_predictions = model.test_step_outputs

    classified_events_list = get_classification_events(frame_predictions, event_list, CLASS_LABELS)

    return classified_events_list