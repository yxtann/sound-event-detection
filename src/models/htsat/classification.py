import os
from loguru import logger
from pathlib import Path
from datasets import ClassLabel
import torch
from torch.utils.data import DataLoader
import lightning as L

from src.config import CLASSES
from src.models.htsat.model import HTSATModel
from src.models.htsat.data import HTSATDataset, process_data_for_classification, format_dataset

# Constants
HTSAT_CHECKPOINT = Path("checkpoints") / "htsat_detector.pth"
NON_EVENT_LABEL = 'non_event'
CLASS_LABELS = ClassLabel(names=CLASSES + [NON_EVENT_LABEL])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(config, test_only = False):
    logger.info("Intializing HTS-AT Model")
    from external.hts_audio_transformer.model.htsat import HTSAT_Swin_Transformer

    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        num_classes=config.classes_num,
        config = config
    )

    sed_model.to(DEVICE)

    if os.path.exists(HTSAT_CHECKPOINT):
        logger.info(f"HTS-AT Model checkpoint exists at {HTSAT_CHECKPOINT}, Loading weights...")
        state_dict = torch.load(HTSAT_CHECKPOINT, map_location=DEVICE)
        sed_model.load_state_dict(state_dict)

    if test_only:
        sed_model.eval()

    model = HTSATModel(sed_model, config)

    return model

def get_config():
    from external.hts_audio_transformer import config

    ## Add / Modify Configurations
    config.debug = True
    config.max_epoch = 10
    config.classes_num = len(CLASSES) + 1
    config.sample_rate = 44100

    config.clip_duration = 20.0
    config.classes = CLASS_LABELS
    config.clip_samples = config.sample_rate * config.clip_duration
    config.htsat_spec_size = 512

    return config

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
    if not os.path.exists(checkpoint_path):
        # TODO
        logger.info("HTS-AT model was trained separately.")
        # logger.info("No model checkpoint, training model...")
        # trained_solver = train_yamnet(checkpoint_path=checkpoint_path)
        # logger.info(f"Saved solver to {checkpoint_path}")
    else:
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