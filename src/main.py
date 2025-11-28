import os
import soundfile as sf
import shutil

from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.config import NUM_STAGES, DETECTOR_MODEL, CLASSIFIER_MODEL, COMBINED_MODEL
from src.models.yamnet_train import run_yamnet
from src.models.audio_mamba_ft import audio_mamba_inference
from src.utils.audio_mamba_metadata_generator import generate_metadata_from_detector


def cut_events_from_audio(extracted_audio_path, events_list):

    if not os.path.exists(extracted_audio_path):
        os.makedirs(extracted_audio_path)
        logger.info(f"Created directory: {extracted_audio_path}")
    else:
        # Clean the filepath as every prediction is new
        shutil.rmtree(extracted_audio_path)
        os.makedirs(extracted_audio_path)
        logger.info(f"Cleaned directory: {extracted_audio_path}")

    for event in tqdm(events_list):
        filename = event["filename"]
        events_time_sections = event["events"]

        audio_array, sr = sf.read(filename)
        base_name = Path(filename).stem

        # Cut the wav file and save it to the extracted_audio_path
        for start_sec, end_sec in events_time_sections:
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            # Slice the numpy array
            sliced_audio = audio_array[start_sample:end_sample]

            # Construct new filename
            new_filename = f"{base_name}_{start_sec:.2f}_{end_sec:.2f}.wav"
            output_path = extracted_audio_path / new_filename

            # Write the new file
            sf.write(output_path, sliced_audio, sr)


def calculate_metrics(pred_event_dict, gt_event_dict):
    logger.warning("Metrics not implemented!")
    # Based on event dictionary

    # Accuracy

    # IoU

    pass


def run_pipeline():

    if DETECTOR_MODEL == "yamnet":
        events_list = run_yamnet(
            checkpoint_path=Path("checkpoints") / "yamnet_detector.pth",
            test_path=Path("data") / "processed" / "yamnet" / "spectrograms_test.pkl",
        )
    elif DETECTOR_MODEL == "crnn":
        pass
    elif DETECTOR_MODEL == "htsat":
        pass

    # Use the above detections to cut audio for the classification stage
    # if we are using a 2-stage model
    if NUM_STAGES == 2:
        if CLASSIFIER_MODEL == "yamnet":
            pass
        elif CLASSIFIER_MODEL == "crnn":
            pass
        elif CLASSIFIER_MODEL == "htsat":
            pass
        elif CLASSIFIER_MODEL == "mamba":
            cut_events_from_audio(
                (Path("data") / "processed" / "yamnet" / "extracted_audio"), events_list
            )
            generate_metadata_from_detector(
                groundtruth_csv="data/processed/detection/test/annotations.csv",
                detector_audio_output_path="data/processed/yamnet/extracted_audio/",
            )
            audio_mamba_inference(
                checkpoint_path="checkpoints/audio_mamba_ft.pth",
                val_json_path="data/processed/yamnet/extracted_audio/audio_mamba_metadata.json",
            )

    # Run metrics
    calculate_metrics({}, {})


if __name__ == "__main__":
    run_pipeline()
