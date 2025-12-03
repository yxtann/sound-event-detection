import os
import soundfile as sf
import shutil
import pickle
import sed_eval

from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.config import NUM_STAGES, DETECTOR_MODEL, CLASSIFIER_MODEL, COMBINED_MODEL, CLASSES, DETECTION_TEST_PATH
from src.models.yamnet_train import run_yamnet
from src.models.audio_mamba_ft import audio_mamba_inference
from src.utils.audio_mamba_metadata_generator import generate_metadata_from_detector
from src.utils.audio_to_spectrograms import create_spectrogram_pkl

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


def calculate_metrics(pred_event_dict, gt_event_dict, time_resolution=1.0, t_collar=0.25):
    logger.warning("Metrics not implemented!")
    # Based on event dictionary

    # Accuracy

    # IoU

    # DCASE SED eval: https://tut-arg.github.io/sed_eval/tutorial.html#id1
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(CLASSES, t_collar=t_collar)
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(CLASSES, time_resolution=time_resolution)
    for file, estimated_event in pred_event_dict.items():
        ref_event = gt_event_dict[file]
        event_based_metrics.evaluate(
            reference_event_list=ref_event,
            estimated_event_list=estimated_event
        )
        segment_based_metrics.evaluate(
            reference_event_list=ref_event,
            estimated_event_list=estimated_event
        )
    print(event_based_metrics)
    print(segment_based_metrics)



def run_pipeline():

    if DETECTOR_MODEL == "yamnet":
        test_path = Path("data") / "processed" / "yamnet" / "spectrograms_test.pkl"
        test_data = pickle.load(open(test_path, "rb"))
        filepaths = [os.path.join(DETECTION_TEST_PATH, file) for file in test_data['files']] # filepath of test wav files
        events_list = run_yamnet(
            filepaths, 
            checkpoint_path="checkpoints/yamnet_detector.pth",
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

    # EVALUATION: can consider moving this to calculate_metrics
    # check if gt pkl file is created
    gt_pkl_path = 'data/processed/yamnet/spectrograms_test_list.pkl'
    if not(os.path.exists(gt_pkl_path)):
        create_spectrogram_pkl()

    # get gt events to use for all models
    gt_events = pickle.load(open(gt_pkl_path, 'rb'))
    gt_event_dict = {ref_event['file']: [{'file':ref_event['file'], 
                        'event_onset':ref_event['onset'], 
                        'event_offset':ref_event['offset'],
                        'event_label':ref_event['event_label']}]
                        for ref_event in gt_events}

    # Run metrics
    calculate_metrics({}, gt_event_dict)


if __name__ == "__main__":
    run_pipeline()
