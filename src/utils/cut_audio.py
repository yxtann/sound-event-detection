import os
import soundfile as sf
import shutil

from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.config import DETECTION_TEST_PATH


def cut_events_from_audio(extracted_audio_path, events_list, data_path=DETECTION_TEST_PATH):

    if not os.path.exists(extracted_audio_path):
        os.makedirs(extracted_audio_path)
        logger.info(f"Created directory: {extracted_audio_path}")
    else:
        # Clean the filepath as every prediction is new
        shutil.rmtree(extracted_audio_path)
        os.makedirs(extracted_audio_path)
        logger.info(f"Cleaned directory: {extracted_audio_path}")

    for filename, events in tqdm(events_list.items()):
        filepath = Path(data_path, filename)
        for i in range(len(events)):
            event = events[i]

            start_sec = event["event_onset"]
            end_sec = event["event_offset"]

            audio_array, sr = sf.read(filepath)
            base_name = Path(filepath).stem

            # Cut the wav file and save it to the extracted_audio_path
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            # Slice the numpy array
            sliced_audio = audio_array[start_sample:end_sample]

            # Construct new filename
            new_filename = f"{base_name}_{start_sec:.2f}_{end_sec:.2f}.wav"
            output_path = extracted_audio_path / new_filename

            # Write the new file
            sf.write(output_path, sliced_audio, sr)

            # Record
            events[i] = events[i] | {"extracted_audio_filename": new_filename}

        events_list[filename] = events

    return events_list