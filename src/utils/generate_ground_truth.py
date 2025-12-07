import os
import pickle

from src.utils.audio_to_spectrograms import create_spectrogram_pkl

def generate_gt_events_dict(gt_pkl_path = 'data/processed/yamnet/spectrograms_test_list.pkl'):
    # check if gt pkl file is created
    if not(os.path.exists(gt_pkl_path)):
        create_spectrogram_pkl()

    # get gt events to use for all models
    gt_events = pickle.load(open(gt_pkl_path, 'rb'))
    gt_event_dict = {ref_event['file']: [{'file':ref_event['file'], 
                        'event_onset':ref_event['onset'], 
                        'event_offset':ref_event['offset'],
                        'event_label':ref_event['event_label']}]
                        for ref_event in gt_events}
    return gt_event_dict