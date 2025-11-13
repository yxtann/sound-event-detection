import os
import wave
import numpy as np
import pandas as pd  
import utils
import librosa
from sklearn import preprocessing
from joblib import Parallel, delayed
from huggingface_hub import snapshot_download

# -----------------------------------------------------------------------
# AUDIO LOADING
# -----------------------------------------------------------------------
def load_audio(filename, mono=True, fs=44100):
    y, sr = librosa.load(filename, sr=fs, mono=mono)
    return y, sr

# -----------------------------------------------------------------------
# FEATURE EXTRACTION
# -----------------------------------------------------------------------
def extract_mbe(_y, _sr, _nfft, _nb_mel, _hop_len):
    S_amplitude = librosa.feature.melspectrogram(
        y=_y,
        sr=_sr,
        n_fft=_nfft,
        hop_length=_hop_len,
        n_mels=_nb_mel,
        power=1.0
    )
    mbe = np.log(S_amplitude + 1e-9)
    return mbe

# -----------------------------------------------------------------------
# LABEL LOADING (Reads annotations.csv)
# -----------------------------------------------------------------------
def load_desc_from_csv(csv_file, class_labels_dict):
    """
    Reads the annotations.csv file and builds the desc_dict.
    """
    _desc_dict = dict()
    df = pd.read_csv(csv_file)
    
    # --- IMPORTANT ---
    # Ddjust these column names if your .csv
    # headers are different.
    filename_col = 'filename' 
    start_time_col = 'onset'   
    end_time_col = 'offset'
    class_name_col = 'event_label'
    # ------------------
    
    for index, row in df.iterrows():
        name = os.path.basename(row[filename_col])
        
        # Get the class index (e.g., 1) from the class name (e.g., "car")
        class_name = row[class_name_col]
        if class_name not in class_labels_dict:
            continue

        class_idx = class_labels_dict[class_name]
        
        # Build the dictionary entry
        if name not in _desc_dict:
            _desc_dict[name] = list()
        
        _desc_dict[name].append([
            float(row[start_time_col]), 
            float(row[end_time_col]), 
            class_idx
        ])
            
    return _desc_dict

# -----------------------------------------------------------------------
# FILE PROCESSING
# -----------------------------------------------------------------------
def process_file(audio_filename, audio_folder, desc_dict, sr, nfft, hop_len, nb_mel_bands, is_mono, __class_labels, feat_folder):
    try:
        audio_file = os.path.join(audio_folder, audio_filename)
        
        y, sr_loaded = load_audio(audio_file, mono=is_mono, fs=sr)
        
        mbe = extract_mbe(y, sr_loaded, nfft, nb_mel_bands, hop_len).T

        label = np.zeros((mbe.shape[0], len(__class_labels)))
        
        if audio_filename in desc_dict:
            tmp_data = np.array(desc_dict[audio_filename])
            frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int)
            frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)
            se_class = tmp_data[:, 2].astype(int)
            
            for ind, val in enumerate(se_class):
                start_idx = min(frame_start[ind], mbe.shape[0] - 1)
                end_idx = min(frame_end[ind], mbe.shape[0])
                if start_idx < end_idx:
                    label[start_idx:end_idx, val] = 1

        tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
        np.savez(tmp_feat_file, mbe, label)
        
        return f"Processed {audio_filename}"
    
    except Exception as e:
        return f"ERROR processing {audio_filename}: {e}"

# ###################################################################
# Main script
# ###################################################################

if __name__ == '__main__':
    is_mono = True
    __class_labels = {
        'gun_shot': 0,
        'doorbell': 1,
        'cough': 2,
        'siren': 3,
        'traffic': 4,
        'airport' : 5,
        'forest' : 6,
        'dog_bark' : 7,
        'coffee_shop' : 8
    }
    
    # -----------------------------------------------------------------
    # Download the repo from the Hub
    # -----------------------------------------------------------------
    
    REPO_ID = "kuross/dl-proj-detection" 
    
    print(f"Downloading dataset from Hugging Face Hub: {REPO_ID}...")
    
    # This function downloads the entire repo to a local cache
    # and returns the path to that folder.
    data_folder = snapshot_download(
        repo_id=REPO_ID, 
        repo_type="dataset"
    ) 
    
    print(f"Dataset downloaded to: {data_folder}")
    # -----------------------------------------------------------------
    
    # Output folder for features (now inside the downloaded folder)
    feat_folder = os.path.join('.', 'feat_folder')
    utils.create_folder(feat_folder) 
    
    # User set parameters 
    nfft = 2048
    hop_len = nfft // 2
    nb_mel_bands = 40
    sr = 44100
    
    # -----------------------------------------------------------------
    # 1. Feature extraction and label generation
    # -----------------------------------------------------------------
    
    # Load ALL labels from the single annotations.csv file
    print("Loading labels from annotations.csv...")
    annotations_file = os.path.join(data_folder, 'annotations.csv')
    desc_dict = load_desc_from_csv(annotations_file, __class_labels)
    print(f"Loaded labels for {len(desc_dict)} audio files.")

    # Get list of all .wav files in the data folder
    audio_files_list = [f for f in os.listdir(data_folder) if f.endswith('.wav')]
    print(f"Found {len(audio_files_list)} .wav files to process.")
    
    # Process all files in parallel
    print("Starting parallel feature extraction...")
    results = Parallel(n_jobs=-1, backend="threading", verbose=10)(
        delayed(process_file)(
            audio_filename,
            data_folder,  # Use the single data folder
            desc_dict,
            sr, nfft, hop_len, nb_mel_bands, is_mono,
            __class_labels, feat_folder
        ) for audio_filename in audio_files_list
    )
    print("Feature extraction complete.")

    # -----------------------------------------------------------------
    # Feature Normalization
    # -----------------------------------------------------------------
    print("Starting feature normalization...")
    
    X_all, Y_all = None, None
    
    # Load ALL the .npz files we just saved
    for audio_filename in audio_files_list:
        tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
        if not os.path.exists(tmp_feat_file):
            print(f"Warning: Missing feature file {tmp_feat_file}")
            continue
            
        try:
            dmp = np.load(tmp_feat_file)
            tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
            
            if X_all is None:
                X_all, Y_all = tmp_mbe, tmp_label
            else:
                X_all = np.concatenate((X_all, tmp_mbe), 0)
                Y_all = np.concatenate((Y_all, tmp_label), 0)
        except Exception as e:
            print(f"Error loading {tmp_feat_file}: {e}")

    # Normalize ALL the data
    print("Fitting StandardScaler on all data...")
    scaler = preprocessing.StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    # Save the final, normalized .npz file
    normalized_feat_file = os.path.join(feat_folder, 'mbe_mon_all.npz')
    np.savez(normalized_feat_file, X_all, Y_all)
    print('Normalized feature file saved: {}'.format(normalized_feat_file))
    print("All processing complete.")