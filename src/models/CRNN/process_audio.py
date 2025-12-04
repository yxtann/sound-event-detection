import os
import wave
import numpy as np
import pandas as pd  
import utils
import librosa
from sklearn import preprocessing
from joblib import Parallel, delayed
from huggingface_hub import snapshot_download
from huggingface_hub import login

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
    
    filename_col = 'filename' 
    start_time_col = 'onset'   
    end_time_col = 'offset'
    class_name_col = 'event_label'
    # ------------------
    
    for index, row in df.iterrows():
        name = os.path.basename(row[filename_col])
        
        # Get the class index (e.g., 1) from the class name
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
    
# -----------------------------------------------------------------------
# HELPER: GATHER FEATURES
# -----------------------------------------------------------------------
def gather_features(file_list, feat_folder, is_mono):
    X_all, Y_all, F_all = None, None, None

    for i, audio_filename in enumerate(file_list):
        tmp_feat_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_filename, 'mon' if is_mono else 'bin'))
        if not os.path.exists(tmp_feat_file):
            continue

        try:
            dmp = np.load(tmp_feat_file)
            tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']

            # Create an array of file indices for this file's frames
            tmp_f = np.full((tmp_mbe.shape[0],), i, dtype=np.int32)

            if X_all is None:
                X_all, Y_all, F_all = tmp_mbe, tmp_label, tmp_f
            else:
                X_all = np.concatenate((X_all, tmp_mbe), 0)
                Y_all = np.concatenate((Y_all, tmp_label), 0)
                F_all = np.concatenate((F_all, tmp_f), 0)
        except Exception as e:
            print(f"Error loading {tmp_feat_file}: {e}")

    return X_all, Y_all, F_all

# ###################################################################
# Main script
# ###################################################################

if __name__ == '__main__':
    login(token="") # Add hugging face token
    is_mono = True
    __class_labels = {
        'car_horn': 0,
        'cough': 1,
        'dog_bark': 2,
        'siren': 3,
        'gun_shot': 4
    }
    
    # -----------------------------------------------------------------
    # Download the repo from the Hub
    # -----------------------------------------------------------------
    
    REPO_ID = "kuross/dl-proj-detection" 
    print(f"Downloading dataset from Hugging Face Hub: {REPO_ID}...")

    root_folder = snapshot_download(repo_id=REPO_ID, repo_type="dataset", max_workers=4)
    print(f"Dataset downloaded to: {root_folder}")

    # Define specific subfolders
    train_dir = os.path.join(root_folder, 'train')
    test_dir = os.path.join(root_folder, 'test')

    # Specific Paths to CSVs
    train_csv = os.path.join(train_dir, 'annotations.csv')
    test_csv = os.path.join(test_dir, 'annotations.csv')
    
    # Output folder for features (now inside the downloaded folder)
    feat_folder = os.path.join('.', 'feat_folder')
    utils.create_folder(feat_folder) 
    
    # User set parameters 
    nfft = 2048
    hop_len = nfft // 2
    nb_mel_bands = 40
    sr = 44100
    
    # -----------------------------------------------------------------
    # Feature extraction and label generation
    # -----------------------------------------------------------------
    
    # LOAD LABELS FROM TRAIN AND TEST
    print(f"Loading Train labels from {train_csv}...")
    train_desc_dict = load_desc_from_csv(train_csv, __class_labels)
    
    print(f"Loading Test labels from {test_csv}...")
    test_desc_dict = load_desc_from_csv(test_csv, __class_labels)

    # Get file lists (exclude the csv files themselves from the wav list)
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.wav')]
    test_files  = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    
    print(f"Found {len(train_files)} Train files and {len(test_files)} Test files.")
    
    # PROCESS TRAINING DATA
    print("Processing TRAINING files...")
    Parallel(n_jobs=-1, backend="threading", verbose=5)(
        delayed(process_file)(
            f, train_dir, train_desc_dict, sr, nfft, hop_len, nb_mel_bands, is_mono, __class_labels, feat_folder
        ) for f in train_files
    )

    # PROCESS TEST DATA
    print("Processing TEST files...")
    Parallel(n_jobs=-1, backend="threading", verbose=5)(
        delayed(process_file)(
            f, test_dir, test_desc_dict, sr, nfft, hop_len, nb_mel_bands, is_mono, __class_labels, feat_folder
        ) for f in test_files
    )

    # -----------------------------------------------------------------
    # Feature Normalization
    # -----------------------------------------------------------------
    print("Starting feature normalization...")

    X_all, Y_all = None, None

    print("Gathering Training Features...")
    # Now also gathers file indices (F_train)
    X_train, Y_train, F_train = gather_features(train_files, feat_folder, is_mono)

    print("Gathering Test Features...")
    # Now also gathers file indices (F_test)
    X_test, Y_test, F_test = gather_features(test_files, feat_folder, is_mono)

    if X_train is not None and X_test is not None:
        print("Fitting StandardScaler on TRAINING data only...")
        scaler = preprocessing.StandardScaler()

        # FIT on Train
        X_train = scaler.fit_transform(X_train)

        # TRANSFORM Test (using Train stats)
        X_test = scaler.transform(X_test)

        # --- SAVE ---
        train_out = os.path.join(feat_folder, 'train_data.npz')
        test_out = os.path.join(feat_folder, 'test_data.npz')

        # Save X, Y, F (indices), and the original file lists
        np.savez(train_out, X_train, Y_train, F_train, train_files)
        np.savez(test_out, X_test, Y_test, F_test, test_files)

        print("Processing Complete.")
        print(f"Saved Training Data: {train_out} (Shape: {X_train.shape})")
        print(f"Saved Test Data:     {test_out}  (Shape: {X_test.shape})")
    else:
        print("Error: Could not gather features. Check if .wav files exist and were processed.")