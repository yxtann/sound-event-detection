import os
import numpy as np

eps = np.finfo(float).eps


def create_folder(_fold_path):
    """
    Creates a folder if it doesn't already exist.
    """
    os.makedirs(_fold_path, exist_ok=True)


def reshape_3Dto2D(A):
    """
    Combines the first two dimensions (batch, time) into one.
    """
    if A.ndim < 3:
        return A
    # (Batch, Time, Features) -> (Batch * Time, Features)
    return A.reshape(-1, A.shape[2])


def split_multi_channels(data, num_channels):
    """
    Splits the feature dimension of multi-channel data.
    Assumes features are interleaved (e.g., [L1, R1, L2, R2,...]).
    Input shape: (Batch, Time, F_total)
    Output shape: (Batch, Channel, Time, F_new)
    
    """
    if data.ndim != 3:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", data.shape)
        return None # Return None instead of exiting
        
    in_shape = data.shape
    hop = in_shape[2] // num_channels
    
    # (Batch, Time, F_total) -> (Batch, Time, Channel, F_new)
    data_reshaped = data.reshape((in_shape[0], in_shape[1], num_channels, hop))
    
    # (Batch, Time, Channel, F_new) -> (Batch, Channel, Time, F_new)
    data_transposed = data_reshaped.transpose((0, 2, 1, 3))
    return data_transposed


def split_in_seqs(data, subdivs):
    """
    Splits the data (along axis 0) into non-overlapping sequences.
    Discards any trailing data that doesn't fit a full sequence.
    """
    
    # Calculate number of full sequences
    n_seqs = data.shape[0] // subdivs
    if n_seqs == 0:
        # Return an empty array in the expected shape
        other_dims = (1,) if data.ndim == 1 else data.shape[1:]
        return np.empty((0, subdivs) + other_dims, dtype=data.dtype)

    # Calculate truncation point
    n_frames = n_seqs * subdivs
    
    # Truncate the data
    truncated_data = data[:n_frames]
    
    # Reshape
    # Handle the 1D case (add a new feature dimension)
    if data.ndim == 1:
        return truncated_data.reshape((n_seqs, subdivs, 1))
    
    # Handle 2D, 3D, and N-D cases
    else:
        new_shape = (n_seqs, subdivs) + data.shape[1:]
        return truncated_data.reshape(new_shape)
    
    
def find_contiguous_regions(activity_array):
    """
    Finds start and end indices of contiguous runs of '1's.
    e.g., [0, 1, 1, 1, 0, 0, 1, 1] -> [(1, 4), (6, 8)]
    """
    # Find the changes (diff)
    change_indices = np.diff(np.r_[0, activity_array, 0])
    # +1 means we stepped onto a 1 (onset)
    onset = np.where(change_indices == 1)[0]
    # -1 means we stepped off a 1 (offset)
    offset = np.where(change_indices == -1)[0]
    
    return zip(onset, offset)

def decode_predictions(preds, filenames, class_labels, frames_per_sec):
    """
    Converts binary predictions into the dictionary format shown in your image.
    """
    results = {}
    time_resolution = 1.0 / frames_per_sec
    
    # preds shape: (Num_Files, Time_Steps, Num_Classes)
    for i, file_pred in enumerate(preds):
        # Handle case where we have more preds than filenames (e.g. if files were split)
        if i >= len(filenames): break
            
        filename = filenames[i]
        results[filename] = []
        
        # Iterate over each class (Car Horn, Dog Bark, etc.)
        for class_idx, label in class_labels.items():
            # Get the binary trace for this class
            class_activity = file_pred[:, class_idx]
            
            # Find events
            events = find_contiguous_regions(class_activity)
            
            for onset_frame, offset_frame in events:
                # Convert frame index to seconds
                onset_time = onset_frame * time_resolution
                offset_time = offset_frame * time_resolution
                
                # Construct the dictionary entry
                event_entry = {
                    'file': filename,
                    'event_onset': round(onset_time, 3),  # Round to 3 decimals
                    'event_offset': round(offset_time, 3),
                    'event_label': label # e.g., 'car_horn'
                }
                results[filename].append(event_entry)
                
    return results