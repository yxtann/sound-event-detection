import os
import numpy as np

# Use 'float' or 'np.float64' instead of the deprecated 'np.float'
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
    # This transpose is the same operation as the original for loop.
    data_transposed = data_reshaped.transpose((0, 2, 1, 3))
    return data_transposed


def split_in_seqs(data, subdivs):
    """
    Splits the data (along axis 0) into non-overlapping sequences.
    Discards any trailing data that doesn't fit a full sequence.
    """
    
    # 1. Calculate number of full sequences
    n_seqs = data.shape[0] // subdivs
    if n_seqs == 0:
        # Return an empty array in the expected shape
        other_dims = (1,) if data.ndim == 1 else data.shape[1:]
        return np.empty((0, subdivs) + other_dims, dtype=data.dtype)

    # 2. Calculate truncation point
    n_frames = n_seqs * subdivs
    
    # 3. Truncate the data
    truncated_data = data[:n_frames]
    
    # 4. Reshape
    # Handle the 1D case (add a new feature dimension)
    if data.ndim == 1:
        return truncated_data.reshape((n_seqs, subdivs, 1))
    
    # Handle 2D, 3D, and N-D cases
    else:
        new_shape = (n_seqs, subdivs) + data.shape[1:]
        return truncated_data.reshape(new_shape)