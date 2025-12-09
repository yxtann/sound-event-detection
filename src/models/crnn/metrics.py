import numpy as np
from sklearn.metrics import average_precision_score

eps = np.finfo(float).eps

# --- Internal Helper Function ---

def _reshape_3Dto2D(data):
    """
    Combines the first two dimensions (batch, time) into one.
    """
    if data.ndim < 3:
        return data  # Already 2D
    # (Batch, Time, Features) -> (Batch * Time, Features)
    return data.reshape(-1, data.shape[-1])

def iou_framewise(O, T):
    """
    Calculates Class-wise Intersection over Union (IoU) and returns the mean.
    Input: Binary (Thresholded) predictions.
    """
    if O.ndim > 2: O = _reshape_3Dto2D(O)
    if T.ndim > 2: T = _reshape_3Dto2D(T)
    
    # Axis 0 = Time/Batch axis. We sum down this axis to get counts per class.
    # Intersection: Both O and T are 1
    intersection = np.logical_and(O == 1, T == 1).sum(axis=0)
    
    # Union: Either O or T is 1
    union = np.logical_or(O == 1, T == 1).sum(axis=0)
    
    # Calculate IoU per class
    # Add eps to denominator to avoid division by zero
    iou_per_class = intersection / (union + eps)
    
    # Return the average across all classes (Macro IoU)
    return np.mean(iou_per_class)

def map_framewise(O_probs, T):
    """
    Calculates Mean Average Precision.
    Input: RAW PROBABILITIES (Not thresholded).
    """
    if O_probs.ndim > 2: O_probs = _reshape_3Dto2D(O_probs)
    if T.ndim > 2: T = _reshape_3Dto2D(T)
    
    # 'macro': Calculate AP for each class and take the average
    # This ensures rare classes (like gunshots) have equal weight to common ones
    try:
        score = average_precision_score(T, O_probs, average='macro')
    except ValueError:
        score = 0.0 # Handle edge cases where a class might not be present in batch
        
    return score

def f1_overall_framewise(O, T):
    """
    Calculates framewise F1-score.
    """
    if O.ndim > 2:
        O = _reshape_3Dto2D(O)
    if T.ndim > 2:
        T = _reshape_3Dto2D(T)
        
    # This trick (2*T - O) correctly finds TP, FP, FN
    # TP = (2*1 - 1) == 1
    TP = ((2 * T - O) == 1).sum()
    
    # Nref = T.sum() (Total positives in Ground Truth)
    # Nsys = O.sum() (Total positives in Prediction)
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    recall = float(TP) / float(Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def er_overall_framewise(O, T):
    """
    Calculates framewise Error Rate.
    """
    if O.ndim > 2:
        O = _reshape_3Dto2D(O)
    if T.ndim > 2:
        T = _reshape_3Dto2D(T)
        
    # False Positives (FP)
    FP = np.logical_and(T == 0, O == 1).sum() 
    # False Negatives (FN)
    FN = np.logical_and(T == 1, O == 0).sum()

    Nref = T.sum()
    if Nref == 0:
        # Avoid division by zero if ground truth is all-negative
        return float(FP) / (eps + O.shape[0] * O.shape[1]) # Return FP rate
        
    S = np.minimum(FP, FN)
    D = np.maximum(0, FN - FP) # Deletions
    I = np.maximum(0, FP - FN) # Insertions

    ER = float(S + D + I) / float(Nref + eps)
    return ER


# --- 1-Second Block Metrics (Vectorized) ---

def _vectorized_block_max(data, block_size):
    """
    Helper function to replace the slow 'for' loop.
    This does the block-wise max operation in one go.
    """
    if data.ndim > 2:
        data = _reshape_3Dto2D(data)
        
    # Find number of full blocks (truncates excess frames)
    n_full_blocks = data.shape[0] // block_size
    truncated_frames = int(n_full_blocks * block_size)
    
    # Truncate array to be a multiple of block_size
    data_trunc = data[:truncated_frames, :]
    
    # Reshape and get max
    # (Total_Frames, Classes) -> (N_Blocks, Block_Size, Classes)
    data_reshaped = data_trunc.reshape((n_full_blocks, block_size, data.shape[1]))
    
    # Take max over the 'Block_Size' dimension (axis=1)
    # (N_Blocks, Block_Size, Classes) -> (N_Blocks, Classes)
    block_max = data_reshaped.max(axis=1)
    return block_max


def f1_overall_1sec(O, T, block_size):
    """
    Calculates 1-second block F1-score.
    """
    O_block = _vectorized_block_max(O, block_size)
    T_block = _vectorized_block_max(T, block_size)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    """
    Calculates 1-second block Error Rate.
    """
    O_block = _vectorized_block_max(O, block_size)
    T_block = _vectorized_block_max(T, block_size)
    return er_overall_framewise(O_block, T_block)


# --- Main Function ---

def compute_scores(pred, y, pred_probs=None, frames_in_1_sec=50):
    """
    Main wrapper to compute segment-based F1 and ER.
    """
    scores = dict()
    scores['f1_overall_1sec'] = f1_overall_1sec(pred, y, frames_in_1_sec)
    scores['er_overall_1sec'] = er_overall_1sec(pred, y, frames_in_1_sec)
    scores['iou_macro'] = iou_framewise(pred, y)

    if pred_probs is not None:
        scores['map_macro'] = map_framewise(pred_probs, y)
    else:
        scores['map_macro'] = 0.0 # Placeholder

    return scores