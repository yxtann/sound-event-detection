from __future__ import print_function
import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plot
import pprint
import json

from pyarrow import bool8
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.models.crnn.utils import (
    create_folder,
    split_in_seqs,
    split_multi_channels,
    decode_predictions,
)
from src.models.crnn.metrics import compute_scores

plot.switch_backend("agg")
sys.setrecursionlimit(10000)


# --- DATA LOADING ---
def load_specific_file(filepath):
    """Loads a specific .npz file"""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None

    print(f"Loading data from {filepath}...")
    dmp = np.load(filepath)
    return dmp["arr_0"], dmp["arr_1"]


def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=""):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label="train loss")
    plot.plot(range(_nb_epoch), _val_loss, label="val loss")
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _f1, label="f")
    plot.plot(range(_nb_epoch), _er, label="er")
    plot.legend()
    plot.grid(True)

    plot.savefig(__models_dir + __fig_name + extension)
    plot.close()
    print("figure name : {}".format(__fig_name))


# --- PREPROCESSING ---
def preprocess_data(_X, _Y, _seq_len, _nb_ch):
    # Split into sequences
    _X = split_in_seqs(_X, _seq_len)
    _Y = split_in_seqs(_Y, _seq_len)

    # Split for channels
    _X = split_multi_channels(_X, _nb_ch)
    return _X, _Y


# --- Model Definition ---
class CRNN(nn.Module):
    def __init__(
        self,
        in_channels,
        n_freq,
        n_classes,
        cnn_nb_filt,
        cnn_pool_size,
        rnn_nb,
        fc_nb,
        dropout_rate,
    ):
        super(CRNN, self).__init__()
        self.cnn_layers = nn.Sequential()
        in_feat = in_channels
        for i, pool_size in enumerate(cnn_pool_size):
            self.cnn_layers.add_module(
                f"conv{i}",
                nn.Conv2d(in_feat, cnn_nb_filt, kernel_size=(3, 3), padding="same"),
            )
            self.cnn_layers.add_module(f"bn{i}", nn.BatchNorm2d(cnn_nb_filt))
            self.cnn_layers.add_module(f"relu{i}", nn.ReLU())
            self.cnn_layers.add_module(
                f"pool{i}", nn.MaxPool2d(kernel_size=(1, pool_size))
            )
            self.cnn_layers.add_module(f"drop{i}", nn.Dropout(dropout_rate))
            in_feat = cnn_nb_filt

        final_freq = n_freq
        for pool in cnn_pool_size:
            final_freq = final_freq // pool

        self.rnn_input_size = cnn_nb_filt * final_freq
        self.rnn_nb = rnn_nb

        in_feat_rnn = self.rnn_input_size
        for i, rnn_nodes in enumerate(rnn_nb):
            gru = nn.GRU(
                input_size=in_feat_rnn,
                hidden_size=rnn_nodes,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.add_module(f"gru{i}", gru)
            in_feat_rnn = rnn_nodes

        self.fc_layers = nn.Sequential()
        in_feat_fc = rnn_nb[-1]
        for i, fc_nodes in enumerate(fc_nb):
            self.fc_layers.add_module(f"fc{i}", nn.Linear(in_feat_fc, fc_nodes))
            self.fc_layers.add_module(f"drop{i}", nn.Dropout(dropout_rate))
            in_feat_fc = fc_nodes

        self.output_layer = nn.Linear(in_feat_fc, n_classes)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1, 3)
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        x = x.reshape(batch_size, time_steps, -1)

        for i, rnn_nodes in enumerate(self.rnn_nb):
            gru = getattr(self, f"gru{i}")
            x, _ = gru(x)
            (forward_out, backward_out) = torch.chunk(x, 2, dim=2)
            x = forward_out * backward_out

        x = self.fc_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x


#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################


def run_model(create_data: bool = False):

    if create_data:
        from src.models.crnn.process_audio import run_audio_processing

        run_audio_processing()

    # Settings
    is_mono = True
    feat_folder = "./feat_folder"
    __fig_name = "{}_{}".format(
        "mon" if is_mono else "bin", time.strftime("%Y_%m_%d_%H_%M_%S")
    )
    retrain = False

    nb_ch = 1 if is_mono else 2
    batch_size = 128
    seq_len = 256
    nb_epoch = 100
    patience = int(0.25 * nb_epoch)
    sr = 44100
    nfft = 2048
    frames_1_sec = int(sr / (nfft / 2.0))

    print("\n\nUNIQUE ID: {}".format(__fig_name))
    print(
        "TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}".format(
            nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec
        )
    )

    __models_dir = "checkpoints"
    create_folder(__models_dir)

    cnn_nb_filt = 128
    cnn_pool_size = [5, 2, 2]
    rnn_nb = [32, 32]
    fc_nb = [32]
    dropout_rate = 0.2
    print(
        "MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}".format(
            cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_workers = min(os.cpu_count(), 4)

    # -----------------------------------------------------------------
    # DATA SPLITTING LOGIC
    # -----------------------------------------------------------------

    # Path definitions
    train_file = os.path.join(feat_folder, "train_data.npz")
    test_file = os.path.join(feat_folder, "test_data.npz")

    # Load Training Data
    X_train_full, Y_train_full = load_specific_file(train_file)
    if X_train_full is None:
        exit()

    # Load Final Test Data (Hold out)
    X_test_final, Y_test_final = load_specific_file(test_file)
    if X_test_final is None:
        exit()

    # Track original number of frames before preprocessing
    original_test_frames = X_test_final.shape[0]

    # Split "Training Data" into "Train" and "Validation"
    print("Splitting Training data into Train (80%) and Validation (20%)...")
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=42
    )

    # Preprocess all three sets
    print("Preprocessing datasets...")
    X_train, Y_train = preprocess_data(X_train, Y_train, seq_len, nb_ch)
    X_val, Y_val = preprocess_data(X_val, Y_val, seq_len, nb_ch)
    X_test, Y_test = preprocess_data(
        X_test_final, Y_test_final, seq_len, nb_ch
    )  # Final test set

    # Model initialization
    in_channels = X_train.shape[1]
    n_freq = X_train.shape[3]
    n_classes = Y_train.shape[-1]

    model = CRNN(
        in_channels,
        n_freq,
        n_classes,
        cnn_nb_filt,
        cnn_pool_size,
        rnn_nb,
        fc_nb,
        dropout_rate,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Create 3 DataLoaders ---
    # Training Loader
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Validation Loader (Used for Early Stopping)
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Test Loader (Used ONLY at the end)
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # -----------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------

    best_epoch, pat_cnt, best_er, f1_for_best_er = 0, 0, 99999, 0.0
    tr_loss, val_loss = [0] * nb_epoch, [0] * nb_epoch
    f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch
    posterior_thresh = 0.5

    # Retrain if explicitly set 'retrain=True' or no checkpoints exist
    try:
        checkpoint = os.path.join(
            __models_dir,
            [
                filename
                for filename in os.listdir(__models_dir)
                if filename.startswith("mon")
            ][0],
        )  # Should only have 1 file
        print(f"Found checkpoint at {checkpoint}")
    except:
        print("No checkpoint found! Retraining...")
        checkpoint = None

    if retrain or checkpoint == None:
        for i in tqdm(range(nb_epoch), desc="Training: "):
            print("Epoch : {} ".format(i), end="")

            # --- Training Phase ---
            model.train()
            running_tr_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_tr_loss += loss.item()

            tr_loss[i] = running_tr_loss / len(train_loader)

            # --- Validation Phase ---
            model.eval()
            all_preds = []
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    )
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    all_preds.append(outputs.cpu().numpy())

            val_loss[i] = running_val_loss / len(val_loader)

            # Metrics calculation on VALIDATION set
            val_probs = np.concatenate(all_preds, axis=0)
            val_thresh = (val_probs > posterior_thresh).astype(int)

            # Using Y_val here, not Y_test
            score_list = compute_scores(
                pred=val_thresh,
                y=Y_val,
                pred_probs=val_probs,  # Passed for mAP calculation
                frames_in_1_sec=frames_1_sec,
            )

            f1_overall_1sec_list[i] = score_list["f1_overall_1sec"]
            er_overall_1sec_list[i] = score_list["er_overall_1sec"]
            pat_cnt = pat_cnt + 1

            # Early Stopping Logic
            if er_overall_1sec_list[i] < best_er:
                best_er = er_overall_1sec_list[i]
                f1_for_best_er = f1_overall_1sec_list[i]
                best_epoch = i
                pat_cnt = 0

                best_model_path = os.path.join(
                    __models_dir, "{}_model.pth".format(__fig_name)
                )

                # Clean up the older "best" models
                for file in os.listdir(__models_dir):
                    if file.endswith(".pth") and file.startswith("mon"):
                        os.remove(os.path.join(__models_dir, file))

                # Save the current best model
                torch.save(
                    model.state_dict(),
                    best_model_path,
                )

            print(
                "tr Loss : {:.4f}, val Loss : {:.4f}, F1 : {:.4f}, ER : {:.4f} | Best ER : {:.4f}, best_epoch: {}".format(
                    tr_loss[i],
                    val_loss[i],
                    f1_overall_1sec_list[i],
                    er_overall_1sec_list[i],
                    best_er,
                    best_epoch,
                )
            )

            # plot_functions(nb_epoch, tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list, '_main')

            if pat_cnt > patience:
                print("Early stopping.")
                break

    print("\n\n--- TRAINING COMPLETE. EVALUATING ON UNSEEN TEST SET ---")

    # -----------------------------------------------------------------
    # FINAL EVALUATION ON UNSEEN TEST SET
    # -----------------------------------------------------------------

    # Load best model
    # best_model_path = os.path.join(__models_dir, '{}_model.pth'.format(__fig_name))

    best_model_path = (
        checkpoint
        if checkpoint
        else os.path.join(__models_dir, "{}_model.pth".format(__fig_name))
    )
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_test_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_test_preds.append(outputs.cpu().numpy())

    test_probs = np.concatenate(all_test_preds, axis=0)
    test_pred_thresh = (test_probs > posterior_thresh).astype(int)

    # Calculate metrics on Test Data
    test_scores = compute_scores(
        pred=test_pred_thresh,
        y=Y_test,
        pred_probs=test_probs,  # Important: Pass raw probs for mAP
        frames_in_1_sec=frames_1_sec,
    )

    print(f"Final Test F1 Score (1-sec block): {test_scores['f1_overall_1sec']:.4f}")
    print(f"Final Test Error Rate (1-sec block): {test_scores['er_overall_1sec']:.4f}")

    # Check if 'iou_macro' and 'map_macro' exist in keys (Safety check)
    if "iou_macro" in test_scores:
        print(f"Final Test IoU (Macro): {test_scores['iou_macro']:.4f}")
    if "map_macro" in test_scores:
        print(f"Final Test mAP (Macro): {test_scores['map_macro']:.4f}")

    ## Output Standard Format for metrics calculation
    idx_to_label = {0: "car_horn", 1: "cough", 2: "dog_bark", 3: "siren", 4: "gun_shot"}

    print(len(test_pred_thresh))
    print(test_pred_thresh.shape)

    ####################
    # NOTE: ADDED CODE #
    ####################

    # Get test file list to track boundaries
    test_dir = os.path.join("data/processed/detection/test")
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".wav")])

    # Load individual files to track frame counts
    feat_folder = "./feat_folder"
    test_file_frames = []
    for filename in test_files:
        feat_file = os.path.join(feat_folder, f"{filename}_mon.npz")
        if os.path.exists(feat_file):
            dmp = np.load(feat_file)
            test_file_frames.append(dmp["arr_0"].shape[0])
        else:
            test_file_frames.append(0)

    # After getting test_pred_thresh, reshape and split by file:
    test_pred_reshaped = test_pred_thresh.reshape(-1, test_pred_thresh.shape[-1])

    # Split predictions back to original files
    file_predictions = []
    current_idx = 0
    for num_frames in test_file_frames:
        # Calculate how many sequences this file produced
        num_seqs = num_frames // seq_len
        seq_frames = num_seqs * seq_len
        end_idx = current_idx + seq_frames
        file_pred = test_pred_reshaped[current_idx:end_idx]
        file_predictions.append(file_pred)
        current_idx = end_idx

    # Now decode with correct mapping
    formatted_results = {}
    for filename, file_pred in zip(test_files, file_predictions):
        if file_pred.shape[0] > 0:  # Skip empty files
            formatted_results.update(
                decode_predictions(
                    preds=file_pred[np.newaxis, :, :],
                    filenames=[filename],
                    class_labels=idx_to_label,
                    frames_per_sec=frames_1_sec,
                )
            )

    ########################
    # NOTE: END ADDED CODE #
    ########################

    pp = pprint.PrettyPrinter(indent=4)
    print("\n--- Detected Events (Formatted) ---")
    keys_to_show = list(formatted_results.keys())[:5]
    subset = {k: formatted_results[k] for k in keys_to_show}
    pp.pprint(subset)

    # Define output filename
    output_json_file = "crnn_results.json"

    print(f"Saving results to {output_json_file}...")
    with open(output_json_file, "w") as f:
        # indent=4 makes it human-readable (pretty printed)
        json.dump(formatted_results, f, indent=4)

    print("Done.")

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_size_mb = total_params * 4 / (1024**2)

    print(f"--- Model Statistics ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Estimated Size:       {param_size_mb:.2f} MB")
    print(f"------------------------")

    return formatted_results


if __name__ == "__main__":
    run_model()
