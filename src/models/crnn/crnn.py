from __future__ import print_function
from sympy.logic import false

import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plot
import pprint
import json
import math

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

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
    """Loads a specific .npz file, including filename info"""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None, None, None

    print(f"Loading data from {filepath}...")
    dmp = np.load(filepath, allow_pickle=True)

    return dmp["arr_0"], dmp["arr_1"], dmp["arr_2"], dmp["arr_3"]


def plot_functions(
    checkpoint_dir, fig_name, _nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=""
):
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

    plot.savefig(checkpoint_dir + fig_name + extension)
    plot.close()
    print("figure name : {}".format(fig_name))


# --- PREPROCESSING ---
def preprocess_data(_X, _Y, _F, _seq_len, _nb_ch):
    # Split into sequences
    _X = split_in_seqs(_X, _seq_len)
    _Y = split_in_seqs(_Y, _seq_len)
    # Also split the filename indices
    _F = split_in_seqs(_F, _seq_len)

    # Split for channels
    _X = split_multi_channels(_X, _nb_ch)
    return _X, _Y, _F


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
def run_crnn(create_data: bool = False):

    if create_data:
        from src.models.crnn.process_audio import run_audio_processing

        run_audio_processing()

    is_mono = True
    feat_folder = "./feat_folder"
    fig_name = "{}_{}".format(
        "mon" if is_mono else "bin", time.strftime("%Y_%m_%d_%H_%M_%S")
    )

    nb_ch = 1 if is_mono else 2
    batch_size = 64
    nb_epoch = 1000
    patience = 5
    sr = 44100
    nfft = 2048
    frames_1_sec = (sr * 1.0) / (nfft / 2.0)
    seq_len = int(math.ceil(20 * frames_1_sec))

    print("\n\nUNIQUE ID: {}".format(fig_name))
    print(
        "TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}".format(
            nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec
        )
    )

    checkpoints_dir = "checkpoints"
    checkpoint_name = "crnn_model.pth"

    create_folder(checkpoints_dir)

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

    # Load Training Data (now includes file indices and filenames)
    X_train_full, Y_train_full, F_train_full, train_filenames_list = load_specific_file(
        train_file
    )
    if X_train_full is None:
        exit()

    # Load Final Test Data (now includes file indices and filenames)
    X_test_final, Y_test_final, F_test_final, test_filenames_list = load_specific_file(
        test_file
    )
    if X_test_final is None:
        exit()

    # Split "Training Data" into "Train" and "Validation"
    print("Splitting Training data into Train (80%) and Validation (20%)...")
    # Important: Split F (file indices) along with X and Y
    X_train, X_val, Y_train, Y_val, F_train, F_val = train_test_split(
        X_train_full, Y_train_full, F_train_full, test_size=0.2, random_state=42
    )

    # Preprocess all three sets (now includes F)
    print("Preprocessing datasets...")
    X_train, Y_train, F_train = preprocess_data(
        X_train, Y_train, F_train, seq_len, nb_ch
    )
    X_val, Y_val, F_val = preprocess_data(X_val, Y_val, F_val, seq_len, nb_ch)
    # F_test_seq will hold the file index for each sequence in the test set
    X_test, Y_test, F_test_seq = preprocess_data(
        X_test_final, Y_test_final, F_test_final, seq_len, nb_ch
    )

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

    best_epoch, pat_cnt, best_er, f1_for_best_er, best_val_loss = 0, 0, 99999, 0.0, 9999
    tr_loss, val_loss = [0] * nb_epoch, [0] * nb_epoch
    f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch
    posterior_thresh = 0.5

    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        for i in tqdm(range(nb_epoch)):
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
            # val_probs = np.concatenate(all_preds, axis=0)
            # val_thresh = (val_probs > posterior_thresh).astype(int)

            # Using Y_val here, not Y_test
            # score_list = compute_scores(
            #     pred=val_thresh,
            #     y=Y_val,
            #     pred_probs=val_probs,  # Passed for mAP calculation
            #     frames_in_1_sec=frames_1_sec,
            # )

            # f1_overall_1sec_list[i] = score_list["f1_overall_1sec"]
            # er_overall_1sec_list[i] = score_list["er_overall_1sec"]
            pat_cnt = pat_cnt + 1

            # Early Stopping Logic
            # if er_overall_1sec_list[i] < best_er:
            if val_loss[i] < best_val_loss:
            
                # best_er = er_overall_1sec_list[i]
                # f1_for_best_er = f1_overall_1sec_list[i]
                best_val_loss = val_loss[i]

                # Save the best model
                torch.save(
                    model.state_dict(), os.path.join(checkpoints_dir, "crnn_model.pth")
                )
                best_epoch = i
                pat_cnt = 0

            print(
                "tr Loss : {:.4f}, val Loss : {:.4f}, Best val loss : {:.4f}, best_epoch: {}".format(
                    tr_loss[i],
                    val_loss[i],
                    best_val_loss,
                    best_epoch,
                )
            )

            # plot_functions(
            #     checkpoints_dir,
            #     fig_name,
            #     nb_epoch,
            #     tr_loss,
            #     val_loss,
            #     f1_overall_1sec_list,
            #     er_overall_1sec_list,
            #     "_main",
            # )

            if pat_cnt > patience:
                print("Early stopping.")
                break

        print("\n\n--- TRAINING COMPLETE. EVALUATING ON UNSEEN TEST SET ---")
    # -----------------------------------------------------------------
    # FINAL EVALUATION ON UNSEEN TEST SET
    # -----------------------------------------------------------------

    # Load from checkpoint

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_test_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_test_preds.append(outputs.cpu().numpy())

    test_probs = np.concatenate(all_test_preds, axis=0)
    
    test_probs = gaussian_filter1d(
        input=test_probs, 
        sigma=3,
        axis=1,
        mode='nearest'
    )

    test_pred_thresh = (test_probs > posterior_thresh).astype(int)

    # Calculate metrics on Test Data
    # test_scores = compute_scores(
    #     pred=test_pred_thresh,
    #     y=Y_test,
    #     pred_probs=test_probs,  # Important: Pass raw probs for mAP
    #     frames_in_1_sec=frames_1_sec,
    # )

    # print(f"Final Test F1 Score (1-sec block): {test_scores['f1_overall_1sec']:.4f}")
    # print(f"Final Test Error Rate (1-sec block): {test_scores['er_overall_1sec']:.4f}")

    # Check if 'iou_macro' and 'map_macro' exist in keys (Safety check)
    # if "iou_macro" in test_scores:
    #     print(f"Final Test IoU (Macro): {test_scores['iou_macro']:.4f}")
    # if "map_macro" in test_scores:
    #     print(f"Final Test mAP (Macro): {test_scores['map_macro']:.4f}")

    ## Output Standard Format for metrics calculation
    idx_to_label = {0: "car_horn", 1: "cough", 2: "dog_bark", 3: "siren", 4: "gun_shot"}

    print(f"Number of test sequences: {len(test_pred_thresh)}")

    test_seq_file_indices = F_test_seq[:, 0].astype(int)
    test_filenames = [test_filenames_list[i] for i in test_seq_file_indices]


    print("Decoding events from predictions...")
    formatted_results = decode_predictions(
        preds=test_pred_thresh,
        filenames=test_filenames,
        class_labels=idx_to_label,
        frames_per_sec=frames_1_sec,
    )

    pp = pprint.PrettyPrinter(indent=4)
    print("\n--- Detected Events (Formatted - Top 5 Files) ---")
    # Sort keys for consistent display
    sorted_keys = sorted(formatted_results.keys())
    keys_to_show = sorted_keys[:5]
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
    formatted_results = run_crnn(create_data=False)
    print(formatted_results)
