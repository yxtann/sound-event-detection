from __future__ import print_function
import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
import metrics  # TO-DO : Create metrics.py
import utils    # TO-DO : Create utils.py

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

plot.switch_backend('agg')
sys.setrecursionlimit(10000)


def load_data(_feat_folder, _mono):
    """
    Loads the single, normalized data file created by our process_audio.py script.
    """
    
    # This is the filename we created in the modified feature.py
    filename = 'mbe_{}_all.npz'.format('mon' if _mono else 'bin')
    feat_file = os.path.join(_feat_folder, filename)
    
    if not os.path.exists(feat_file):
        print(f"Error: Feature file not found at {feat_file}")
        print("Please make sure 'process_audio.py' ran successfully and")
        print(f"that '{filename}' exists in '{_feat_folder}'")
        return None, None
    
    # Load the file
    print(f"Loading data from {feat_file}...")
    dmp = np.load(feat_file)
    
    _X_all = dmp['arr_0']
    _Y_all = dmp['arr_1']
    
    return _X_all, _Y_all


def plot_functions(_nb_epoch, _tr_loss, _val_loss, _f1, _er, extension=''):
    plot.figure()

    plot.subplot(211)
    plot.plot(range(_nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(_nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(_nb_epoch), _f1, label='f')
    plot.plot(range(_nb_epoch), _er, label='er')
    plot.legend()
    plot.grid(True)

    plot.savefig(__models_dir + __fig_name + extension)
    plot.close()
    print('figure name : {}'.format(__fig_name))


def preprocess_data(_X, _Y, _X_test, _Y_test, _seq_len, _nb_ch):
    # split into sequences
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)

    _X_test = utils.split_in_seqs(_X_test, _seq_len)
    _Y_test = utils.split_in_seqs(_Y_test, _seq_len)

    _X = utils.split_multi_channels(_X, _nb_ch)
    _X_test = utils.split_multi_channels(_X_test, _nb_ch)
    return _X, _Y, _X_test, _Y_test


# --- Model Definition ---

class CRNN(nn.Module):
    def __init__(self, in_channels, n_freq, n_classes, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate):
        super(CRNN, self).__init__()
        
        self.cnn_layers = nn.Sequential()
        
        # CNN layers
        in_feat = in_channels
        for i, pool_size in enumerate(cnn_pool_size):
            self.cnn_layers.add_module(f'conv{i}', nn.Conv2d(in_feat, cnn_nb_filt, kernel_size=(3, 3), padding='same'))
            self.cnn_layers.add_module(f'bn{i}', nn.BatchNorm2d(cnn_nb_filt))
            self.cnn_layers.add_module(f'relu{i}', nn.ReLU())
            self.cnn_layers.add_module(f'pool{i}', nn.MaxPool2d(kernel_size=(1, pool_size)))
            self.cnn_layers.add_module(f'drop{i}', nn.Dropout(dropout_rate))
            in_feat = cnn_nb_filt # Input for next layer is output of this one

        # Calculate RNN input size
        # Input freq is n_freq.
        final_freq = n_freq
        for pool in cnn_pool_size:
            final_freq = final_freq // pool
        
        self.rnn_input_size = cnn_nb_filt * final_freq
        self.rnn_nb = rnn_nb
        
        # RNN layers (defined individually to handle 'mul' merge)
        in_feat_rnn = self.rnn_input_size
        for i, rnn_nodes in enumerate(rnn_nb):
            gru = nn.GRU(
                input_size=in_feat_rnn,
                hidden_size=rnn_nodes,
                num_layers=1, # Keras stacks single layers
                batch_first=True,
                bidirectional=True
            )
            self.add_module(f'gru{i}', gru)
            in_feat_rnn = rnn_nodes # 'mul' merge mode means output features == rnn_nodes
        
        # FC layers
        self.fc_layers = nn.Sequential()
        in_feat_fc = rnn_nb[-1] # Output of last GRU
        for i, fc_nodes in enumerate(fc_nb):
            self.fc_layers.add_module(f'fc{i}', nn.Linear(in_feat_fc, fc_nodes))
            self.fc_layers.add_module(f'drop{i}', nn.Dropout(dropout_rate))
            in_feat_fc = fc_nodes

        # Output layer
        self.output_layer = nn.Linear(in_feat_fc, n_classes)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, time, freq)
        
        # CNN
        x = self.cnn_layers(x)
        
        x = x.permute(0, 2, 1, 3) 
        
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        x = x.reshape(batch_size, time_steps, -1)
        
        # RNN
        for i, rnn_nodes in enumerate(self.rnn_nb):
            gru = getattr(self, f'gru{i}') # Get the GRU module
            x, _ = gru(x)
            # x shape: (batch, time, 2 * rnn_nodes)
            # Split and multiply for 'mul' merge
            (forward_out, backward_out) = torch.chunk(x, 2, dim=2)
            x = forward_out * backward_out # (batch, time, rnn_nodes)

        # FC (TimeDistributed(Dense) is just nn.Linear)
        x = self.fc_layers(x)
        
        # Output
        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return x


#######################################################################################
# MAIN SCRIPT STARTS HERE
#######################################################################################

if __name__ == '__main__':

    is_mono = True  # True: mono-channel input, False: binaural input

    feat_folder = './feat_folder'
    __fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))


    nb_ch = 1 if is_mono else 2
    batch_size = 128     
    seq_len = 256         
    nb_epoch = 500      
    patience = int(0.25 * nb_epoch)  # Patience for early stopping

    # Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
    # Make sure the nfft and sr are the same as in feature.py
    sr = 44100
    nfft = 2048

    # This means 1 second of audio = 43 frames
    frames_1_sec = int(sr/(nfft/2.0))

    print('\n\nUNIQUE ID: {}'.format(__fig_name))
    print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
        nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

    # Folder for saving model and training curves
    __models_dir = 'models/'
    utils.create_folder(__models_dir)

    # CRNN model definition
    cnn_nb_filt = 128         # CNN filter size
    cnn_pool_size = [5, 2, 2] # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
    rnn_nb = [32, 32]         # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
    fc_nb = [32]              # Number of FC nodes.  Length of fc_nb =  number of FC layers
    dropout_rate = 0.5        # Dropout after each layer
    print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
        cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))

    # --- PyTorch setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Set up for parallel processing (optional, but good practice)
    num_workers = min(os.cpu_count(), 4) 
    
    # -----------------------------------------------------------------
    # DATA LOADING 
    # -----------------------------------------------------------------

    # Load ALL feature and labels
    # The input to load_data is the result of process_audio.py -> mbe_mon_all.npz
    X_all, Y_all = load_data(feat_folder, is_mono)

    if X_all is None:
        print("Failed to load data. Exiting.")
        exit()
    
    # Create a single train/test split (e.g., 80% train, 20% test)
    X, X_test, Y, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)
    
    # Pre-process it
    X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)

    # Get shapes for model initialization
    # X shape: (num_samples, nb_ch, seq_len, freq)
    in_channels = X.shape[1]  # 1 channel
    n_freq = X.shape[3]       # 40 mel bands
    n_classes = Y.shape[-1]   # number of sound classes
    
    # Load model
    model = CRNN(
        in_channels=in_channels,
        n_freq=n_freq,
        n_classes=n_classes,
        cnn_nb_filt=cnn_nb_filt,
        cnn_pool_size=cnn_pool_size,
        rnn_nb=rnn_nb,
        fc_nb=fc_nb,
        dropout_rate=dropout_rate
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Create DataLoaders
    X_train_t = torch.tensor(X, dtype=torch.float32)
    Y_train_t = torch.tensor(Y, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_t, Y_test_t)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    # -----------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------

    # Initialise the variables
    # pat_cnt -> patience count : number of epochs since the model last improved
    # best_er -> best error rate
    # f1 score for the best error rate
    best_epoch, pat_cnt, best_er, f1_for_best_er = 0, 0, 99999, 0.0
    tr_loss, val_loss = [0] * nb_epoch, [0] * nb_epoch
    f1_overall_1sec_list, er_overall_1sec_list = [0] * nb_epoch, [0] * nb_epoch
    posterior_thresh = 0.5

    for i in range(nb_epoch):
        print('Epoch : {} '.format(i), end='')
        
        # --- Training Phase ---
        model.train()
        running_tr_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
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
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                all_preds.append(outputs.cpu().numpy())
        
        val_loss[i] = running_val_loss / len(test_loader)

        # Concatenate predictions from all batches
        # We need to un-do the sequencing to get the metric arrays
        pred = np.concatenate(all_preds, axis=0)

        # Apply 0.5 threshold
        pred_thresh = pred > posterior_thresh
        
        score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)

        f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
        er_overall_1sec_list[i] = score_list['er_overall_1sec']
        pat_cnt = pat_cnt + 1

        if er_overall_1sec_list[i] < best_er:
            best_er = er_overall_1sec_list[i]
            f1_for_best_er = f1_overall_1sec_list[i]
            
            # Save the best model
            torch.save(model.state_dict(), os.path.join(__models_dir, '{}_model.pth'.format(__fig_name)))
            
            best_epoch = i
            pat_cnt = 0
        
        print('tr Loss : {:.4f}, val Loss : {:.4f}, F1 : {:.4f}, ER : {:.4f} | Best ER : {:.4f}, best_epoch: {}'.format(
                tr_loss[i], val_loss[i], f1_overall_1sec_list[i], er_overall_1sec_list[i], best_er, best_epoch))

        plot_functions(nb_epoch, tr_loss, val_loss, f1_overall_1sec_list, er_overall_1sec_list, '_main')

        if pat_cnt > patience:
            print("Early stopping.")
            break
    
    print('\n\n--- FINAL RESULTS ---')
    print('saved model for the best_epoch: {} with best_er: {:.4f} and f1_for_best_er: {:.4f}'.format(
          best_epoch, best_er, f1_for_best_er))