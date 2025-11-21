from __future__ import print_function
import os
import numpy as np
import time
import sys
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
import metrics 
import utils    

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

plot.switch_backend('agg')
sys.setrecursionlimit(10000)

# --- DATA LOADING ---
def load_specific_file(filepath):
    """ Loads a specific .npz file """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None
    
    print(f"Loading data from {filepath}...")
    dmp = np.load(filepath)
    return dmp['arr_0'], dmp['arr_1']

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

# --- PREPROCESSING ---
def preprocess_data(_X, _Y, _seq_len, _nb_ch):
    # Split into sequences
    _X = utils.split_in_seqs(_X, _seq_len)
    _Y = utils.split_in_seqs(_Y, _seq_len)

    # Split for channels
    _X = utils.split_multi_channels(_X, _nb_ch)
    return _X, _Y

# --- Model Definition ---
class CRNN(nn.Module):
    def __init__(self, in_channels, n_freq, n_classes, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate):
        super(CRNN, self).__init__()
        self.cnn_layers = nn.Sequential()
        in_feat = in_channels
        for i, pool_size in enumerate(cnn_pool_size):
            self.cnn_layers.add_module(f'conv{i}', nn.Conv2d(in_feat, cnn_nb_filt, kernel_size=(3, 3), padding='same'))
            self.cnn_layers.add_module(f'bn{i}', nn.BatchNorm2d(cnn_nb_filt))
            self.cnn_layers.add_module(f'relu{i}', nn.ReLU())
            self.cnn_layers.add_module(f'pool{i}', nn.MaxPool2d(kernel_size=(1, pool_size)))
            self.cnn_layers.add_module(f'drop{i}', nn.Dropout(dropout_rate))
            in_feat = cnn_nb_filt 

        final_freq = n_freq
        for pool in cnn_pool_size:
            final_freq = final_freq // pool
        
        self.rnn_input_size = cnn_nb_filt * final_freq
        self.rnn_nb = rnn_nb
        
        in_feat_rnn = self.rnn_input_size
        for i, rnn_nodes in enumerate(rnn_nb):
            gru = nn.GRU(input_size=in_feat_rnn, hidden_size=rnn_nodes, num_layers=1, batch_first=True, bidirectional=True)
            self.add_module(f'gru{i}', gru)
            in_feat_rnn = rnn_nodes 
        
        self.fc_layers = nn.Sequential()
        in_feat_fc = rnn_nb[-1] 
        for i, fc_nodes in enumerate(fc_nb):
            self.fc_layers.add_module(f'fc{i}', nn.Linear(in_feat_fc, fc_nodes))
            self.fc_layers.add_module(f'drop{i}', nn.Dropout(dropout_rate))
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
            gru = getattr(self, f'gru{i}')
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

if __name__ == '__main__':

    is_mono = True 
    feat_folder = './feat_folder'
    __fig_name = '{}_{}'.format('mon' if is_mono else 'bin', time.strftime("%Y_%m_%d_%H_%M_%S"))

    nb_ch = 1 if is_mono else 2
    batch_size = 128     
    seq_len = 256         
    nb_epoch = 500      
    patience = int(0.25 * nb_epoch)
    sr = 44100
    nfft = 2048
    frames_1_sec = int(sr/(nfft/2.0))

    print('\n\nUNIQUE ID: {}'.format(__fig_name))
    print('TRAINING PARAMETERS: nb_ch: {}, seq_len: {}, batch_size: {}, nb_epoch: {}, frames_1_sec: {}'.format(
        nb_ch, seq_len, batch_size, nb_epoch, frames_1_sec))

    __models_dir = 'models/'
    utils.create_folder(__models_dir)

    cnn_nb_filt = 128        
    cnn_pool_size = [5, 2, 2] 
    rnn_nb = [32, 32]        
    fc_nb = [32]              
    dropout_rate = 0.5        
    print('MODEL PARAMETERS:\n cnn_nb_filt: {}, cnn_pool_size: {}, rnn_nb: {}, fc_nb: {}, dropout_rate: {}'.format(
        cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    num_workers = min(os.cpu_count(), 4) 
    
    # -----------------------------------------------------------------
    # DATA SPLITTING LOGIC
    # -----------------------------------------------------------------

    # Path definitions
    train_file = os.path.join(feat_folder, 'train_data.npz')
    test_file = os.path.join(feat_folder, 'test_data.npz')

    # Load Training Data
    X_train_full, Y_train_full = load_specific_file(train_file)
    if X_train_full is None: exit()

    # Load Final Test Data (Hold out)
    X_test_final, Y_test_final = load_specific_file(test_file)
    if X_test_final is None: exit()

    # Split "Training Data" into "Train" and "Validation"
    print("Splitting Training data into Train (80%) and Validation (20%)...")
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2, random_state=42)

    # Preprocess all three sets
    print("Preprocessing datasets...")
    X_train, Y_train = preprocess_data(X_train, Y_train, seq_len, nb_ch)
    X_val,   Y_val   = preprocess_data(X_val, Y_val, seq_len, nb_ch)
    X_test,  Y_test  = preprocess_data(X_test_final, Y_test_final, seq_len, nb_ch) # Final test set

    # Model initialization
    in_channels = X_train.shape[1]  
    n_freq = X_train.shape[3]       
    n_classes = Y_train.shape[-1]   
    
    model = CRNN(in_channels, n_freq, n_classes, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # --- Create 3 DataLoaders ---
    # Training Loader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Validation Loader (Used for Early Stopping)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Test Loader (Used ONLY at the end)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -----------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------
    
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
        
        val_loss[i] = running_val_loss / len(val_loader)

        # Metrics calculation on VALIDATION set
        pred = np.concatenate(all_preds, axis=0)
        pred_thresh = pred > posterior_thresh
        
        # Using Y_val here, not Y_test
        score_list = metrics.compute_scores(pred_thresh, Y_val, frames_in_1_sec=frames_1_sec)

        f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
        er_overall_1sec_list[i] = score_list['er_overall_1sec']
        pat_cnt = pat_cnt + 1

        # Early Stopping Logic
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
    
    # -----------------------------------------------------------------
    # FINAL EVALUATION ON UNSEEN TEST SET
    # -----------------------------------------------------------------
    print('\n\n--- TRAINING COMPLETE. EVALUATING ON UNSEEN TEST SET ---')
    
    # Load best model
    best_model_path = os.path.join(__models_dir, '{}_model.pth'.format(__fig_name))
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_test_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_test_preds.append(outputs.cpu().numpy())

    test_pred = np.concatenate(all_test_preds, axis=0)
    test_pred_thresh = test_pred > posterior_thresh
    
    # Calculate metrics on Test Data
    test_scores = metrics.compute_scores(test_pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)
    
    print(f"Final Test F1 Score: {test_scores['f1_overall_1sec']:.4f}")
    print(f"Final Test Error Rate: {test_scores['er_overall_1sec']:.4f}")