# src/data_loader.py

import wfdb
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter
import os
import random
from torch.utils.data import DataLoader

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def select_device(device_preference='auto'):
    if device_preference == 'auto':
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_preference)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def load_data(config):
    record_ids = list(range(100, 235))
    normal_classes = config['data']['normal_classes']
    abnormal_classes = config['data']['abnormal_classes']
    
    all_X_data = []
    all_y_data = []
    resample_length = config['data']['resample_length']
    
    lowcut = 1
    highcut = 50.0
    fs = 360
    
    for record_id in record_ids:
        try:
            record = wfdb.rdrecord(os.path.join(config['data']['path'], f'{record_id}'))
            annotation = wfdb.rdann(os.path.join(config['data']['path'], f'{record_id}'), 'atr')
            
            ecg_signal = record.p_signal 
            ann_sample = annotation.sample  
            ann_symbols = annotation.symbol
    
            # Apply bandpass filter
            ecg_signal[:, 1] = bandpass_filter(ecg_signal[:, 1], lowcut, highcut, fs)
    
            X_data = []
            y_data = []
            for i in range(len(ann_sample)-2):
                x = ecg_signal[ann_sample[i]:ann_sample[i+1],:] # n x 2 channels
                y = 0 if ann_symbols[i+1] in normal_classes else 1
                # Resample beat-to-beat signal to resample_length
                x_interp = np.zeros((resample_length,2))
                x_interp[:,0] = np.interp(np.linspace(0,x.shape[0]-1,resample_length),np.arange(x.shape[0]),x[:,0])
                x_interp[:,1] = np.interp(np.linspace(0,x.shape[0]-1,resample_length),np.arange(x.shape[0]),x[:,1])
                X_data.append(x_interp)
                y_data.append(y)
     
            X_data = np.stack(X_data, axis=0) # num samples x 128 x 2
            y_data = np.stack(y_data, axis=0) # num_samples x 1
    
            all_X_data.append(X_data)
            all_y_data.append(y_data)
    
        except FileNotFoundError:
            continue
    
    # Concatenate all data
    all_X_data = np.concatenate(all_X_data, axis=0)
    all_y_data = np.concatenate(all_y_data, axis=0)
    
    # Normalization
    X_data_min = np.percentile(all_X_data.reshape(all_X_data.shape[0]*all_X_data.shape[1],all_X_data.shape[2]),3,axis=0)
    X_data_max = np.percentile(all_X_data.reshape(all_X_data.shape[0]*all_X_data.shape[1],all_X_data.shape[2]),97,axis=0)
    X_data_min = np.expand_dims(X_data_min,axis=(0,1))
    X_data_max = np.expand_dims(X_data_max,axis=(0,1))
    
    all_X_data = (all_X_data - X_data_min) / (X_data_max - X_data_min)
    all_X_data = np.transpose(all_X_data, (0, 2, 1)) # num samples x 2 x 128
    all_X_data = np.clip(all_X_data,a_min=0.0,a_max=1.0)
    
    # Min/max normalization
    X_data_min = np.min(all_X_data)
    X_data_max = np.max(all_X_data)
    all_X_data = (all_X_data - X_data_min) / (X_data_max - X_data_min)
    
    # Separate normal and abnormal data
    normal_data = all_X_data[all_y_data == 0]
    normal_labels = all_y_data[all_y_data == 0] 
    anomalous_data = all_X_data[all_y_data == 1]
    anomalous_labels = all_y_data[all_y_data == 1] 
    print(f"Total normal data samples: {normal_data.shape[0]}")
    print(f"Total anomalous data samples: {anomalous_data.shape[0]}")
    
    # Split normal data into training and testing
    X_train, normal_test_data, y_train, normal_test_labels = train_test_split(
        normal_data, normal_labels, test_size=config['data']['test_size'], random_state=42)
    
    # Test set includes remaining normal data and all abnormal data
    X_test = np.concatenate([normal_test_data, anomalous_data], axis=0)
    y_test = np.concatenate([normal_test_labels, anomalous_labels], axis=0)
    
    # Convert to tensors
    train_data = torch.tensor(X_train).float()
    train_labels = torch.tensor(y_train).long()
    test_data = torch.tensor(X_test).float()
    test_labels = torch.tensor(y_test).long()
    
    # Binary labels: True for normal, False for abnormal
    train_labels = (train_labels == 0).bool()
    test_labels = (test_labels == 0).bool()
    
    # Create DataLoader for training
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    
    return train_loader, test_data, test_labels
