# src/evaluate.py
import numpy as np
import torch
import torch.nn as nn
import yaml
import os
from torch.utils.data import DataLoader
from models import VEncoder, Generator1, Generator2
from data_loader import load_data, select_device
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def compute_spectral_error(residuals, confidence_maps, W):
    residuals_fft = torch.fft.fft(residuals)
    spectral_error = torch.abs(residuals_fft)**2 - (confidence_maps * W)
    return torch.mean(spectral_error, dim=1)

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Select device
    device = select_device(config['device'])
    print(f"Using device: {device}")
    
    # Load test data
    _, test_data, test_labels = load_data(config)
    
    # Model parameters
    nz = config['model']['nz']
    nx = config['model']['nx']
    ngf = config['model']['ngf']
    ndf = config['model']['ndf']
    nc = config['training']['nc']
    ngpu = config['training']['ngpu']
    
    # Initialize models
    netE = VEncoder(nc, ndf, nx, ngpu).to(device)
    netG1 = Generator1(nx, ngf, nc, ngpu).to(device)
    netG2 = Generator2(nx, ngf, nc, ngpu).to(device)
    
    # Load trained model weights
    netE.load_state_dict(torch.load('outputs/models/netE_epoch_100.pth', map_location=device))
    netG1.load_state_dict(torch.load('outputs/models/netG1_epoch_100.pth', map_location=device))
    netG2.load_state_dict(torch.load('outputs/models/netG2_epoch_100.pth', map_location=device))
    
    netE.eval()
    netG1.eval()
    netG2.eval()
    
    # Evaluation parameters
    batch_size = config['evaluation']['batch_size']
    z_epochs = config['evaluation']['z_epochs']
    z_lr = config['evaluation']['z_lr']
    W = 0  # Adjust based on your needs
    
    # Prepare DataLoader for test data
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Loss function
    criterion = nn.L1Loss()
    
    # Initialize lists to store errors
    normal_errors = []
    abnormal_errors = []
    
    data = test_data
    labels = test_labels
    reals = data.to(device)
    labels = labels.to(device)
    half_reals = reals[:,:,0:64]
    target_reals = reals[:,:,64:]
    b_size = reals.size(0)
    
    with torch.no_grad():
        mu, sigma = netE(half_reals)
    
    # Optimize Z
    z = torch.zeros(b_size, nz, 1, requires_grad=True, device=device)
    optimizerZ = torch.optim.Adam([z], lr=z_lr)
    for Zepoch in range(z_epochs):
        optimizerZ.zero_grad()
        xz = mu + (z * sigma)
        half_fakes = netG1(xz)
        loss = criterion(half_fakes, target_reals)
        loss.backward()
        optimizerZ.step()
    
    # Phase 3: Update generators and encoder with optimal z
    z = z.detach()
    xz = mu + (z * sigma)
    half_fakes = netG1(xz)
    confidence_est = netG2(xz)
    normal_confidence_maps = confidence_est[labels==True,0,:]
    abnormal_confidence_maps = confidence_est[labels==False,0,:]
    normal_residuals = (torch.abs(half_fakes[labels==True,0,:]-target_reals[labels==True,0,:]))
    abnormal_residuals = (torch.abs(half_fakes[labels==False,0,:]-target_reals[labels==False,0,:]))

    # Compute spectral error
    normal_errors_spectral = compute_spectral_error(normal_residuals, normal_confidence_maps, W)
    abnormal_errors_spectral = compute_spectral_error(abnormal_residuals, abnormal_confidence_maps, W)
    normal_errors = normal_errors_spectral.cpu()  # Replace with other error metrics as needed
    abnormal_errors = abnormal_errors_spectral.cpu()
    normal_errors = normal_errors.cpu()
    abnormal_errors = abnormal_errors.cpu()
    
    # Calculate metrics
    thresholds = np.linspace(0, 1, 10000)
    tpr_values = []
    fpr_values = []
    pre_values = []
    acc_values = []
    for t in thresholds:
        TP = torch.sum(abnormal_errors > t).item()
        FN = torch.sum(abnormal_errors <= t).item()
        FP = torch.sum(normal_errors > t).item()
        TN = torch.sum(normal_errors <= t).item()
        
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0 
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0 
        PRE = TP / (TP + FP) if (TP + FP) > 0 else 0 
        ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        
        tpr_values.append(TPR)
        fpr_values.append(FPR)
        pre_values.append(PRE)
        acc_values.append(ACC)
    
    fprs = np.array(fpr_values)
    tprs = np.array(tpr_values)
    pres = np.array(pre_values)
    accs = np.array(acc_values)
    
    # Calculate AUC and F1 Score
    auc_score = auc(fprs, tprs)
    epsilon = 1e-10
    f1_score = 2 * (pres * tprs) / (pres + tprs + epsilon)
    max_f1_index = np.argmax(f1_score)
    max_f1_score = f1_score[max_f1_index] * 100 
    corresponding_tpr = tprs[max_f1_index] * 100 
    corresponding_pres = pres[max_f1_index] * 100 
    accuracy = np.max(accs) * 100 
    
    print(f"F1: {max_f1_score:.1f}%")
    print(f"Recall: {corresponding_tpr:.1f}%")
    print(f"Precision: {corresponding_pres:.1f}%")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fprs, tprs, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.text(0.6, 0.2, f"F1 Score: {max_f1_score:.2f}%", fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.6, 0.1, f"Accuracy: {accuracy:.2f}%", fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/ROC.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
