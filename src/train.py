# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from models import VEncoder, Generator1, Generator2, Discriminator
from data_loader import load_data, set_random_seed, select_device
from utils import weights_init
import matplotlib.pyplot as plt

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Set random seed
    set_random_seed(config['random_seed'])
    
    # Select device
    device = select_device(config['device'])
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_data, test_labels = load_data(config)
    
    # Model parameters
    nz = config['model']['nz']
    nx = config['model']['nx']
    ngf = config['model']['ngf']
    ndf = config['model']['ndf']
    nc = config['training']['nc']
    ngpu = config['training']['ngpu']
    
    # Initialize models
    netE = VEncoder(nc, ndf, nx, ngpu).to(device)
    netD = Discriminator(nc, ndf, ngpu).to(device)
    netG1 = Generator1(nx, ngf, nc, ngpu).to(device)
    netG2 = Generator2(nx, ngf, nc, ngpu).to(device)
    
    # Apply weights initialization
    netE.apply(weights_init)
    netD.apply(weights_init)
    netG1.apply(weights_init)
    netG2.apply(weights_init)
    
    print(netE)
    print(netD)
    print(netG1)
    print(netG2)
    
    # Loss function
    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['beta1'], 0.999))
    optimizerG2 = optim.Adam(netG2.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['beta1'], 0.999))
    optimizerG1andE = optim.Adam([
        {'params': netG1.parameters(), 'lr': config['training']['learning_rate'], 'betas': (config['training']['beta1'], 0.999)},
        {'params': netE.parameters(), 'lr': config['training']['learning_rate'], 'betas': (config['training']['beta1'], 0.999)}
    ])
    
    # Lists to track losses
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting Training Loop...")
    
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            ###########################
            # Update D network
            ###########################
            netG1.zero_grad()
            netG2.zero_grad()
            netE.zero_grad()
            netD.zero_grad()
            
            # Train with real data
            reals = data.to(device)
            half_reals = reals[:,:,0:64]
            target_reals = reals[:,:,64:]
            b_size = reals.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(reals).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            # Train with fake data
            label.fill_(fake_label)
            z = torch.randn(b_size, nz, 1, device=device)
            mu, sigma = netE(half_reals)
            xz = mu + (z * sigma)
            half_fakes = netG1(xz)
            fakes = torch.cat([half_reals, half_fakes], 2)
            
            output = netD(fakes.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ###########################
            # Update G and E networks
            ###########################
            netG1.zero_grad()
            netG2.zero_grad()
            netE.zero_grad()
            netD.zero_grad()
            
            # Prepare labels and noise
            z = torch.randn(2 * b_size, nz, 1, device=device)
            mu = torch.cat([mu, mu], 0)
            sigma = torch.cat([sigma, sigma], 0)
            xz = mu + (z * sigma)
            half_fakes = netG1(xz)
            half_reals = torch.cat([half_reals,half_reals],0)
            target_reals =torch.cat([target_reals,target_reals],0)
            fakes = torch.cat([half_reals, half_fakes], 2)
            
            # Diversity loss
            z_diffs = torch.sum(torch.abs(z[0:b_size] - z[b_size:2*b_size]), dim=(1,2))
            i_diffs = torch.sum(torch.abs(half_fakes[0:b_size] - half_fakes[b_size:2*b_size]), dim=(1,2)) + 1e-10
            loss_diversity = torch.mean(z_diffs / i_diffs)
            w_diversity = config['weights']['diversity']
            w_netE = config['weights']['netE']
            diversity_loss = w_diversity * loss_diversity
            netE_loss = w_netE * netE.kl
            
            # Generator loss
            label = torch.full((2 * b_size,), real_label, dtype=torch.float, device=device)
            output = netD(fakes).view(-1)
            G_label = torch.full((2 * b_size,), real_label, dtype=torch.float, device=device)
            G_loss = criterion(output, G_label)
            errG = G_loss + diversity_loss + netE_loss
            errG.backward()
            optimizerG1andE.step()
            
            ###########################
            # Optimize Z
            ###########################
            z = torch.zeros(b_size, nz, 1, requires_grad=True, device=device)
            optimizerZ = optim.Adam([z], lr=config['evaluation']['z_lr'])
            criterionC = nn.L1Loss()
            half_reals = reals[:,:,0:64]
            target_reals = reals[:,:,64:]
            
            with torch.no_grad():
                mu, sigma = netE(half_reals)
            for Zepoch in range(config['evaluation']['z_epochs']):
                optimizerZ.zero_grad()
                xz = mu + (z * sigma)
                half_fakes = netG1(xz)
                output = criterionC(half_fakes, target_reals)
                output.backward()
                optimizerZ.step()
            
            ###########################
            # Update Generators and Encoder with optimal Z
            ###########################
            z = z.detach()
            netG1.zero_grad()
            netG2.zero_grad()
            netE.zero_grad()
            netD.zero_grad()
            
            mu, sigma = netE(half_reals)
            xz = mu + (z * sigma)
            half_fakes = netG1(xz)
            confidence_est = netG2(xz)
            confidence = torch.cat([half_reals, confidence_est], 2)
            
            residuals = torch.abs(target_reals - half_fakes)
            loss_confidence = criterionC(confidence_est, residuals)
            w_confidence = config['weights']['confidence']
            C_loss = w_confidence * loss_confidence
            C_loss.backward()
            optimizerG2.step()
            
            ###########################
            # Logging
            ###########################
            if i % 5 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] '
                      f'Loss_D: {errD.item():.4f} '
                      f'Loss_C: {C_loss.item():.4f} '
                      f'Loss_G: {errG.item():.4f} '
                      f'Loss_Diversity: {diversity_loss.item():.4f} '
                      f'Loss_NetE: {netE_loss.item():.4f}')
                
                # Plotting example
                plt.figure()
                plt.plot(confidence[5,0,:].detach().cpu().numpy(), label='Confidence')
                plt.plot(fakes[5,0,:].detach().cpu().numpy(), label='Fakes')
                plt.plot(reals[5,0,:].detach().cpu().numpy(), color='black', label='Reals')
                plt.legend()
                plt.show()
                plt.close()
            
            # Save losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            iters += 1
        
        # Optionally, save model checkpoints at each epoch
        os.makedirs('outputs/models', exist_ok=True)
        torch.save(netE.state_dict(), f'outputs/models/netE_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'outputs/models/netD_epoch_{epoch}.pth')
        torch.save(netG1.state_dict(), f'outputs/models/netG1_epoch_{epoch}.pth')
        torch.save(netG2.state_dict(), f'outputs/models/netG2_epoch_{epoch}.pth')
    
    # Save final models
    torch.save(netE.state_dict(), 'outputs/models/netE_final.pth')
    torch.save(netD.state_dict(), 'outputs/models/netD_final.pth')
    torch.save(netG1.state_dict(), 'outputs/models/netG1_final.pth')
    torch.save(netG2.state_dict(), 'outputs/models/netG2_final.pth')

if __name__ == "__main__":
    main()
