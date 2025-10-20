# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:49:16 2024

@author: skrisliu

Full data training
ATC only

"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data 
from torch.utils.data import Dataset, DataLoader
import time

site = 'psu'
year = '2023'


if int(year)%4==0:
    NODS = 366
else:
    NODS = 365
    
timesleep = 0

MODEL = 'unet'

#%% cuda
print(torch.__version__)
print('CUDA:',torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#%%
x1, x2 = 0,256
y1, y2 = 0,256


#%% Load data
# Clear image close to zeros
fp = site + '/datacube/' + site + year + 'meanbands.npy'
ims = np.load(fp)
ims = np.transpose(ims,[2,0,1])
ims = ims/5000

# LST in Kelvin
fp = site + '/datacube/' + site + year + 'lsts.npy'
lsts = np.load(fp)
mask = lsts==0
lsts = lsts*0.00341802 + 149.0 - 273.15
lsts[mask] = 0

# cloud masks
fp = site + '/datacube/' + site + year + 'clearmasks.npy'
clearmasks = np.load(fp)
im1z, im1x, im1y = clearmasks.shape


#%% subset, 1134, 1467
ims = ims[:,x1:x2,y1:y2]
clearmasks = clearmasks[:,x1:x2,y1:y2]
lsts = lsts[:,x1:x2,y1:y2]

im1z, im1x, im1y = clearmasks.shape


#%%
era5lst_single = np.load(site + '/'+site+'2023lst_era5.npy')
era5lst = np.zeros((NODS, x2-x1, y2-y1), dtype=np.float32)
for i in range(x2-x1):
    for j in range(y2-y1):
        era5lst[:,i,j] = era5lst_single
       




#%%
# Sinusoidal timestep embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * (-np.log(10000.0) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# U-Net
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=365, time_emb_dim=128):
        super(SimpleUNet, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        B, _, H, W = x.shape
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        t_emb = t_emb.expand(-1, -1, H, W)                    # [B, 64, H, W]

        h = self.encoder(x)
        h = h + t_emb
        out = self.decoder(h)  # [B, 365, H, W]
        return out
    


class ATCUNet(nn.Module):
    def __init__(self, imz=7, height=100, width=100, nods=365, era5lst=None, device='cuda'):
        super(ATCUNet, self).__init__()
        self.height = height
        self.width = width
        self.nods = nods
        self.device = device
        self.era5lst = era5lst  # shape: [365, H, W] or None

        # ATC base parameters
        self.T0 = nn.Parameter(torch.randn(height, width))
        self.A = nn.Parameter(torch.randn(height, width))
        self.phi = nn.Parameter(torch.randn(height, width))
        self.w_era5 = nn.Parameter(torch.randn(height, width))

        # DOY tensor
        base = np.arange(nods) + 1
        base = np.tile(base[:, None, None], (1, height, width))  # shape: [365, H, W]
        self.register_buffer('base', torch.tensor(base, dtype=torch.float32))

        # Diffusion model for residual
        self.diffusion_model = SimpleUNet(in_channels=imz, out_channels=nods)
        # self.diffusion_model = CompactUNet(in_channels=imz, out_channels=nods)

    def atc_only(self):
        atc = self.T0 + self.A * 10 * torch.cos(
            2 * np.pi / self.nods * (self.base - self.phi * 100)
        )
        if self.era5lst is not None:
            atc = atc + self.w_era5 * self.era5lst.to(self.device)
        return atc  # [365, H, W]

    def forward(self, x, t, add_noise=False):
        """
        x: [B, imz=7, H, W]
        t: [B] timestep indices
        Returns: atc_output [B, 365, H, W], final prediction [B, 365, H, W]
        """

        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1, 7, H, W]

        if add_noise:
            x = x + torch.randn_like(x) * 0.1

        B = x.size(0)

        atc = self.atc_only().unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 365, H, W]
        residuals = self.diffusion_model(x, t)  # [B, 365, H, W]

        prediction = atc + residuals
        return atc, prediction


#%% data to torch
era5lst = torch.from_numpy(era5lst)
ims = ims.reshape([1,7,im1x,im1y])
ims = torch.from_numpy(ims)
clearmasks = torch.from_numpy(clearmasks)
lsts = torch.from_numpy(lsts)

## to gpu
era5lst = era5lst.to(device)
ims = ims.to(device)
clearmasks = clearmasks.to(device)
lsts = lsts.to(device)


#%% build model
if MODEL =='unet':
    model = ATCUNet(imz=7, height=im1x, width=im1y, nods=NODS, era5lst=era5lst, device='cuda')
    epoch = 700
model = model.to(device)
print(model)

#%%
newpath = 'savemodel'
if not os.path.exists(newpath):
    os.makedirs(newpath)

#%% Training 1
criterion = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
batch_size = 1

log_loss = []

save_count = 0  # for naming output files

for i in range(epoch):
    t = torch.randint(0, 1000, (batch_size,), device='cuda')    
    atc_out, y_pred = model(ims, t)
    outputs = y_pred[0]
    loss = criterion(outputs[clearmasks], lsts[clearmasks])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i, loss.item())
    log_loss.append(loss.item())
    if i%50==5:
        time.sleep(timesleep)
    
if True:
    if True:
        impre = outputs.detach().cpu().numpy()  # (NODS, H, W) assumed
        for day in range(NODS):
            day_path = f'{site}/save/doy{day+1:03d}'
            os.makedirs(day_path, exist_ok=True)
            np.save(f'{day_path}/prea{save_count:03d}.npy', impre[day, :, :])




































