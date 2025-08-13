# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 11:36:53 2025

@author: skrisliu
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr

site = 'psu'




#%%
fp = site + '/save/doy'


ims = []

for i in range(1,366):
    fp0 = fp + format(i,'03d') + '/prea000.npy'
    im = np.load(fp0)
    ims.append(im)
    
#%%
ims = np.array(ims)
y_pre = ims[:,128,128]  # the center is psu site
y_gt = np.load('psu/psu2023lst_gt.npy')




#%%

# Remove entries where y_gt is NaN
mask = ~np.isnan(y_gt)
y_gt_clean = y_gt[mask]
y_pre_clean = y_pre[mask]

# Calculate metrics
mae = mean_absolute_error(y_gt_clean, y_pre_clean)
rmse = mean_squared_error(y_gt_clean, y_pre_clean, squared=False)
r2 = r2_score(y_gt_clean, y_pre_clean)
r, _ = pearsonr(y_gt_clean, y_pre_clean)
r2 = r**2
bias = np.mean(y_pre_clean - y_gt_clean)
n_samples = len(y_gt_clean)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Bias: {bias:.4f}")




#%% Density Plot with Metrics


xy = np.vstack([y_gt_clean, y_pre_clean])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = y_gt_clean[idx], y_pre_clean[idx], z[idx]

# Plot
plt.figure(figsize=(4.5,4),dpi=200)
# sc = plt.scatter(x+273.15, y+273.15, c=z, s=20, cmap='viridis', edgecolor='none')
sc = plt.scatter(x+273.15, y+273.15, s=20, alpha=0.5)

# 1:1 line
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
plt.plot([-100,500], [-100,500], '-', linewidth=1, color='gray')

# Labels and axes
plt.xlabel("in situ measurement (K)")
plt.ylabel("prediction (K)")
# plt.axis('equal')
# plt.grid(True)

# Annotated metrics with units
textstr = '\n'.join([
    f'N = {n_samples}',
    f'MAE = {mae:.2f} K',
    f'RMSE = {rmse:.2f} K',
    f'R² = {r2:.2f}',
    f'Bias = {bias:.2f} K'
])
plt.annotate(textstr, xy=(0.04, 0.96), xycoords='axes fraction',
             fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

plt.annotate('PSU', xy=(0.98, 0.02), xycoords='axes fraction',
             fontsize=10, ha='right', va='bottom')

plt.xlim(260,315)
plt.ylim(260,315)

plt.tight_layout()
plt.show()


#%%











