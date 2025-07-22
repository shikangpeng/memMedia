#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:24:59 2024

@author: Shikang Peng
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def compute_rdm(activations):
    dissimilarity_matrix = 1 - np.corrcoef(activations)
    return dissimilarity_matrix


# Load tensors
layers = torch.load('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Layers_reddit.pth')

rdms = {}

for layer, tensor_list in layers.items():
    # Create a list of numpy array
    layer_matrix = [tensor.detach().cpu().numpy().flatten() for tensor in tensor_list]

    # Convert the list of numpy arrays into a single numpy array
    layer_matrix_np = np.array(layer_matrix)

    # Compute the RDM and store it
    rdms[layer] = compute_rdm(layer_matrix_np)
    
    
# Plot the array
plt.figure(figsize=(10, 8))
plt.imshow(rdms['Res_109'], cmap='viridis') 
plt.title('Layer RDM')
plt.colorbar()  
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(layer_matrix_np, cmap='viridis')
plt.title('Layer Weights Matrix')
plt.colorbar()  
plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(layer_matrix_np, cmap='viridis') 
cbar = fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.04) 
cbar.set_label('Value') 

ax.set_title('Layer Weights Matrix')
plt.show()


mean_dissimilarities_per_image = {}

for layer, rdm in rdms.items():
    mean_dissimilarities_per_image[layer] = np.mean(np.ma.masked_equal(rdm, 0), axis=1)

dissimilarities = pd.DataFrame(mean_dissimilarities_per_image)
dissimilarities.to_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/meanDis_image.csv', index = False)

res_11 = rdms['Res_11']
np.fill_diagonal(res_11,0)
res_36 = rdms['Res_36']
np.fill_diagonal(res_36,0)
res_73 = rdms['Res_73']
np.fill_diagonal(res_73,0)
res_109 = rdms['Res_109']
np.fill_diagonal(res_109,0)
res_145 = rdms['Res_145']
np.fill_diagonal(res_145,0)
res_155 = rdms['Res_155']
np.fill_diagonal(res_155,0)

np.savetxt('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/rdms/res11_rdm.csv',res_11,delimiter=',')
np.savetxt('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/rdms/res36_rdm.csv',res_36,delimiter=',')
np.savetxt('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/rdms/res73_rdm.csv',res_73,delimiter=',')
np.savetxt('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/rdms/res109_rdm.csv',res_109,delimiter=',')
np.savetxt('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/rdms/res145_rdm.csv',res_145,delimiter=',')
np.savetxt('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/rdms/res155_rdm.csv',res_155,delimiter=',')





