#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:03:53 2024

@author: Shikang Peng
"""

from resmem.model import ResMem, transformer
from resmem.analysis import SaveFeatures
import numpy as np
from PIL import Image
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

    
# setup resmem for prediction
model = ResMem(pretrained=True)
model.eval()

def layer_extract(image_path, layer):
    # use one image
    img = Image.open(image_path)
    img = img.convert('RGB')

    activations = model.resnet_activation(transformer(img).view(-1, 3, 227, 227), layer)
    return activations

def rdm(activations):
    import torch
    import numpy as np

    # Flatten the activations and convert them to numpy
    activations = torch.stack([tensor.detach().flatten() for tensor in activations]).numpy()

    # Compute the Pearson correlation matrix
    correlation_matrix = np.corrcoef(activations)

    # Convert the correlation matrix to a dissimilarity matrix (1 - correlation)
    dissimilarity_matrix = 1 - correlation_matrix

    return torch.tensor(dissimilarity_matrix, dtype=torch.float)




data1 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data1/reddit_scenes_scores.csv')
data2 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data2/reddit_scenes_scores.csv')
data3 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data3/reddit_scenes_scores.csv')
data =  pd.concat([data1,data2,data3], ignore_index=True)


image_paths = data['file_path']


####
#### Stage 1 ####
####

### ResNet_11 feature RDM ###å
Res_11 = []


for file_idx, file in enumerate(image_paths):
    image_path = f'{file}'
    print(f"Processing image {file_idx + 1}/{len(image_paths)}: {image_path} at Layer 11")
    activations = layer_extract(image_path, 10)
    Res_11.append(activations)
    
rdmRes_11 = rdm(Res_11)


# Plot the array
plt.imshow(rdmRes_11.numpy(), cmap='viridis')  # 'cmap' is the color map
plt.colorbar()  
plt.title('Res11 RDM')
plt.show()

####
#### Stage 2 ####
####

### ResNet_35 feature RDM ###
Res_36 = []

for file_idx,  file in enumerate(image_paths):
    image_path = f'{file}'
    print(f"Processing image {file_idx + 1}/{len(image_paths)}: {image_path} at Layer 36")
    activations = layer_extract(image_path, 35)
    Res_36.append(activations)
    
rdmRes_36 = rdm(Res_36)


# Plot the array
plt.imshow(rdmRes_36.numpy(), cmap='viridis')  # 'cmap' is the color map
plt.colorbar()  
plt.title('Res36 RDM')
plt.show()

####
#### Stage 3 ####
####
### ResNet_73 feature RDM ###
Res_73 = []

for file_idx,  file in enumerate(image_paths):
    image_path = f'{file}'
    print(f"Processing image {file_idx + 1}/{len(image_paths)}: {image_path} at Layer 73")
    activations = layer_extract(image_path, 72)
    Res_73.append(activations)
    
rdmRes_73 = rdm(Res_73)


# Plot the array
plt.imshow(rdmRes_73.numpy(), cmap='viridis')  # 'cmap' is the color map
plt.colorbar()  
plt.title('Res73 RDM')
plt.show()


### ResNet_109 feature RDM ###
Res_109 = []

for file_idx,  file in enumerate(image_paths):
    image_path = f'{file}'
    print(f"Processing image {file_idx + 1}/{len(image_paths)}: {image_path} at Layer 109")
    activations = layer_extract(image_path, 108)
    Res_109.append(activations)
    
rdmRes_109 = rdm(Res_109)


# Plot the array
plt.imshow(rdmRes_109.numpy(), cmap='viridis')  # 'cmap' is the color map
plt.colorbar()  
plt.title('Res109 RDM')
plt.show()


### ResNet_145 feature RDM ###
Res_145 = []

for file_idx,  file in enumerate(image_paths):
    image_path = f'{file}'
    print(f"Processing image {file_idx + 1}/{len(image_paths)}: {image_path} at Layer 145")
    activations = layer_extract(image_path, 144)
    Res_145.append(activations)
    
rdmRes_145 = rdm(Res_145)


# Plot the array
plt.imshow(rdmRes_145.numpy(), cmap='viridis')  # 'cmap' is the color map
plt.colorbar()  
plt.title('Res145 RDM')
plt.show()

####
#### Stage 4 ####
####

### ResNet_154 feature RDM ###å
Res_155 = []

for file_idx,  file in enumerate(image_paths):
    image_path = f'{file}'
    print(f"Processing image {file_idx + 1}/{len(image_paths)}: {image_path} at Layer 154")
    activations = layer_extract(image_path, 154)
    Res_155.append(activations)
    
rdmRes_155 = rdm(Res_155)


# Plot the array
plt.imshow(rdmRes_155.numpy(), cmap='viridis')  # 'cmap' is the color map
plt.colorbar()  
plt.title('Res155 RDM')
plt.show()


import torch

torch.save({
    'Res_11': Res_11,
    'Res_36': Res_36,
    'Res_73': Res_73,
    'Res_109': Res_109,
    'Res_145': Res_145,
    'Res_155': Res_155
}, '/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Layers_reddit.pth')


torch.save({
    'Res_11': rdmRes_11,
    'Res_36': rdmRes_36,
    'Res_73': rdmRes_73,
    'Res_109': rdmRes_109,
    'Res_145': rdmRes_145,
    'Res_155': rdmRes_155
}, '/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/rdms_reddit.pth')





















