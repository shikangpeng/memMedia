#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:32:58 2024

@author: Shikang
"""
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import re
import matplotlib.pyplot as plt
import os

# Load the pre-trained ResNet-152 model
resnet152 = models.resnet152(pretrained=True)

# Layers to extract 
layers_to_extract = [11, 36, 73, 109, 145, 155]

# Hook function to store activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks for the specified layers by iterating through the children of the model
conv_layer_list = []
layer_count = 0

# Iterate through model layers to register hooks for Conv2d layers
for name, module in resnet152.named_children():
    for sub_name, sub_module in module.named_modules():
        if isinstance(sub_module, nn.Conv2d):
            layer_count += 1
            conv_layer_list.append((f"{name}.{sub_name}", sub_module))

# Print out all the Conv2d layers and their indices to understand the model structure
print("List of all Conv2d layers in the model:")
for idx, (layer_name, _) in enumerate(conv_layer_list, start=1):
    print(f"Layer {idx}: {layer_name}")

# Register hooks for the specified layers in layers_to_extract
for idx in layers_to_extract:
    if idx - 1 < len(conv_layer_list):
        layer_name, layer_module = conv_layer_list[idx - 1]  # Adjust for 0-based indexing
        layer_module.register_forward_hook(get_activation(f"layer_{idx}"))
        print(f"Registered hook for layer {idx}: {layer_name}")
    else:
        print(f"Layer {idx} is out of range for the current model.")

# Transformer to preprocess the image
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract activations for a single image at a specific layer
def layer_extract(image_path, layer):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transformer(img).unsqueeze(0)  # Add batch dimension

    # Set model to evaluation mode and clear previous activations
    resnet152.eval()
    global activations
    activations = {}

    # Forward pass through the model
    with torch.no_grad():
        _ = resnet152(input_tensor)

    # Return activation of the specified layer
    layer_name = f"layer_{layer}"
    if layer_name in activations:
        return activations[layer_name]
    else:
        print(f"Layer {layer} not found in activations.")
        return None


def rdm(activations):
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    activations = torch.stack([tensor.detach().flatten() for tensor in activations]).numpy()
    
    similarity_matrix = cosine_similarity(activations)
    dissimilarity_matrix = 1 - similarity_matrix
    
    return torch.tensor(dissimilarity_matrix, dtype=torch.float)


# Load image paths from CSV files
# Including faces
# data1 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data1/reddit_scores.csv')
# data2 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data2/reddit_scores.csv')
# data3 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data3/reddit_scores.csv')
# data = pd.concat([data1, data2, data3], ignore_index=True)

# Including scenes only
data1 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data1/reddit_scenes_scores.csv')
data2 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data2/reddit_scenes_scores.csv')
data3 = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data3/reddit_scenes_scores.csv')
data = pd.concat([data1, data2, data3], ignore_index=True)


image_paths = data['file_path']

# Stage 1: Extract activations for Layer 11
Res_11 = []

for file_idx, file in enumerate(image_paths):
    image_path = f'{file}'
    print(f"Processing image {file_idx + 1}/{len(image_paths)}: {image_path} at Layer 11")
    activations = layer_extract(image_path, 11)
    if activations is not None:
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
    activations = layer_extract(image_path, 36)
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
    activations = layer_extract(image_path, 73)
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
    activations = layer_extract(image_path, 109)
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
    activations = layer_extract(image_path, 145)
    Res_145.append(activations)
    
rdmRes_145 = rdm(Res_145)
# Plot the array
plt.imshow(rdmRes_145.numpy(), cmap='viridis')  # 'cmap' is the color map
plt.colorbar()  
plt.title('Res145 RDM')
plt.show()

####
#### Stage 4 ####
####s

### ResNet_155 feature RDM ###å
Res_155 = []

for file_idx,  file in enumerate(image_paths):
    image_path = f'{file}'
    print(f"Processing image {file_idx + 1}/{len(image_paths)}: {image_path} at Layer 155")
    activations = layer_extract(image_path, 155)
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
}, '/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Layers_reddit_scenes_res.pth')


torch.save({
    'Res_11': rdmRes_11,
    'Res_36': rdmRes_36,
    'Res_73': rdmRes_73,
    'Res_109': rdmRes_109,
    'Res_145': rdmRes_145,
    'Res_155': rdmRes_155
}, '/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/rdms_reddit_scenes_res.pth')
