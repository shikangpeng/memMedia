#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:34:37 2024

@author: Shikang Peng
"""

import numpy as np
import csv
import pandas as pd

# Load the specific fetched data for reddit images
df = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data3/reddit_images_data3.csv')
# Sort the data
df_sorted = df.sort_values(by='id').reset_index(drop=True)


# Load the binary stimulus idx
imgLabels = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data3/imageLabels.csv', header = None)
scene_idx = imgLabels.iloc[:,1].rename('scene_idx')
# Concatenate 
# Ensure the same index in both dataframes
scene_idx.index = df_sorted.index
reddit_imgs = pd.concat([df_sorted, scene_idx], axis=1)
reddit_scenes = reddit_imgs[reddit_imgs['scene_idx'] == 1]

reddit_scenes.to_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data3/reddit_scenes.csv', index = False)

