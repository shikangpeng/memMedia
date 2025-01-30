#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:18:16 2024

@author: Shikang Peng
"""


from PIL import Image
from resmem import ResMem, transformer
import os
import pandas as pd

model = ResMem(pretrained=True)
model.eval()

def resmem_predict(image_path):
    # use one image
    img = Image.open(image_path).convert('RGB')
    
    image_x = transformer(img)
    # setup resmem for prediction
    prediction = model(image_x.view(-1, 3, 227, 227))
    return prediction.item()

###
### Reddit Usage ###
###
# Change filename for each fetched data
redditScores = pd.read_csv('/Users/lucian/library/cloudstorage/box-box/memoMedia/reddit_Data/Fetch_Data3/reddit_images_data3.csv')
imgPaths = redditScenes['file_path']

mem_scores = []

for file_idx, file in enumerate(imgPaths):
    print(f"Processing image {file_idx+1}/{len(redditScenes)}: {file}")
    prediction = resmem_predict(file)
    mem_scores.append(prediction)

redditScores['memorability_score'] = mem_scores

redditScores.to_csv('/Users/lucian/library/cloudstorage/box-box/memoMedia/reddit_Data/Fetch_Data3/reddit_scores.csv', index=False)



