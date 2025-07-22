#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 20:40:20 2025

@author: Shikang Peng
"""

import praw
import pandas as pd

all_data = pd.read_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/allData_titles.csv')
urls_to_fetch = list(all_data['post_url'])

# Create Reddit instance
reddit = praw.Reddit(
    client_id='Vgtmmo74wJzLW7yMyffMzg',
    client_secret='Fkn527RRCIIMCXcSONaSAURYkGpdng',
    user_agent='MacOS RedditImageResponses scraper by u/memMedia'
)

results = []
for url in urls_to_fetch:
    try:
        submission = reddit.submission(url=url)

        subreddit_name = submission.subreddit.display_name
        title = submission.title
        caption_length = len(title)
        image_url = submission.url


        results.append({
            "post_url": url,
            "subreddit": subreddit_name,
            "caption_length": caption_length,
        })

    except Exception as e:
        print(f"Error processing {url}: {e}")

        # Append NA row for this URL
        results.append({
            "post_url": url,
            "subreddit": "NA",
            "caption_length": "NA"
        })

df = pd.DataFrame(results)
df.to_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/allData_subreddits.csv')

final_data = pd.merge(all_data, df, how='inner')
final_data.to_csv('/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/allData_final.csv', index=False)



counts = df["subreddit"].value_counts()

print(counts)
