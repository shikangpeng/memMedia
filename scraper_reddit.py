#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:10:45 2024

@author: Shikang Peng
"""
import praw
import requests
import os
import random
from datetime import datetime
import csv
import time
import pandas as pd

# Setup the Reddit API client
reddit = praw.Reddit(
    client_id='Vgtmmo74wJzLW7yMyffMzg',
    client_secret='Fkn527RRCIIMCXcSONaSAURYkGpdng',
    user_agent='MacOS RedditImageResponses scraper by u/memMedia')

subreddits = ['pics', 'pic', 'Images']
data = []
target_per_subreddit = 600 // len(subreddits)
image_directory = '/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data4/reddit_images4'
os.makedirs(image_directory, exist_ok=True)

previous_data_path = '/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/reddit_images_Replication.csv'

if os.path.exists(previous_data_path):
    previous_data = pd.read_csv(previous_data_path)
    fetched_ids = set(previous_data['id'])
    fetched_urls = set(previous_data['url'])
else:
    fetched_ids = set()
    fetched_urls = set()

def download_image(url, file_path):
    max_retries = 3  # Maximum number of retry attempts
    retries = 0
    while retries < max_retries:
        try:
            print("Attempting to download image:", url)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                return True
            else:
                print(f"Failed to download {url}: HTTP {response.status_code}")
                retries += 1
                time.sleep(2)  # Wait for 2 seconds before retrying
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error during download {url}: {e}, retrying...")
            retries += 1
            time.sleep(2)  # Wait for 2 seconds before retrying
    return False

def fetch_valid_posts(subreddit, target_count):
    all_candidates = []
    for submission in subreddit.hot(limit=1000):
        if submission.id in fetched_ids or submission.url in fetched_urls:
            continue  # Skip posts that have already been fetched

        if submission.score < 5 or submission.num_comments < 5:
            continue

        if any(submission.url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            submission.comments.replace_more(limit=0)
            top_comments = sorted(submission.comments, key=lambda x: x.score, reverse=True)[:5]
            top_comments_data = [{
                'comment_id': comment.id,
                'comment_body': comment.body,
                'comment_score': comment.score
            } for comment in top_comments]

            post_date = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            all_candidates.append({
                'id': submission.id,
                'url': submission.url,
                'upvotes': submission.score,
                'comments': submission.num_comments,
                'date_posted': post_date,
                'file_path': '',  # Placeholder for the file path
                'top_comments': top_comments_data
            })

    selected_posts = random.sample(all_candidates, target_count) if len(all_candidates) > target_count else all_candidates
    return selected_posts

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    posts = fetch_valid_posts(subreddit, target_per_subreddit)
    for post_info in posts:
        file_path = os.path.join(image_directory, f"{post_info['id']}.jpg")
        if not os.path.exists(file_path) and download_image(post_info['url'], file_path):
            post_info['file_path'] = file_path
            data.append(post_info)
        else:
            print(f"Skipped downloading existing image from {post_info['url']}")


csv_file = '/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/reddit_Data/Fetch_Data4/reddit_images_data4.csv'
csv_columns = ['id', 'url', 'upvotes', 'comments', 'date_posted', 'file_path', 'top_comments']

try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data_item in data:
            writer.writerow(data_item)
except IOError:
    print("I/O error during CSV file writing")
