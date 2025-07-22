#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:05:49 2024

@author: Shikang Peng
"""

import praw
from datetime import datetime
import csv

# Setup the Reddit API client
reddit = praw.Reddit(
    client_id='XXXXXXXXXXX',
    client_secret='XXXXXXXXXXX',
    user_agent='XXXXXXXXXXX')

# List of Reddit post URLs to process
urls_to_fetch = [
    "https://www.reddit.com/r/example/comments/example_id1/post_title/",
    "https://www.reddit.com/r/example/comments/example_id2/post_title/",
]

# Initialize storage for results
fetched_ids = set()
fetched_urls = set()
all_candidates = []

for url in urls_to_fetch:
    try:
        # Fetch submission from URL
        submission = reddit.submission(url=url)

        # Skip posts that have already been processed
        if submission.id in fetched_ids or submission.url in fetched_urls:
            continue

        # Check if the submission is an image post
        if any(submission.url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            # Fetch top comments
            submission.comments.replace_more(limit=0)
            top_comments = sorted(submission.comments, key=lambda x: x.score, reverse=True)[:5]
            top_comments_data = [{
                'comment_id': comment.id,
                'comment_body': comment.body,
                'comment_score': comment.score
            } for comment in top_comments]

            # Format post date
            post_date = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')

            # Build the data structure
            all_candidates.append({
                'id': submission.id,
                'url': submission.url,
                'upvotes': submission.score,
                'comments': submission.num_comments,
                'date_posted': post_date,
                'file_path': '',  # Placeholder for the file path
                'top_comments': top_comments_data
            })

            # Mark post as fetched
            fetched_ids.add(submission.id)
            fetched_urls.add(submission.url)

    except Exception as e:
        print(f"Error processing URL {url}: {e}")

# Specify CSV output file and columns
csv_file = '/Users/lucian/Library/CloudStorage/Box-Box/memoMedia/object_images/reddit_images_data1.csv'
csv_columns = ['id', 'url', 'upvotes', 'comments', 'date_posted', 'file_path', 'top_comments']

# Save data to CSV
try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data_item in all_candidates:
            # Convert 'top_comments' to a string for CSV (e.g., JSON format)
            data_item['top_comments'] = str(data_item['top_comments'])
            writer.writerow(data_item)
    print(f"Data successfully saved to {csv_file}")
except IOError as e:
    print(f"I/O error during CSV file writing: {e}")
