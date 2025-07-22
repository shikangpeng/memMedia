from google.cloud import vision
from google.oauth2 import service_account
import io
import os
import json

# Path to your service account key file
key_path = "/home/sheskapeng/google-cloud-vision-python/google-cloud-vision-key.json"

# Load the credentials and set the quota project
credentials = service_account.Credentials.from_service_account_file(
    key_path, quota_project_id="memomedia"
)

# Create a client with the updated credentials
client = vision.ImageAnnotatorClient(credentials=credentials)

def detect_labels(file_path):
    """Detects labels in the file and returns labels with confidence scores."""
    # The name of the image file to annotate
    file_name = os.path.abspath(file_path)
    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    label_descriptions_with_scores = [(label.description, label.score) for label in labels]
    return label_descriptions_with_scores

def process_images_in_folder(folder_path):
    """Processes all images in the given folder and saves the labels with confidence scores to a JSON file."""
    output_file = 'image_labels_with_scores.json'
    data = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                print(f'Processing {file_path}')  # Debugging print statement
                labels_with_scores = detect_labels(file_path)
                if labels_with_scores:
                    print(f'Labels for {file}: {labels_with_scores}')  # Debugging print statement
                    data[file] = [{"label": label, "score": score} for label, score in labels_with_scores]
                else:
                    print(f'No labels detected for {file}')  # Debugging print statement

    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f'Labels and confidence scores saved to {output_file}')

# Set the path to your local folder containing images
local_folder_path = "/home/shikang/google-cloud-vision-python/images"  # Update this to your actual folder path

process_images_in_folder(local_folder_path)