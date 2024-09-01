import pandas as pd
import requests
import os
from io import BytesIO
from PIL import Image
from ratelimit import limits, sleep_and_retry
import time

# Define rate limits
CALLS_PER_SECOND = 2
CALLS_PER_MINUTE = 60

@sleep_and_retry
@limits(calls=CALLS_PER_SECOND, period=2)
@limits(calls=CALLS_PER_MINUTE, period=60)
def rate_limited_request(url):
    response = requests.get(url)
    if response.status_code == 429:  # Too Many Requests
        retry_after = int(response.headers.get('Retry-After', 60))
        time.sleep(retry_after)
        return rate_limited_request(url)  # Retry the request
    return response

def fetch_pdb_image(pdb_id):
    # RCSB PDB image URL
    image_url = f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_assembly-1.jpeg"
    
    response = rate_limited_request(image_url)
    if response.status_code == 200:
        print(f"Retrieved image URL for {pdb_id}: {image_url}")
        return Image.open(BytesIO(response.content)), image_url
    else:
        print(f"Failed to retrieve image for {pdb_id}")
        return None, None

# Load your dataset
df = pd.read_csv('./split_pdb_db/chunk_002.csv')  # Replace with your actual file name

# Create a directory to store images
if not os.path.exists('pdb_images'):
    os.makedirs('pdb_images')

# Add new columns for image file paths and URLs if they don't exist
if 'Image_Path' not in df.columns:
    df['Image_Path'] = ''
if 'Image_URL' not in df.columns:
    df['Image_URL'] = ''

# Iterate through the DataFrame
for index, row in df.iterrows():
    if pd.isna(row['Image_URL']) or row['Image_URL'] == '':
        pdb_id = row['PDB_ID']
        if pdb_id != 'NOT':  # Skip entries without a valid PDB ID
            image, image_url = fetch_pdb_image(pdb_id)
            if image:
                image_path = f'pdb_images/{pdb_id}.jpeg'
                image.save(image_path)
                df.at[index, 'Image_Path'] = image_path
                df.at[index, 'Image_URL'] = image_url
            else:
                df.at[index, 'Image_Path'] = 'Not available'
                df.at[index, 'Image_URL'] = 'Not available'
        else:
            df.at[index, 'Image_Path'] = 'Not available'
            df.at[index, 'Image_URL'] = 'Not available'
    
    # Print progress and save every 10th entry
    if (index + 1) % 20 == 0:
        print(f"Processed {index + 1} entries...")
        df.to_csv('pdb_dataset_with_images_chunk_2.csv', index=False)
        print(f"Progress saved at entry {index + 1}")

# Save the final updated dataset
df.to_csv('pdb_dataset_with_images_chunk_2.csv', index=False)

print("Dataset processing completed and saved as 'pdb_dataset_with_images.csv'")
