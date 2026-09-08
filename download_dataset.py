"""
Script to download and extract the PetImages dataset.
This script handles downloading a zip file containing pet images and extracting them to the local filesystem.
"""

from io import BytesIO
from zipfile import ZipFile
import requests
import os
from tqdm import tqdm

# Check if dataset already exists
if not os.path.exists("PetImages"):
    print("Downloading dataset...")
    # Download the dataset from Google Cloud Storage
    response = requests.get("https://storage.googleapis.com/ai-fundamentals/data/petsimages.zip", stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create a BytesIO buffer to store the downloaded data
    buffer = BytesIO()

    with tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            buffer.write(data)
            bar.update(len(data))
    
    print("Extracting dataset...")
    buffer.seek(0)  # Reset buffer position to beginning for extraction
    with ZipFile(buffer) as zip_file:
        total_files = len(zip_file.namelist())
        with tqdm(total=total_files, desc="Extracting") as pbar:
            for file in zip_file.namelist():
                zip_file.extract(file)
                pbar.update(1)
    
    print("Dataset downloaded and extracted successfully.")
else:
    print("Dataset already exists.")